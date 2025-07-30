import argparse
import os
import time
from argparse import Namespace

import ray

from warprec.data.reader import LocalReader
from warprec.data.writer import LocalWriter
from warprec.data.splitting import Splitter
from warprec.data.dataset import TransactionDataset
from warprec.data.filtering import apply_filtering
from warprec.utils.config import load_yaml, load_callback
from warprec.utils.logger import logger
from warprec.recommenders.trainer import Trainer
from warprec.recommenders.base_recommender import generate_model_name
from warprec.evaluation.evaluator import Evaluator
from warprec.evaluation.statistical_significance import compute_paired_statistical_test


def main(args: Namespace):
    """Main function to start the experiment.

    This method will start the train pipeline.
    """
    logger.msg("Starting experiment.")
    experiment_start_time = time.time()

    # Config parser testing
    config = load_yaml(args.config)

    # Load custom callback if specified
    callback = load_callback(
        config.general.callback,
        *config.general.callback.args,
        **config.general.callback.kwargs,
    )

    # Writer module testing
    writer = LocalWriter(config=config)

    # Reader module testing
    reader = LocalReader(config)

    # Dataset loading
    dataset = None
    side_data = None
    user_cluster = None
    item_cluster = None
    if config.reader.loading_strategy == "dataset":
        data = reader.read()

        # Check for optional filtering
        if config.filtering is not None:
            filters = config.get_filters()
            data = apply_filtering(data, filters)

        # Side information reading
        if config.reader.side:
            side_data = reader.read_side_information()

        # Cluster information reading
        if config.reader.clustering:
            user_cluster, item_cluster = reader.read_cluster_information()

        # Splitter testing
        if config.splitter:
            splitter = Splitter(config)

            if config.reader.data_type == "transaction":
                train, test, val = splitter.split_transaction(data)
                dataset = TransactionDataset(
                    train,
                    test,
                    val,
                    side_data=side_data,
                    user_cluster=user_cluster,
                    item_cluster=item_cluster,
                    batch_size=config.evaluation.batch_size,
                    rating_type=config.reader.rating_type,
                    rating_label=config.reader.labels.rating_label,
                    timestamp_label=config.reader.labels.timestamp_label,
                    cluster_label=config.reader.labels.cluster_label,
                    need_session_based_information=config.need_session_based_information,
                    precision=config.general.precision,
                )
            else:
                raise ValueError("Data type not yet supported.")
            # This branch is for 100% train and 0% test
            pass

    elif config.reader.loading_strategy == "split":
        if config.reader.data_type == "transaction":
            train, test, val = reader.read_transaction_split()

            # Side information reading
            if config.reader.side:
                side_data = reader.read_side_information()

            # Cluster information reading
            if config.reader.clustering:
                user_cluster, item_cluster = reader.read_cluster_information()

            dataset = TransactionDataset(
                train,
                test,
                val,
                side_data=side_data,
                user_cluster=user_cluster,
                item_cluster=item_cluster,
                batch_size=config.evaluation.batch_size,
                rating_type=config.reader.rating_type,
                rating_label=config.reader.labels.rating_label,
                timestamp_label=config.reader.labels.timestamp_label,
                cluster_label=config.reader.labels.cluster_label,
                need_session_based_information=config.need_session_based_information,
                precision=config.general.precision,
            )
        else:
            raise ValueError("Data type not yet supported.")

    # Callback on dataset creation
    callback.on_dataset_creation(
        dataset=dataset,
        train_set=train,
        test_set=test,
        val_set=val,
    )

    if config.splitter and config.writer.save_split:
        writer.write_split(dataset)

    # Trainer testing
    models = list(config.models.keys())

    # If statistical significance is required, metrics will
    # be computed user_wise
    requires_stat_significance = (
        config.evaluation.stat_significance.requires_stat_significance()
    )
    if requires_stat_significance:
        logger.attention(
            "Statistical significance is required, metrics will be computed user-wise."
        )
        model_results = {}

    evaluator = Evaluator(
        list(config.evaluation.metrics),
        list(config.evaluation.top_k),
        train_set=dataset.train_set.get_sparse(),
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        compute_per_user=requires_stat_significance,
        feature_lookup=dataset.get_features_lookup(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
    )

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )
    model_timing_report = []
    # Before starting training process, initialize Ray
    ray.init(runtime_env={"py_modules": config.general.custom_models})

    for model_name in models:
        model_exploration_start_time = time.time()

        params = config.models[model_name]
        val_metric, val_k = config.validation_metric(
            params["optimization"]["validation_metric"]
        )
        trainer = Trainer(
            model_name,
            params,
            dataset,
            val_metric,
            val_k,
            beta=config.evaluation.beta,
            pop_ratio=config.evaluation.pop_ratio,
            ray_verbose=config.general.ray_verbose,
            custom_callback=callback,
            custom_models=config.general.custom_models,
            config=config,
        )
        best_model, checkpoint_param, avg_train_time = trainer.train_and_evaluate()
        model_exploration_total_time = time.time() - model_exploration_start_time

        # Callback on training complete
        callback.on_training_complete(model=best_model)

        # Evaluation testing
        model_evaluation_start_time = time.time()
        eval_validation = dataset.val_set is not None
        eval_test = dataset.test_set is not None
        evaluator.evaluate(
            best_model,
            dataset,
            evaluate_on_validation=eval_validation,
            evaluate_on_test=eval_test,
            verbose=True,
        )
        results = evaluator.compute_results()
        model_evaluation_total_time = time.time() - model_evaluation_start_time
        evaluator.print_console(results, config.evaluation.max_metric_per_row)

        if requires_stat_significance:
            model_results[model_name] = (
                results  # Populate model_results for statistical significance
            )

        # Callback after complete evaluation
        callback.on_evaluation_complete(
            model=best_model,
            params=params,
            results=results,
        )

        # Write results of current model
        writer.write_results(
            results,
            model_name,
        )

        # Recommendation
        if params["meta"]["save_recs"]:
            umap_i, imap_i = dataset.get_inverse_mappings()
            recs = best_model.get_recs(
                dataset.train_set,
                umap_i,
                imap_i,
                k=config.writer.recommendation.k,
                batch_size=config.evaluation.batch_size,
            )
            writer.write_recs(recs, model_name)

        # Save params
        model_params = {model_name: best_model.get_params()}
        writer.write_params(model_params)

        # Model serialization
        if params["meta"]["save_model"]:
            writer.write_model(best_model)

            if params["meta"]["keep_all_ray_checkpoints"]:
                for check_path, param in checkpoint_param:
                    if os.path.exists(check_path):
                        source_path = os.path.join(check_path, "checkpoint.pt")
                        checkpoint_name = generate_model_name(model_name, param)
                        writer.checkpoint_from_ray(source_path, checkpoint_name)

        # Timing report for the current model
        model_timing_report.append(
            {
                "Model_Name": model_name,
                "Data_Preparation_Time": data_preparation_time,
                "Hyperparameter_Exploration_Time": model_exploration_total_time,
                "Average_Training_Time": avg_train_time,
                "Evaluation_Time": model_evaluation_total_time,
                "Total_Time": model_exploration_total_time
                + model_evaluation_total_time,
            }
        )

    if config.general.time_report:
        writer.write_time_report(model_timing_report)

    if requires_stat_significance:
        logger.msg(
            f"Computing statistical significance tests for {len(models)} models."
        )

        stat_significance = config.evaluation.stat_significance.model_dump(
            exclude=["corrections"]  # type: ignore[arg-type]
        )
        corrections = config.evaluation.stat_significance.corrections.model_dump()

        for stat_name, enabled in stat_significance.items():
            if enabled:
                test_results = compute_paired_statistical_test(
                    model_results, stat_name, **corrections
                )
                writer.write_statistical_significance_test(test_results, stat_name)

        logger.positive("Statistical significance tests completed successfully.")
    logger.positive(
        "All models trained and evaluated successfully. WarpRec is shutting down."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="store",
        required=True,
        help="Config file local path",
    )
    args = parser.parse_args()
    main(args)
