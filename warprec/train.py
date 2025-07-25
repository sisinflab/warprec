import argparse
import os
from argparse import Namespace

from warprec.data.reader import LocalReader
from warprec.data.writer import LocalWriter
from warprec.data.splitting import Splitter
from warprec.data.dataset import TransactionDataset
from warprec.utils.config import load_yaml, load_callback
from warprec.utils.logger import logger
from warprec.recommenders.trainer import Trainer
from warprec.recommenders.base_recommender import generate_model_name
from warprec.evaluation.evaluator import Evaluator


def main(args: Namespace):
    """Main function to start the experiment.

    This method will start the train pipeline.
    """
    logger.msg("Starting experiment.")

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
                    batch_size=config.general.batch_size,
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
                batch_size=config.general.batch_size,
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

    evaluator = Evaluator(
        list(config.evaluation.metrics),
        list(config.evaluation.top_k),
        train_set=dataset.train_set.get_sparse(),
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        feature_lookup=dataset.get_features_lookup(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
    )

    for model_name in models:
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
            config=config,
        )
        best_model, checkpoint_param = trainer.train_and_evaluate()

        # Callback on training complete
        callback.on_training_complete(model=best_model)

        # Evaluation testing
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
        evaluator.print_console(results, config.evaluation.max_metric_per_row)

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
                batch_size=config.general.batch_size,
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
