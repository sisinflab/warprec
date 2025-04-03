from elliotwo.data import LocalReader, Splitter, TransactionDataset
from elliotwo.recommenders import Trainer
from elliotwo.evaluation import Evaluator


def main():
    reader = LocalReader()
    data = reader.read("tests/test_dataset/movielens.csv", sep=",")

    splitter = Splitter()
    train, test, val = splitter.split_transaction(
        data, strategy="temporal", test_ratio=0.1, val_ratio=0.1
    )

    dataset = TransactionDataset(
        train, test, val, batch_size=1024, rating_type="explicit"
    )

    # Tuning parameters
    model = "EASE"
    params = {
        "l2": ["grid", 5, 10, 100, 500],
        "meta": {"implementation": "elliot"},
    }
    val_metric = "nDCG"
    top_k = 5
    ######################

    trainer = Trainer(
        model_name=model,
        param=params,
        dataset=dataset,
        metric_name=val_metric,
        top_k=top_k,
    )
    best_model, _ = trainer.train_and_evaluate()

    # Evaluation params
    metrics = ["nDCG", "Precision", "Recall", "HitRate"]
    cutoffs = [10, 20, 50]
    result_dict = {}

    evaluator = Evaluator(
        metric_list=metrics, k_values=cutoffs, train_set=dataset.train_set.get_sparse()
    )

    # Test
    evaluator.evaluate(model=best_model, dataset=dataset, test_set=True, verbose=True)
    results = evaluator.compute_results()
    evaluator.print_console(results, metrics, "Test")
    result_dict["Test"] = results

    # Validation
    evaluator.evaluate(model=best_model, dataset=dataset, test_set=False, verbose=True)
    results = evaluator.compute_results()
    evaluator.print_console(results, metrics, "Validation")
    result_dict["Validation"] = results


if __name__ == "__main__":
    main()
