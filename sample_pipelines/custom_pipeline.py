"""
This is a sample pipeline without the usage of a configuration file.
You can customize your process in this way, creating multiple reader/writer
or more training loops if needed.
This approach is advised for expert user of the framework.
"""

from warprec.data.dataset import Dataset
from warprec.data.reader import LocalReader
from warprec.data.splitting import Splitter
from warprec.recommenders.collaborative_filtering_recommender.knn import ItemKNN
from warprec.evaluation import Evaluator
from warprec.utils.enums import SplittingStrategies


def main():
    reader = LocalReader()
    data = reader.read_tabular(
        local_path="tests/test_dataset/movielens.csv",
        sep=",",
        column_names=["user_id", "item_id", "rating", "timestamp"],
    )

    splitter = Splitter()
    train, _, test = splitter.split_transaction(
        data, test_strategy=SplittingStrategies.TEMPORAL_HOLDOUT, test_ratio=0.1
    )

    dataset = Dataset(
        train_data=train,
        eval_data=test,
        rating_type="explicit",
        rating_label="rating",
    )

    # Initialize the model
    model = ItemKNN(
        params={
            "k": 100,
            "similarity": "cosine",
        },
        interactions=dataset.train_set,
        info=dataset.info(),
    )

    # Evaluation params
    metrics = ["nDCG", "Precision", "Recall", "HitRate"]
    cutoffs = [10, 20, 50]

    # Evaluate the model
    evaluator = Evaluator(
        metric_list=metrics, k_values=cutoffs, train_set=dataset.train_set.get_sparse()
    )
    evaluator.evaluate(
        model=model,
        dataloader=dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=dataset,
    )
    results = evaluator.compute_results()
    evaluator.print_console(results, "Sample Pipeline Evaluation")


if __name__ == "__main__":
    main()
