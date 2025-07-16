"""
This is a sample pipeline without the usage of a configuration file.
You can customize your process in this way, creating multiple reader/writer
or more training loops if needed.
This approach is advised for expert user of the framework.
"""

import os
import pandas as pd
from warprec.data.dataset import TransactionDataset
from warprec.recommenders.general_recommender import ProxyRecommender
from warprec.evaluation.evaluator import Evaluator

# Dataset creation parameters
SPLIT_DIR = "tests/test_proxy/"
SEPARATOR = "\t"
EXT = ".tsv"
SPLIT_NAMES = ["train", "test", "val"]
DATASET_KEYWORDS = ["train_data", "test_data", "val_data"]
BATCH_SIZE = 1024
RATING_TYPE = "explicit"

# Recommendation file parameters for ProxyRecommender
RECOMMENDATION_FILE = "tests/test_proxy/ItemKNN.tsv"
RECOMMENDATION_SEPARATOR = "\t"
HEADER = False

# Evaluation
METRICS = ["nDCG", "Precision", "Recall", "ItemCoverage", "Gini"]
TOP_K_VALUES = [10, 20, 50]
METRICS_PER_ROW = 8


def main():
    split_data = {}
    for split in SPLIT_NAMES:
        split_path = os.path.join(SPLIT_DIR, f"{split}{EXT}")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file {split_path} does not exist.")

        # Read the split files
        split_data[split] = pd.read_csv(split_path, sep=SEPARATOR)

    dataset_input = {k: v for k, v in zip(DATASET_KEYWORDS, split_data.values())}

    dataset = TransactionDataset(
        **dataset_input,
        batch_size=BATCH_SIZE,
        rating_type=RATING_TYPE,
    )

    model = ProxyRecommender(
        params={
            "recommendation_file": RECOMMENDATION_FILE,
            "separator": RECOMMENDATION_SEPARATOR,
            "header": HEADER,
        },
        info=dataset.info(),
    )

    model.fit(dataset.train_set)

    evaluator = Evaluator(
        metric_list=METRICS,
        k_values=TOP_K_VALUES,
        train_set=dataset.train_set.get_sparse(),
        side_information=None,
    )

    if dataset.val_set is not None:
        evaluator.evaluate(model, dataset, test_set=False)
        res_dict = evaluator.compute_results()
        evaluator.print_console(res_dict, "Validation", METRICS_PER_ROW)

    if dataset.test_set is not None:
        evaluator.evaluate(model, dataset, test_set=True)
        res_dict = evaluator.compute_results()
        evaluator.print_console(res_dict, "Test", METRICS_PER_ROW)


if __name__ == "__main__":
    main()
