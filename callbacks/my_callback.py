from typing import Any

import torch
import matplotlib.pyplot as plt

from warprec.data import Dataset
from warprec.utils.callback import WarpRecCallback


class ComputeNDCGOverIterations(WarpRecCallback):
    """Custom example of implementation of a WarpRec custom callback.

    In this example we plot the nDCG value over the iterations
    using matplotlib.

    Args:
        *args (Any): The arguments of the callback.
        **kwargs (Any): The keyword arguments of the callback.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._save_path = kwargs.get("save_path", None)
        self._ndcg_scores: list[float] = []

    def on_trial_save(self, iteration, trials, trial, **info):
        ndcg_score = trial.last_result.get("score", 0.0)
        self._ndcg_scores.append(ndcg_score)

    def on_training_complete(self, model, *args, **kwargs):
        iterations = list(range(1, len(self._ndcg_scores) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self._ndcg_scores, marker="o", linestyle="-")

        plt.title("nDCG@5 over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("nDCG@5")
        plt.grid(True)
        plt.xticks(iterations)
        plt.tight_layout()

        if self._save_path:
            try:
                plt.savefig(self._save_path)
                print(f"Plot successfully saved to: {self._save_path}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error during the saving process in {self._save_path}: {e}")
            plt.close()
        else:
            plt.show()


class CustomStashCallback(WarpRecCallback):
    """Custom example of implementation of a WarpRec custom callback.

    In this example we add custom data to the Dataset stash.
    """

    def on_dataset_creation(self, main_dataset, validation_folds, *args, **kwargs):
        def compute_item_popularity(dataset: Dataset):
            interaction_matrix = dataset.train_set.get_sparse()
            item_popularity = interaction_matrix.sum(
                axis=0
            ).A1  # Sum over users to get item popularity
            tensor_popularity = torch.tensor(item_popularity, dtype=torch.float32)

            dataset.add_to_stash(
                "item_popularity", tensor_popularity
            )  # Stash the popularity tensor

        compute_item_popularity(main_dataset)

        if (
            validation_folds is not None and len(validation_folds) > 0
        ):  # Check if validation folds exist
            for fold in validation_folds:
                compute_item_popularity(fold)
