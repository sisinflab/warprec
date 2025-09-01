import torch

from warprec.data.dataset import Dataset
from warprec.utils.callback import WarpRecCallback


class CustomStashCallback(WarpRecCallback):
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
