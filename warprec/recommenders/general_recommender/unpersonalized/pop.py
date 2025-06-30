# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.recommenders.base_recommender import Recommender
from warprec.data.dataset import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="Pop")
class Pop(Recommender):
    """Definition of Popularity unpersonalized model.
    This model will recommend items based on their popularity,
    ensuring that previously seen items are not recommended again.

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Raises:
        ValueError: If the items value was not passed through the info dict.
    """

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, info=info, *args, **kwargs)
        self._name = "Pop"

        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        self.popularity = torch.zeros(
            self.items, device=self._device, dtype=torch.float32
        )
        self.item_count = torch.tensor(0, device=self._device, dtype=torch.float32)

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        X = interactions.get_sparse()

        # Count the number of items to define the popularity
        self.popularity += torch.tensor(
            X.sum(axis=0).A1, device=self._device, dtype=torch.float32
        )
        # Count the total number of interactions
        self.item_count += torch.tensor(
            X.sum(), device=self._device, dtype=torch.float32
        )

        if report_fn is not None:
            report_fn(self)

    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction using a normalized popularity value.

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", interaction_matrix.shape[0])

        # Calculate batch_size
        batch_size = end_idx - start_idx

        # Repeat self.popularity to get a tensor of shape batch_size x num_items
        normalized_popularity = self.popularity / self.item_count
        predictions = normalized_popularity.repeat(batch_size, 1).to(self._device)

        # Mask seen items
        seen_mask = torch.tensor(interaction_matrix.toarray() != 0, device=self._device)
        predictions[seen_mask] = -torch.inf

        return predictions

    def forward(self, *args, **kwargs):
        """Empty definition, not used by the Pop model."""
        pass
