# pylint: disable = R0801, E1102
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry


@model_registry.register(name="Random")
class Random(Recommender):
    """Definition of Random unpersonalized model.
    This model will recommend items based on a random number generator,
    ensuring that previously seen items are not recommended again.

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
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
        self._name = "Random"

    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction using a random number generator.

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Generate random predictions
        predictions = torch.rand(interaction_matrix.shape, device=self._device)

        # Mask seen items
        seen_mask = torch.tensor(interaction_matrix.toarray() != 0, device=self._device)
        predictions[seen_mask] = -torch.inf

        return predictions

    def fit(self, interactions, *args, report_fn, **kwargs):
        """Empty definition, not used by the Random model."""
        if report_fn is not None:
            report_fn(self)
        pass

    def forward(self, *args, **kwargs):
        """Empty definition, not used by the Random model."""
        pass
