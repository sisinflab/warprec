# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="UserKNN")
class UserKNN(Recommender):
    """Implementation of UserKNN algorithm from
        GroupLens: an open architecture for collaborative filtering of netnews 1994.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/192844.192905>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items value was not passed through the info dict.

    Attributes:
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
    """

    k: int
    similarity: str

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, seed=seed, *args, **kwargs)
        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity
        self.user_similarity = filtered_sim_matrix.numpy()

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.

        Raises:
            ValueError: If the 'train_sparse' keyword argument is not provided.
        """
        # Get train batch from kwargs
        train_sparse: Optional[csr_matrix] = kwargs.get("train_sparse")
        if train_sparse is None:
            raise ValueError(
                "predict() for UserKNN requires 'train_sparse' as a keyword argument."
            )

        # Compute predictions and convert to Tensor
        predictions = self.user_similarity[user_indices.cpu(), :] @ train_sparse
        predictions = torch.from_numpy(predictions)

        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, num_items]

        # Case 'sampled': prediction on a sampled set of items
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(
                max=self.items - 1
            ),  # [batch_size, pad_seq]
        )
