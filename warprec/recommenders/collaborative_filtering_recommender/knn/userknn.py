# pylint: disable = R0801, E1102
from typing import Any, Optional

from torch import Tensor

from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="UserKNN")
class UserKNN(Recommender):
    """Implementation of UserKNN algorithm from
        GroupLens: an open architecture for collaborative filtering of netnews 1994.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
    """

    k: int
    similarity: str

    @classmethod
    def estimate_space(
        cls,
        params: dict,
        info: dict,
        interactions: Optional[Interactions] = None,
        **kwargs: Any,
    ) -> dict:
        interactions = cls._require_interactions_for_estimate(
            interactions, cls.__name__
        )
        X = interactions.get_sparse()
        n_users = info["n_users"]

        train_matrix_mb = cls._sparse_size_mb(X)
        similarity_peak_mb, _ = cls._topk_similarity_size_mb(
            side_len=n_users, k=params["k"], data_dtype=X.dtype
        )

        return {
            "train_ram_mb": train_matrix_mb + similarity_peak_mb,
            "notes": "UserKNN analytical train-space estimate",
        }

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # Store the training matrix for prediction
        self.train_matrix = interactions.get_sparse()

        X = self.train_matrix
        similarity = similarities_registry.get(self.similarity)

        # Compute the top-k similarity blockwise, keeping it sparse throughout
        self.user_similarity = self._blockwise_topk_similarity(X, similarity, self.k)

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
        """
        # Compute predictions and convert to Tensor. The similarity matrix may
        # be stored sparsely, which keeps the product sparse until densified.
        rows = user_indices.cpu().tolist()
        predictions = self._as_dense_tensor(
            self.user_similarity[rows, :] @ self.train_matrix
        )

        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(
                max=self.n_items - 1
            ),  # [batch_size, pad_seq]
        )
