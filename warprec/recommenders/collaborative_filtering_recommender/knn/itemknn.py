# pylint: disable = R0801, E1102
from typing import Any, Optional

from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="ItemKNN")
class ItemKNN(ItemSimRecommender):
    """Implementation of ItemKNN algorithm from
        Amazon.com recommendations: item-to-item collaborative filtering 2003.

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
        n_items = info["n_items"]

        train_matrix_mb = cls._sparse_size_mb(X)
        similarity_peak_mb, _ = cls._topk_similarity_size_mb(
            side_len=n_items, k=params["k"], data_dtype=X.dtype
        )

        return {
            "train_ram_mb": train_matrix_mb + similarity_peak_mb,
            "notes": "ItemKNN analytical train-space estimate",
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
        super().__init__(params, info, interactions, *args, seed=seed, **kwargs)

        X = self.train_matrix
        similarity = similarities_registry.get(self.similarity)

        # Compute the top-k similarity blockwise, keeping it sparse throughout
        self.item_similarity = self._blockwise_topk_similarity(
            X.T.tocsr(), similarity, self.k
        )
