# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
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
        similarity_matrix_mb = cls._dense_size_mb((n_items, n_items), X.dtype)

        return {
            "train_ram_mb": train_matrix_mb + similarity_matrix_mb,
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

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X.T))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity
        self.item_similarity = filtered_sim_matrix.numpy()
