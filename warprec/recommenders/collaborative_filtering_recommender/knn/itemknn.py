# pylint: disable = R0801, E1102
from typing import Any

import torch
from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="ItemKNN")
class ItemKNN(ItemSimRecommender):
    """Implementation of ItemKNN algorithm from
        Amazon.com recommendations: item-to-item collaborative filtering 2003.

    For further details, please refer to the `paper <http://ieeexplore.ieee.org/document/1167344/>`_.

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

        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X.T))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity
        self.item_similarity = filtered_sim_matrix.numpy()
