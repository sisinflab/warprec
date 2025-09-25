# pylint: disable = R0801, E1102
from typing import Any

import torch
from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="ItemKNN")
class ItemKNN(ItemSimRecommender):
    """Implementation of ItemKNN algorithm from
        Amazon.com recommendations: item-to-item collaborative filtering 2003.

    For further details, check the `paper <http://ieeexplore.ieee.org/document/1167344/>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
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
        interactions: Interactions,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(
            params, interactions, device=device, seed=seed, info=info, *args, **kwargs
        )
        self._name = "ItemKNN"

        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X.T))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity
        self.item_similarity = filtered_sim_matrix.numpy()
