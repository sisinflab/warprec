# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import torch
from torch import nn
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import ItemSimRecommender
from elliotwo.utils.registry import model_registry, similarities_registry


@model_registry.register(name="AttributeItemKNN")
class AttributeItemKNN(ItemSimRecommender):
    """Implementation of AttributeItemKNN algorithm from
        MyMediaLite: A free recommender system library 2011.

    For further details, check the `paper <https://www.researchgate.net/publication/221141162_MyMediaLite_A_free_recommender_system_library>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
        normalize (bool): Wether or not to normalize the interactions.
    """

    k: int
    similarity: str
    normalize: bool

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
        self._name = "AttributeItemKNN"

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on the sparse representation of the features.
        During the train a similarity matrix {item x item} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        X_feat = interactions.get_side_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Apply normalization of features if requested
        if self.normalize:
            X_feat = self._normalize(X_feat)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X_feat))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity with a new nn.Parameter
        self.item_similarity = nn.Parameter(filtered_sim_matrix)

        if report_fn is not None:
            report_fn(self)
