# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import numpy as np
import torch
from torch import nn
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.dataset import Interactions
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="EsKNN")
class EsKNN(ItemSimRecommender):
    """

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        l2 (float): The normalization value.
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
        normalize (bool): Wether or not to normalize the interactions.
    """

    l2: float
    #k: int
    similarity: str
    #normalize: bool
    filtering: float

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
        self._name = "EsKNN"

    # IMPLEMENTATION 1
    # Classic ItemKNN -> EASE for auto encoding
    # def fit(
    #     self,
    #     interactions: Interactions,
    #     *args: Any,
    #     report_fn: Optional[Callable] = None,
    #     **kwargs: Any,
    # ):
    #     """
    #     Args:
    #         interactions (Interactions): The interactions that will be used to train the model.
    #         *args (Any): List of arguments.
    #         report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
    #         **kwargs (Any): The dictionary of keyword arguments.
    #     """
    #     X = interactions.get_sparse()
    #     similarity = similarities_registry.get(self.similarity)

    #     # Apply normalization of interactions if requested
    #     if self.normalize:
    #         X = self._normalize(X)

    #     # Compute similarity matrix
    #     sim_matrix = similarity.compute(X.T)

    #     # L2 penalization
    #     sim_matrix += self.l2 * np.identity(sim_matrix.shape[0])

    #     # Compute top_k filtering
    #     filtered_sim_matrix = self._apply_topk_filtering(torch.from_numpy(sim_matrix), self.k)

    #     # The rest of the EASE method
    #     B = np.linalg.inv(filtered_sim_matrix.numpy())
    #     B /= -np.diag(B)
    #     np.fill_diagonal(B, 0.0)

    #     self.item_similarity = nn.Parameter(torch.tensor(B, dtype=torch.float32))

    #     if report_fn is not None:
    #         report_fn(self)

    # IMPLEMENTATION 2
    # Compute similarity as for ItemKNN -> Apply EASE alg -> Apply top-k filtering
    # def fit(
    #     self,
    #     interactions: Interactions,
    #     *args: Any,
    #     report_fn: Optional[Callable] = None,
    #     **kwargs: Any,
    # ):
    #     """
    #     Args:
    #         interactions (Interactions): The interactions that will be used to train the model.
    #         *args (Any): List of arguments.
    #         report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
    #         **kwargs (Any): The dictionary of keyword arguments.
    #     """
    #     X = interactions.get_sparse()
    #     similarity = similarities_registry.get(self.similarity)

    #     # Apply normalization of interactions if requested
    #     if self.normalize:
    #         X = self._normalize(X)

    #     # Compute similarity matrix
    #     sim_matrix = similarity.compute(X.T)

    #     # L2 penalization
    #     sim_matrix += self.l2 * np.identity(sim_matrix.shape[0])

    #     # The rest of the EASE method
    #     B = np.linalg.inv(sim_matrix)
    #     B /= -np.diag(B)
    #     np.fill_diagonal(B, 0.0)

    #     # Compute top_k filtering
    #     filtered_sim_matrix = self._apply_topk_filtering(torch.from_numpy(B), self.k)

    #     self.item_similarity = nn.Parameter(torch.tensor(filtered_sim_matrix, dtype=torch.float32))

    #     if report_fn is not None:
    #         report_fn(self)

    # IMPLEMENTATION 3
    # Compute similarity -> Apply EASE alg [no knn]
    # def fit(
    #     self,
    #     interactions: Interactions,
    #     *args: Any,
    #     report_fn: Optional[Callable] = None,
    #     **kwargs: Any,
    # ):
    #     """
    #     Args:
    #         interactions (Interactions): The interactions that will be used to train the model.
    #         *args (Any): List of arguments.
    #         report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
    #         **kwargs (Any): The dictionary of keyword arguments.
    #     """
    #     X = interactions.get_sparse()
    #     similarity = similarities_registry.get(self.similarity)

    #     # Compute similarity matrix
    #     sim_matrix = similarity.compute(X.T)

    #     # L2 penalization
    #     sim_matrix += self.l2 * np.identity(sim_matrix.shape[0])

    #     # The rest of the EASE method
    #     B = np.linalg.inv(sim_matrix)
    #     B /= -np.diag(B)
    #     np.fill_diagonal(B, 0.0)

    #     self.item_similarity = nn.Parameter(torch.tensor(B, dtype=torch.float32))

    #     if report_fn is not None:
    #         report_fn(self)

    # IMPLEMENTATION 4
    # Compute similarity -> Apply EASE alg -> Apply top-k based on filtering
    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """
        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)
        filter_ratio = int(self.filtering * self.items)

        # Compute similarity matrix
        sim_matrix = similarity.compute(X.T)

        # L2 penalization
        sim_matrix += self.l2 * np.identity(sim_matrix.shape[0])

        # The rest of the EASE method
        B = np.linalg.inv(sim_matrix)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(torch.from_numpy(B), filter_ratio)

        self.item_similarity = nn.Parameter(torch.tensor(filtered_sim_matrix, dtype=torch.float32))

        if report_fn is not None:
            report_fn(self)
