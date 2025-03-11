from typing import Any
from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np
from torch import nn, Tensor
from scipy.sparse import csr_matrix
from pandas import DataFrame
from elliotwo.data.dataset import Interactions


class AbstractRecommender(nn.Module, ABC):
    """Abstract class that defines the basic functionalities of a recommendation model.

    Args:
        *args (Any): Argument for PyTorch nn.Module.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._name = ""

    @abstractmethod
    def fit(self, *args, **kwargs):
        """This method will train the model on the dataset."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """This method will return the prediction of the model.

        Args:
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """

    def get_recs(
        self, X: Interactions, umap_i: dict, imap_i: dict, k: int
    ) -> DataFrame:
        """This method turns the learned parameters into new
        recommendations in DataFrame format.

        Args:
            X (Interactions): The set that will be used to
                produce recommendations.
            umap_i (dict): The inverse mapping from index -> user_id.
            imap_i (dict): The inverse mapping from index -> item_id.
            k (int): The top k recommendation to be produced.

        Returns:
            DataFrame: A DataFrame containing the top k recommendations for each user.
        """
        # Extract information from model
        scores = self.forward(X.get_sparse())
        top_k_items = torch.topk(scores, k, dim=1).indices
        user_ids = torch.arange(scores.shape[0]).unsqueeze(1).expand(-1, k)
        recommendations = torch.stack((user_ids, top_k_items), dim=2).reshape(-1, 2)

        # Extract user and items idxs
        user_idxs = recommendations[:, 0].tolist()
        item_idxs = recommendations[:, 1].tolist()

        # Map them back to original labels
        user_label = [umap_i[idx] for idx in user_idxs]
        item_label = [imap_i[idx] for idx in item_idxs]

        # Zip array and turn it into DataFrame
        real_recs = np.array(list(zip(user_label, item_label)))
        recommendations = pd.DataFrame(real_recs)

        return recommendations

    @property
    def name(self):
        """The name of the model."""
        return self._name


class ItemSimilarityRecommender(AbstractRecommender):
    """ItemSimilarityRecommender implementation.

    A ItemSimilarityRecommender is a Collaborative Filtering recommendation model
    which learns a similarity matrix B and produces recommendations using the computation: X@B.

    Args:
        items (int): The number of items that will be learned.
        *args (Any): Argument for PyTorch nn.Module.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

    def __init__(
        self,
        items: int,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.item_similarity = nn.Parameter(torch.rand(items, items))

    def forward(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            interaction_matrix (csr_matrix): The interactions matrix
                that will be used to predict.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        r = interaction_matrix @ self.item_similarity.detach().numpy()  # pylint: disable=not-callable

        # Masking interaction already seen in train
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r)
