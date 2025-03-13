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
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

    def __init__(
        self,
        params: dict,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.init_params(params)
        self._name = ""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any):
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

    def init_params(self, params: dict):
        """This method sets up the model with the correct parameters.

        Args:
            params (dict): The dictionary with the model params.
        """
        for ann, _ in self.__class__.__annotations__.items():
            setattr(self, ann, params[ann])

    @property
    def name(self):
        """The name of the model."""
        return self._name

    @property
    def name_param(self):
        """The name of the model with all it's parameters."""
        name = self._name
        for ann, _ in self.__class__.__annotations__.items():
            value = getattr(self, ann, None)
            name += f"_{ann}={value:.4f}"
        return name


def generate_model_name(model_name: str, params: dict) -> str:
    """
    Generate a model name string based on the model name and its parameters.

    Args:
        model_name (str): The base name of the model.
        params (dict): Dictionary containing parameter names and values.

    Returns:
        str: The formatted model name including parameters.
    """
    param_str = "_".join(f"{key}={value:.4f}" for key, value in params.items())
    return f"{model_name}_{param_str}"


class ItemSimilarityRecommender(AbstractRecommender):
    """ItemSimilarityRecommender implementation.

    A ItemSimilarityRecommender is a Collaborative Filtering recommendation model
    which learns a similarity matrix B and produces recommendations using the computation: X@B.

    Args:
        params (dict): The dictionary with the model params.
        items (int): The number of items that will be learned.
        *args (Any): Argument for PyTorch nn.Module.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

    def __init__(
        self,
        params: dict,
        items: int,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(params, *args, **kwargs)
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
