import random
from typing import Callable, Optional, Any
from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np
from torch import nn, Tensor
from pandas import DataFrame
from scipy.sparse import csr_matrix
from warprec.data.dataset import Interactions


class Recommender(nn.Module, ABC):
    """Abstract class that defines the basic functionalities of a recommendation model.

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.init_params(params)
        self.set_seed(seed)
        self._device = torch.device(device)
        self._name = ""

    @abstractmethod
    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable],
        **kwargs: Any,
    ):
        """This method will train the model on the dataset.

        Args:
            interactions (Interactions): The interactions object used for the training.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any):
        """This method process a forward step of the model.

        All recommendation models that implement a neural network or any
        kind of backpropagation must implement this method, other model
        can leave this empty.

        Args:
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.
        """

    @abstractmethod
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """This method will produce the final predictions in the form of
        a dense Tensor.

        Args:
            interaction_matrix (csr_matrix): The sparse interaction matrix.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """

    def get_recs(
        self,
        X: Interactions,
        umap_i: dict,
        imap_i: dict,
        k: int,
        batch_size: int = 1024,
    ) -> DataFrame:
        """This method turns the learned parameters into new
        recommendations in DataFrame format, without column headers.

        Args:
            X (Interactions): The set that will be used to
                produce recommendations.
            umap_i (dict): The inverse mapping from index -> user_id.
            imap_i (dict): The inverse mapping from index -> item_id.
            k (int): The top k recommendation to be produced.
            batch_size (int): Number of users per batch

        Returns:
            DataFrame: A DataFrame (without header) containing the top k recommendations
                    for each user, including predicted ratings.
        """
        sparse_matrix = X.get_sparse()
        num_users = sparse_matrix.shape[0]
        all_recommendations = []

        for batch_start in range(0, num_users, batch_size):
            batch_end = min(batch_start + batch_size, num_users)

            # Process current batch
            batch_slice = slice(batch_start, batch_end)
            batch_scores = self.predict(
                sparse_matrix[batch_slice], start=batch_start, end=batch_end
            )

            # Get top-k items and their scores for current batch
            top_k_scores, top_k_items = torch.topk(batch_scores, k, dim=1)

            batch_users = (
                torch.arange(batch_start, batch_end).unsqueeze(1).expand(-1, k)
            )

            # Store batch recommendations with scores
            batch_recs = torch.stack(
                (batch_users, top_k_items, top_k_scores), dim=2
            ).reshape(-1, 3)
            all_recommendations.append(batch_recs)

        # Combine all batches
        recommendations = torch.cat(all_recommendations, dim=0)

        # Extract user, item indices and predicted scores
        user_idxs = recommendations[:, 0].tolist()
        item_idxs = recommendations[:, 1].tolist()
        pred_scores = recommendations[:, 2].tolist()

        # Map them back to original labels
        user_labels = [umap_i[idx] for idx in user_idxs]
        item_labels = [imap_i[idx] for idx in item_idxs]

        # Zip and turn into DataFrame (no header)
        real_recs = np.array(list(zip(user_labels, item_labels, pred_scores)))
        recommendations_df = pd.DataFrame(real_recs)

        return recommendations_df

    def init_params(self, params: dict):
        """This method sets up the model with the correct parameters.

        Args:
            params (dict): The dictionary with the model params.
        """
        for ann, _ in self.__class__.__annotations__.items():
            setattr(self, ann, params[ann])

    def get_params(self) -> dict:
        """Get the model parameters as a dictionary.

        Returns:
            dict: The dictionary containing the model parameters.
        """
        params = {}
        for ann, _ in self.__class__.__annotations__.items():
            params[ann] = getattr(self, ann)
        return params

    def set_seed(self, seed: int):
        """Set random seed for reproducibility.

        Args:
            seed (int): The seed value to be used.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _normalize(self, X: csr_matrix) -> csr_matrix:
        """Normalize matrix rows to unit length.

        Args:
            X (csr_matrix): The matrix to normalize.

        Returns:
            csr_matrix: The normalized matrix.
        """
        norms = np.sqrt(X.power(2).sum(axis=1))
        norms[norms == 0] = 1e-10
        return X.multiply(1 / norms)

    def _apply_topk_filtering(self, sim_matrix: Tensor, k: int) -> Tensor:
        """Keep only top-k similarities per item.

        Args:
            sim_matrix (Tensor): The similarity tensor to filter.
            k (int): The top k values to filter.

        Returns:
            Tensor: The filtered similarity tensor.
        """
        # Safety check for k size
        k = min(k, sim_matrix.size(1) - 1)

        # Get top-k values and indices
        values, indices = torch.topk(sim_matrix, k=k, dim=1)

        # Create sparse similarity matrix with top-k values
        return torch.zeros_like(sim_matrix).scatter_(1, indices, values)

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
            if isinstance(value, float):
                name += f"_{ann}={value:.4f}"
            else:
                name += f"_{ann}={value}"
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


"""
    In this section we have some common interfaces that Recommender model might use.
    If you want to implement your own Recommender model you can either use the
    Recommender class or one of the classes below. These function as a common
    starting point.
"""


class ItemSimRecommender(Recommender):
    """ItemSimilarity common interface.

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Raises:
        ValueError: If the items value was not passed through the info dict.
    """

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, *args, **kwargs)
        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        # Model initialization
        self.item_similarity = nn.Parameter(torch.rand(self.items, self.items))

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        r = interaction_matrix @ self.item_similarity.detach().numpy()  # pylint: disable=not-callable

        # Masking interaction already seen in train
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r).to(self._device)

    def forward(self, *args, **kwargs):
        """Forward method is empty because we don't need
        back propagation.
        """
