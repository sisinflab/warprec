import random
from typing import Any
from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from pandas import DataFrame
from scipy.sparse import coo_matrix, csr_matrix
from torch_sparse import SparseTensor
from tqdm import tqdm

from warprec.data.dataset import Dataset, Interactions, Sessions


class Recommender(nn.Module, ABC):
    """Abstract class that defines the basic functionalities of a recommendation model.

    Args:
        params (dict): The dictionary with the model params.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

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
        super().__init__()
        self.init_params(params)
        self.set_seed(seed)
        self._device = torch.device(device)
        self._name = ""

    @abstractmethod
    def predict_full(
        self,
        user_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """This method will produce the final predictions in the form of
        a dense Tensor.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """

    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """This method will produce predictions only of given item
        indices.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.

        Raises:
            NotImplementedError: If the model does not support sampled prediction.
        """
        raise NotImplementedError("This model does not support sampled prediction.")

    def get_recs(
        self,
        dataset: Dataset,
        k: int,
    ) -> DataFrame:
        """This method turns the learned parameters into new
        recommendations in DataFrame format, without column headers.

        Args:
            dataset (Dataset): The dataset that will be used to
                retrieve train data and mappings.
            k (int): The top k recommendation to be produced.

        Returns:
            DataFrame: A DataFrame (without header) containing the top k recommendations
                    for each user, including predicted ratings.
        """
        # Retrieve interaction data
        train_sparse = dataset.train_set.get_sparse()
        umap_i, imap_i = dataset.get_inverse_mappings()

        # Pre-allocate a Tensor with user indices
        num_users = train_sparse.shape[0]
        all_user_indices = torch.arange(num_users, device=self._device)
        batch_size = dataset._batch_size

        # Main evaluation loop
        all_recommendations = []
        for i in tqdm(range(0, num_users, batch_size)):
            # Extract the correct batch of user indices
            user_indices = all_user_indices[i : i + batch_size]

            # If we are evaluating a sequential model, compute user history
            user_seq, seq_len = None, None
            if isinstance(self, SequentialRecommenderUtils):
                user_seq, seq_len = dataset.train_session.get_user_history_sequences(
                    user_indices.tolist(),
                    self.max_seq_len,  # Sequence length truncated
                )

            predictions = self.predict_full(
                train_sparse=train_sparse,
                user_indices=user_indices,
                user_seq=user_seq,
                seq_len=seq_len,
            ).to(self._device)  # Get ratings tensor [batch_size x items]

            # Get top-k items and their scores for current batch
            top_k_scores, top_k_items = torch.topk(predictions, k, dim=1)
            batch_users = user_indices.unsqueeze(1).expand(-1, k)

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
        real_recs = np.array(
            list(zip(map(str, user_labels), map(str, item_labels), pred_scores))
        )
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


class IterativeRecommender(Recommender):
    """Interface for recommendation model that use
    an iterative approach to be trained.

    Attributes:
        optimizer (Optimizer): The optimizer used during the
            training process.
        epochs (int): The number of epochs used to
            train the model.
        learning_rate (float): The learning rate using
            during optimization.
        weight_decay (float): The l2 regularization applied
            to the model.
    """

    optimizer: Optimizer
    epochs: int
    learning_rate: float
    weight_decay: float

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any):
        """This method process a forward step of the model.

        All recommendation models that implement a neural network or any
        kind of backpropagation must implement this method.

        Args:
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.
        """

    @abstractmethod
    def get_dataloader(
        self, interactions: Interactions, sessions: Sessions, **kwargs: Any
    ) -> DataLoader:
        """Returns a PyTorch DataLoader for the given interactions.

        The DataLoader should provide batches suitable for the model's training.

        Args:
            interactions (Interactions): The interaction of users with items.
            sessions (Sessions): The sessions of the users,
            **kwargs (Any): Additional keyword arguments.

        Returns:
            DataLoader: The dataloader that will be used by the model during train.
        """

    @abstractmethod
    def train_step(self, batch: Any, epoch: int, *args: Any, **kwargs: Any) -> Tensor:
        """Performs a single training step for a given batch.

        This method should compute the forward pass, calculate the loss,
        and return the loss value.
        It should NOT perform zero_grad, backward, or step on the optimizer,
        as these will be handled by the generic training loop.

        Args:
            batch (Any): A single batch of data from the DataLoader.
            epoch (int): The current epoch iteration.
            *args (Any): The argument list.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tensor: The computed loss for the batch.
        """


class GraphRecommenderUtils(ABC):
    """Common definition for graph recommenders.

    Collection of common method used by all graph recommenders.
    """

    def get_adj_mat(
        self,
        interaction_matrix: coo_matrix,
        n_users: int,
        n_items: int,
        device: torch.device | str = "cpu",
    ) -> SparseTensor:
        """Get the normalized interaction matrix of users and items.

        Args:
            interaction_matrix (coo_matrix): The full interaction matrix in coo format.
            n_users (int): The number of users.
            n_items (int): The number of items.
            device (torch.device | str): Device to use for the adjacency matrix.

        Returns:
            SparseTensor: The sparse adjacency matrix.
        """
        # Extract user and items nodes
        user_nodes = interaction_matrix.row
        item_nodes = interaction_matrix.col + n_users

        # Unify arcs in both directions
        row = np.concatenate([user_nodes, item_nodes])
        col = np.concatenate([item_nodes, user_nodes])

        # Create the edge tensor
        edge_index_np = np.vstack([row, col])  # Efficient solution
        # Creating a tensor directly from a numpy array instead of lists
        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)

        # Create the SparseTensor using the edge indexes.
        # This is the format expected by LGConv
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(n_users + n_items, n_users + n_items),
        ).to(device)

        # LGConv will handle the normalization
        # so there is no need to do it here
        return adj

    def get_ego_embeddings(
        self, user_embedding: nn.Embedding, item_embedding: nn.Embedding
    ) -> Tensor:
        """Get the initial embedding of users and items and combine to an embedding matrix.

        Args:
            user_embedding (nn.Embedding): The user embeddings.
            item_embedding (nn.Embedding): The item embeddings.

        Returns:
            Tensor: Combined user and item embeddings.
        """
        user_embeddings = user_embedding.weight
        item_embeddings = item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings


class SequentialRecommenderUtils(ABC):
    """Common definition for sequential recommenders.

    Collection of common method used by all sequential recommenders.

    Attributes:
        max_seq_len (int): This value will be used to truncate user sequences.
            More recent transaction will have priority over older ones in case
            a sequence needs to be truncated. If a sequence is smaller than the
            max_seq_len, it will be padded.
    """

    max_seq_len: int = 0

    def _gather_indexes(self, output: Tensor, gather_index: Tensor) -> Tensor:
        """Gathers the output from specific indexes for each batch.

        Args:
            output (Tensor): The tensor to gather the indices from.
            gather_index (Tensor): The indices to gather.

        Returns:
            Tensor: The gathered values flattened.
        """
        gather_index = gather_index.view(-1, 1, 1).expand(-1, 1, output.shape[-1])
        output_flatten = output.gather(dim=1, index=gather_index)
        return output_flatten.squeeze(1)


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


class ItemSimRecommender(Recommender):
    """ItemSimilarity common interface.

    Args:
        params (dict): The dictionary with the model params.
        interactions (Interactions): The training interactions.
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
        interactions: Interactions,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(
            params, interactions, device=device, seed=seed, *args, **kwargs
        )
        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        self.item_similarity = np.zeros(self.items)

    @torch.no_grad()
    def predict_full(
        self,
        user_indices: Tensor,
        train_batch: csr_matrix,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            user_indices (Tensor): The batch of user indices.
            train_batch (csr_matrix): The batch of train sparse
                interaction matrix.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Compute predictions and convert to Tensor
        predictions = train_batch @ self.item_similarity  # pylint: disable=not-callable
        predictions = torch.from_numpy(predictions)
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        train_batch: csr_matrix,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        This method will produce predictions only for given item indices.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices to sample.
            train_batch (csr_matrix): The batch of train sparse
                interaction matrix.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.
        """
        # Compute predictions
        predictions = train_batch @ self.item_similarity  # pylint: disable=not-callable

        # Convert to Tensor and gather only required indices
        predictions = torch.from_numpy(predictions)
        predictions = predictions.gather(
            1, item_indices.clamp(min=0)
        )  # [batch_size, pad_seq]
        return predictions.to(self._device)
