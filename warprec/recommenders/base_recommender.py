import random
from typing import Any, Optional
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

from warprec.data.dataset import Interactions, Sessions
from warprec.utils.enums import DataLoaderType


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

    Attributes:
        DATALOADER_TYPE (Optional[DataLoaderType]): The type of dataloader used
            by this model. This value will be used to pre-compute the required
            data structure before starting the training process.
    """

    DATALOADER_TYPE: Optional[DataLoaderType] = None

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
        return self.__class__.__name__

    @property
    def name_param(self):
        """The name of the model with all it's parameters."""
        name = self.name
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

    @abstractmethod
    def predict_full(
        self,
        user_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """This method will produce the final predictions in the form of
        a dense Tensor.

        Args:
            user_indices (Tensor): The batch of user indices.
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """

    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """This method will produce predictions only of given item
        indices.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices.
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.

        Raises:
            NotImplementedError: If the model does not support sampled prediction.
        """
        raise NotImplementedError("This model does not support sampled prediction.")

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
        predictions = torch.from_numpy(predictions).to(self._device)
        predictions = predictions.gather(
            1, item_indices.clamp(min=0)
        )  # [batch_size, pad_seq]
        return predictions
