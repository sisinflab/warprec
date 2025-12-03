import random
from typing import Any, Optional
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch import nn, Tensor
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType


class Recommender(nn.Module, ABC):
    """Abstract class that defines the basic functionalities of a recommendation model.

    Args:
        params (dict): The dictionary with the model params.
        interactions (Interactions): The training interactions.
        info (dict): The dictionary containing dataset information.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        DATALOADER_TYPE (Optional[DataLoaderType]): The type of dataloader used
            by this model. This value will be used to pre-compute the required
            data structure before starting the training process.

    Raises:
        ValueError: If the info dictionary does not contain the number of items
            and users of the dataset.
    """

    DATALOADER_TYPE: Optional[DataLoaderType] = None

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__()
        self.init_params(params)
        self.set_seed(seed)

        # Initialize the dataset dimensions
        self.n_users = info.get("n_users")
        self.n_items = info.get("n_items")
        if not self.n_users or not self.n_items:
            raise ValueError(
                f"Incorrect initialization: 'n_users' ({self.n_users}) e 'n_items' ({self.n_items}) "
                "must be present in the 'info' dictionary."
            )

    @abstractmethod
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        user_seq: Optional[Tensor] = None,
        seq_len: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """This method will produce the final predictions in the form of
        a dense Tensor.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            user_seq (Optional[Tensor]): Padded sequences of item IDs for users to predict for.
            seq_len (Optional[Tensor]): Actual lengths of these sequences, before padding.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """

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

    @property
    def device(self) -> torch.device:
        """Get the device where the model is located.

        Returns:
            torch.device: The device of the model.
        """
        # Search through parameters
        try:
            return next(self.parameters()).device
        except StopIteration:
            pass

        # If no parameter found, search through buffers
        try:
            return next(self.buffers()).device
        except StopIteration:
            pass

        # Fallback: Device will be cpu
        return torch.device("cpu")


class IterativeRecommender(Recommender):
    """Interface for recommendation model that use
    an iterative approach to be trained.

    Attributes:
        epochs (int): The number of epochs used to
            train the model.
        learning_rate (float): The learning rate using
            during optimization.
    """

    epochs: int
    learning_rate: float

    def _init_weights(self, module: nn.Module):
        """A comprehensive default weight initialization method.
        This method is called recursively by `self.apply(self._init_weights)`
        and handles the most common layer types found in recommendation models.

        It can be overridden by subclasses for model-specific initialization.

        The default strategies are:
        - Xavier Normal for Linear, Embedding, and Convolutional layers.
        - Xavier Uniform for Recurrent layers (GRU, LSTM).
        - Identity-like initialization for LayerNorm.
        - Zeros for all biases.

        Args:
            module (nn.Module): The module to initialize.
        """
        # --- Layers with standard weight matrices ---
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            xavier_normal_(module.weight.data)
            if hasattr(module, "bias") and module.bias is not None:
                constant_(module.bias.data, 0)

        # --- Embedding Layer ---
        elif isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

        # --- Recurrent Layers ---
        elif isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in module.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    xavier_uniform_(param.data)
                elif "bias" in name:
                    constant_(param.data, 0)

        # --- Normalization Layers ---
        elif isinstance(module, nn.LayerNorm):
            constant_(module.bias.data, 0)
            constant_(module.weight.data, 1.0)

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

    def _generate_square_subsequent_mask(self, seq_len: int) -> Tensor:
        """Generate a square mask for the sequence.

        Args:
            seq_len (int): Length of the sequence.

        Returns:
            Tensor: A square mask of shape [seq_len, seq_len] with True for positions
                    that should not be attended to.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()


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
        self.n_items = info.get("n_items", None)
        if not self.n_items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        self.item_similarity = np.zeros(self.n_items)

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.

        Raises:
            ValueError: If 'train_batch' is not provided in kwargs.
        """
        # Get train batch from kwargs
        train_batch: Optional[csr_matrix] = kwargs.get("train_batch")
        if train_batch is None:
            raise ValueError(
                f"predict() for {self.name} requires 'train_batch' as a keyword argument."
            )

        # Compute predictions and convert to Tensor
        predictions = train_batch @ self.item_similarity  # pylint: disable=not-callable
        predictions = torch.from_numpy(predictions)

        # Return full or sampled predictions
        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, n_items]
        else:
            # Case 'sampled': prediction on a sampled set of items
            return predictions.gather(
                1,
                item_indices.to(predictions.device).clamp(
                    max=self.n_items - 1
                ),  # [batch_size, pad_seq]
            )
