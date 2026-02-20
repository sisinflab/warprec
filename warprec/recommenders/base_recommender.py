# pylint: disable = unused-argument
import random
from typing import Any, Optional, List, Dict, no_type_check
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
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__()
        self.init_params(params)
        self.set_seed(seed)
        self.info = info

        # Initialize the dataset dimensions
        self.n_users = info.get("n_users")
        self.n_items = info.get("n_items")
        if not self.n_users or not self.n_items:
            raise ValueError(
                f"Incorrect initialization: 'n_users' ({self.n_users}) e 'n_items' ({self.n_items}) "
                "must be present in the 'info' dictionary."
            )

    @no_type_check
    @abstractmethod
    def predict(
        self,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """This method will produce the final predictions in the form of
        a dense Tensor.

        Args:
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
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
            if ann in params:
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

    def get_state(self) -> Dict[str, Any]:
        """Returns the enriched state_dict of the WarpRec model.

        The returned dictionary contains all the information required to
        fully restore the model state, including additional metadata
        beyond the standard PyTorch state_dict.

        Returns:
            Dict[str, Any]: An enriched dictionary representing the model state.
        """
        state = {
            "name": self.name,
            "params": self.get_params(),
            "info": self.info,
            "state_dict": self.state_dict(),
        }
        return state

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Any, strict: bool = True, **kwargs: Any
    ) -> "Recommender":
        """Load a WarpRec checkpoint model state with custom parameters.

        Args:
            checkpoint (Any): The checkpoint containing the model state and
                other parameter required for initialization.
            strict (bool): Wether or not to load the model using strict mode.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            Recommender: The Recommender model instance.

        Raises:
            ValueError: When trying to load a model checkpoint of a different model.
        """
        if checkpoint["name"] != cls.__name__:
            raise ValueError(
                f"Warning: Loading a {checkpoint['name']} checkpoint into {cls.__name__} class."
            )

        # Common initialization params + additional parameters
        init_args = {
            "params": checkpoint["params"],
            "info": checkpoint["info"],
            **kwargs,
        }

        # Initialize the model and return the instance
        model = cls(**init_args)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model

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
        # Layers with standard weight matrices
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            xavier_normal_(module.weight.data)
            if hasattr(module, "bias") and module.bias is not None:
                constant_(module.bias.data, 0)

        # Embedding Layer
        elif isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

        # Recurrent Layers
        elif isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in module.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    xavier_uniform_(param.data)
                elif "bias" in name:
                    constant_(param.data, 0)

        # Normalization Layers
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
            sessions (Sessions): The sessions of the users.
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


class ContextRecommenderUtils(nn.Module, ABC):
    """Common definition for context-aware recommenders.

    This Mixin handles:
        1. Initialization of context dimensions.
        2. Creation of standard Biases (Global, User, Item, Context).
        3. Creation of Context Embeddings (to avoid boilerplate loops in models).
        4. Helper methods for Linear computation and Regularization.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        interactions (Optional[Interactions]): The training interactions.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        n_users (int): Number of users.
        n_items (int): Number of items.
        embedding_size (int): The size of the latent vectors.
        batch_size (int): The batch size used for training.
        neg_samples (int): Number of negative samples for training.
        merged_feature_embedding (Optional[nn.Embedding]): Single feature embedding.
        merged_feature_bias (Optional[nn.Embedding]): Single feature bias.
        feature_offsets (Optional[Tensor]): Offset buffer to index the single embedding.
        merged_context_embedding (Optional[nn.Embedding]): Single context embedding.
        merged_context_bias (Optional[nn.Embedding]): Single context bias.
        context_offsets (Optional[Tensor]): Offset buffer to index the single context.
    """

    # Type hints used in general mixin implementations
    n_users: int
    n_items: int
    embedding_size: int
    batch_size: int
    neg_samples: int

    # Explicit Type Hinting for Dynamic Attributes to fix Linting errors
    merged_feature_embedding: Optional[nn.Embedding]
    merged_feature_bias: Optional[nn.Embedding]
    feature_offsets: Optional[Tensor]

    merged_context_embedding: Optional[nn.Embedding]
    merged_context_bias: Optional[nn.Embedding]
    context_offsets: Optional[Tensor]

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        interactions: Optional[Interactions] = None,
        **kwargs: Any,
    ):
        # Feature info extraction
        self.feature_dims: dict = info.get("feature_dims", {})
        self.feature_labels = list(self.feature_dims.keys())

        # Context info extraction
        self.context_dims: dict = info.get("context_dims", {})
        self.context_labels = list(self.context_dims.keys())

        # Call super init to populate n_users, n_items, embedding_size
        super().__init__(params, info, *args, **kwargs)  # type: ignore[call-arg]

        # Define Embeddings (Latent Factors)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Context Embeddings
        if self.context_dims:
            ctx_dims_list = [self.context_dims[name] for name in self.context_labels]
            self.total_ctx_dim = sum(ctx_dims_list)

            # Offsets: [0, dim_0, dim_0+dim_1, ...]
            ctx_offsets = torch.tensor([0] + ctx_dims_list[:-1]).cumsum(0)
            self.register_buffer("context_offsets", ctx_offsets)

            self.merged_context_embedding = nn.Embedding(
                self.total_ctx_dim, self.embedding_size
            )
            self.merged_context_bias = nn.Embedding(self.total_ctx_dim, 1)
        else:
            self.register_buffer("context_offsets", None)
            self.merged_context_embedding = None
            self.merged_context_bias = None

        # Feature Embeddings
        if self.feature_dims:
            feat_dims_list = [self.feature_dims[name] for name in self.feature_labels]
            self.total_feat_dim = sum(feat_dims_list)

            feat_offsets = torch.tensor([0] + feat_dims_list[:-1]).cumsum(0)
            self.register_buffer("feature_offsets", feat_offsets)

            self.merged_feature_embedding = nn.Embedding(
                self.total_feat_dim, self.embedding_size
            )
            self.merged_feature_bias = nn.Embedding(self.total_feat_dim, 1)
        else:
            self.register_buffer("feature_offsets", None)
            self.merged_feature_embedding = None
            self.merged_feature_bias = None

        # Define Biases (Standard Linear Infrastructure)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items + 1, 1, padding_idx=self.n_items)

        # Fixed feature lookup Tensor
        if interactions is not None:
            item_features = interactions.get_side_tensor()
            self.register_buffer("item_features", item_features)
        else:
            self.register_buffer(
                "item_features", torch.zeros(self.n_items + 1, dtype=torch.long)
            )

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ) -> DataLoader:
        """Common dataloader retrieval used by contextual models.

        Args:
            interactions (Interactions): The interaction of users with items.
            sessions (Sessions): The sessions of the users.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            DataLoader: The appropriate dataloader for the training.
        """
        return interactions.get_pointwise_dataloader(
            neg_samples=self.neg_samples,
            include_side_info=bool(self.feature_dims),
            include_context=bool(self.context_dims),
            batch_size=self.batch_size,
            **kwargs,
        )

    def compute_first_order(
        self,
        user: Tensor,
        item: Tensor,
        features: Optional[Tensor],
        contexts: Optional[Tensor],
    ) -> Tensor:
        """Computes the First-Order Linear part.

        Formula: global_bias + user_bias + item_bias + sum(feature_biases) + sum(context_biases)

        Args:
            user (Tensor): User indices.
            item (Tensor): Item indices.
            features (Optional[Tensor]): Feature indices [batch_size, n_features].
            contexts (Optional[Tensor]): Context indices [batch_size, n_contexts].

        Returns:
            Tensor: The linear score [batch_size].
        """
        linear_part = (
            self.global_bias
            + self.user_bias(user).squeeze(-1)
            + self.item_bias(item).squeeze(-1)
        )

        # Add feature biases
        if features is not None and self.merged_feature_bias is not None:
            global_indices = features + self.feature_offsets
            feat_bias = self.merged_feature_bias(global_indices).sum(dim=1).squeeze(-1)
            linear_part += feat_bias

        # Add context biases
        if contexts is not None and self.merged_context_bias is not None:
            global_indices = contexts + self.context_offsets
            ctx_bias = self.merged_context_bias(global_indices).sum(dim=1).squeeze(-1)
            linear_part += ctx_bias

        return linear_part

    def get_reg_params(
        self,
        user: Tensor,
        item: Tensor,
        features: Optional[Tensor],
        contexts: Optional[Tensor],
    ) -> List[Tensor]:
        """Helper to extract ALL embeddings and biases for regularization.

        Args:
            user (Tensor): User indices.
            item (Tensor): Item indices.
            features (Optional[Tensor]): Feature indices.
            contexts (Optional[Tensor]): Context indices.

        Returns:
            List[Tensor]: List of embeddings and biases to be passed to the Reg Loss.
        """
        reg_params = [
            self.user_embedding(user),
            self.item_embedding(item),
            self.user_bias(user),
            self.item_bias(item),
        ]

        if features is not None and self.merged_feature_embedding is not None:
            global_indices = features + self.feature_offsets
            reg_params.append(self.merged_feature_embedding(global_indices))
            reg_params.append(self.merged_feature_bias(global_indices))

        if contexts is not None and self.merged_context_embedding is not None:
            global_indices = contexts + self.context_offsets
            reg_params.append(self.merged_context_embedding(global_indices))
            reg_params.append(self.merged_context_bias(global_indices))

        return reg_params

    def _get_feature_embeddings(self, target_items: Tensor) -> Tensor:
        """Helper to retrieve feature embeddings for a set of items."""
        if not self.feature_dims or self.item_features is None:
            return None

        # Indices Lookup
        flat_items = target_items.view(-1).cpu()
        raw_indices = self.item_features[flat_items].to(target_items.device)  # type: ignore[index]

        # Apply Offsets
        global_indices = raw_indices + self.feature_offsets

        # Single Lookup
        embeddings = self.merged_feature_embedding(global_indices)

        # Reshape to match input
        target_shape = target_items.shape
        return embeddings.view(
            *target_shape, len(self.feature_labels), self.embedding_size
        )

    def _get_context_embeddings(self, contexts: Tensor) -> Optional[Tensor]:
        """Retrieves context embeddings as a single Tensor."""
        if not self.context_dims or self.merged_context_embedding is None:
            return None

        global_indices = contexts + self.context_offsets
        return self.merged_context_embedding(global_indices)

    def _get_feature_bias(self, target_items: Tensor) -> Tensor:
        """Helper to retrieve the sum of feature biases for a set of items."""
        if not self.feature_dims or self.item_features is None:
            return torch.zeros(target_items.shape, device=target_items.device)

        flat_items = target_items.view(-1).cpu()
        raw_indices = self.item_features[flat_items].to(target_items.device)  # type: ignore[index]

        global_indices = raw_indices + self.feature_offsets

        # Lookup & Sum
        biases = self.merged_feature_bias(global_indices).sum(dim=1).squeeze(-1)

        return biases.view(target_items.shape)


# pylint: disable = too-few-public-methods
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
        info (dict): The dictionary containing dataset information.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Raises:
        ValueError: If the items value was not passed through the info dict.
    """

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, seed=seed, *args, **kwargs)
        self.n_items = info.get("n_items", None)
        if not self.n_items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        self.item_similarity = np.zeros(self.n_items)

    def predict(
        self,
        train_batch: csr_matrix,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            train_batch (csr_matrix): The batch of user interaction vectors in sparse format.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Compute predictions and convert to Tensor
        predictions = train_batch @ self.item_similarity  # pylint: disable=not-callable
        predictions = torch.from_numpy(predictions)

        # Return full or sampled predictions
        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(
                max=self.n_items - 1
            ),  # [batch_size, pad_seq]
        )
