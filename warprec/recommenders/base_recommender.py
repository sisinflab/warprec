# pylint: disable = unused-argument
import random
import json
import inspect
import hashlib
from typing import Any, Optional, List, Dict, no_type_check
from abc import ABC, abstractmethod

import torch
import lightning as L
import numpy as np
import coolname
from torch import nn, Tensor
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.utils.data import DataLoader

from scipy.sparse import csr_matrix, diags, issparse
from sklearn.preprocessing import normalize

from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.config.model_configuration import LRSchedulerConfig, OptimizerConfig
from warprec.utils.logger import logger
from warprec.utils.registry import lr_scheduler_registry, optimizer_registry


# Memory budget for a single dense similarity block, in bytes
SIMILARITY_BLOCK_BYTES = 64 * 1024**2


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
        self.seed = seed
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
            "artifacts": self._learned_artifacts(),
        }
        return state

    def _learned_artifacts(self) -> Dict[str, Any]:
        """The attributes that hold what this model learned.

        Models trained in a single closed-form step keep their result in plain
        attributes - a similarity matrix, a pair of profiles - rather than in
        parameters, so ``state_dict`` is empty for them and a checkpoint alone
        would not be enough to serve them. Capturing the public attributes keeps
        the fitted model self-contained.

        Returns:
            Dict[str, Any]: The attribute names and their values.
        """
        return {
            name: value
            for name, value in self.__dict__.items()
            if not name.startswith("_") and name != "training"
        }

    @classmethod
    def estimate_space(
        cls,
        params: dict,
        info: dict,
        interactions: Optional[Interactions] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Estimate the train memory footprint of the model in MB."""
        raise NotImplementedError(
            f"Model '{cls.__name__}' does not implement estimate_space()."
        )

    @staticmethod
    def _require_interactions_for_estimate(
        interactions: Optional[Interactions], model_name: str
    ) -> Interactions:
        if interactions is None:
            raise ValueError(f"{model_name} requires interactions to estimate space.")
        return interactions

    @staticmethod
    def _bytes_to_mb(value: float) -> float:
        return float(value) / 1024**2

    @staticmethod
    def _dense_size_mb(shape: tuple, dtype: Any) -> float:
        return Recommender._bytes_to_mb(np.prod(shape) * np.dtype(dtype).itemsize)

    @staticmethod
    def _csr_size_mb(matrix: Any) -> float:
        return Recommender._bytes_to_mb(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        )

    @staticmethod
    def _sparse_size_mb(matrix: Any) -> float:
        if matrix is None:
            return 0.0

        total_bytes = matrix.data.nbytes if hasattr(matrix, "data") else 0
        total_bytes += matrix.indices.nbytes if hasattr(matrix, "indices") else 0
        total_bytes += matrix.indptr.nbytes if hasattr(matrix, "indptr") else 0
        total_bytes += matrix.row.nbytes if hasattr(matrix, "row") else 0
        total_bytes += matrix.col.nbytes if hasattr(matrix, "col") else 0
        return Recommender._bytes_to_mb(total_bytes)

    @staticmethod
    def _compressed_sparse_size_mb(
        nnz: int,
        ptr_len: int,
        data_dtype: Any,
        index_dtype: Any = np.int32,
    ) -> float:
        total_bytes = (
            nnz * np.dtype(data_dtype).itemsize
            + nnz * np.dtype(index_dtype).itemsize
            + ptr_len * np.dtype(index_dtype).itemsize
        )
        return Recommender._bytes_to_mb(total_bytes)

    @staticmethod
    def _estimated_sparse_square_size_mb(
        source_nnz: int,
        side_len: int,
        data_dtype: Any,
        overlap: float = 0.9,
        index_dtype: Any = np.int32,
    ) -> float:
        estimated_nnz = max(
            side_len,
            int(np.ceil(source_nnz * overlap)),
        )
        estimated_nnz = min(side_len * side_len, estimated_nnz)
        return Recommender._compressed_sparse_size_mb(
            nnz=estimated_nnz,
            ptr_len=side_len + 1,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
        )

    @staticmethod
    def _coo_size_mb(nnz: int, data_dtype: Any, index_dtype: Any = np.int32) -> float:
        total_bytes = (
            nnz * np.dtype(data_dtype).itemsize
            + 2 * nnz * np.dtype(index_dtype).itemsize
        )
        return Recommender._bytes_to_mb(total_bytes)

    @staticmethod
    def _peak_size_mb(*values: float) -> float:
        return float(max(values, default=0.0))

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

        # A model whose constructor fits on the training data cannot be rebuilt
        # from parameters alone. When the caller has no interactions to hand -
        # serving, typically - the fitted attributes saved with the checkpoint
        # are restored instead of refitting.
        artifacts = checkpoint.get("artifacts")
        if cls._fits_on_interactions() and "interactions" not in kwargs:
            # A model fitted in one closed-form step keeps its result in plain
            # attributes, so restoring those is enough to score again.
            if artifacts and not issubclass(cls, IterativeRecommender):
                return cls._from_artifacts(checkpoint, artifacts, strict=strict)

            # An iteratively trained model keeps its result in parameters, but
            # its constructor still derives the graph or the matrix shapes from
            # the interactions, so those have to be supplied.
            raise ValueError(
                f"{cls.__name__} derives its structure from the training "
                "interactions, so they must be passed to from_checkpoint() "
                "alongside the checkpoint."
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

    @classmethod
    def _fits_on_interactions(cls) -> bool:
        """Whether this model requires the training interactions to be built.

        Returns:
            bool: True when ``interactions`` is a required constructor argument.
        """
        parameter = inspect.signature(cls.__init__).parameters.get("interactions")
        return parameter is not None and parameter.default is inspect.Parameter.empty

    @classmethod
    def _from_artifacts(
        cls, checkpoint: Any, artifacts: Dict[str, Any], strict: bool = True
    ) -> "Recommender":
        """Rebuild a fitted model from its saved attributes, without refitting.

        Args:
            checkpoint (Any): The checkpoint being loaded.
            artifacts (Dict[str, Any]): The attributes saved with it.
            strict (bool): Whether to load the state dict in strict mode.

        Returns:
            Recommender: The restored model.
        """
        model = cls.__new__(cls)
        nn.Module.__init__(model)

        for name, value in artifacts.items():
            setattr(model, name, value)

        # Some of these models keep their result in buffers rather than in plain
        # attributes. The constructor that would have registered them has been
        # skipped, so they are declared here before the state is loaded into them.
        state_dict = checkpoint["state_dict"]
        for name, value in state_dict.items():
            if "." not in name and not hasattr(model, name):
                model.register_buffer(name, torch.empty_like(value))

        model.load_state_dict(state_dict, strict=strict)
        return model

    @staticmethod
    def _similarity_block_rows(side_len: int, block_bytes: int) -> int:
        """Rows per block for the blocked similarity builder.

        Sized against the widest element the similarities may return, so the
        slab honours the budget whatever dtype comes back, and never larger
        than the matrix itself.

        Args:
            side_len (int): The side of the square similarity matrix.
            block_bytes (int): Memory budget for a single dense block.

        Returns:
            int: The number of rows to process per block.
        """
        widest_itemsize = np.dtype(np.float64).itemsize
        rows = max(1, block_bytes // (side_len * widest_itemsize))
        return min(rows, side_len)

    @staticmethod
    def _tfidf(matrix: csr_matrix, normalize_tf: bool = False) -> csr_matrix:
        """Apply a TF-IDF weighting to a {row x feature} profile matrix.

        The inverse document frequency is taken over the rows of the matrix
        itself, so an item profile is weighted by how many items carry each
        feature and a user profile by how many users carry it. Weighting a user
        profile by the *item* frequencies, or skipping the term altogether,
        leaves a per-row scaling, and a per-row scaling cannot reorder a row:
        it cancels in cosine and factors out of a dot product.

        Args:
            matrix (csr_matrix): The profile matrix to weight.
            normalize_tf (bool): Whether to turn the raw counts into per-row
                frequencies before weighting. Item profiles keep the raw counts,
                user profiles are aggregations over a history whose length
                varies, so they are normalized first.

        Returns:
            csr_matrix: The L2 normalized TF-IDF profile.
        """
        matrix = matrix.tocsr()
        n_rows = matrix.shape[0]

        # Document frequency: the rows each feature appears in
        df = np.diff(matrix.tocsc().indptr)

        # IDF with smoothing
        idf = np.log((n_rows + 1) / (df + 1)) + 1

        tf = matrix
        if normalize_tf:
            row_sums = np.asarray(tf.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            tf = csr_matrix(tf.multiply(1 / row_sums[:, np.newaxis]))

        # L2 normalize
        return normalize(tf @ diags(idf), norm="l2", axis=1)

    @staticmethod
    def _blockwise_topk_similarity(
        matrix: Any,
        similarity: Any,
        k: int,
        block_bytes: int = SIMILARITY_BLOCK_BYTES,
    ) -> csr_matrix:
        """Build a top-k similarity matrix without materializing it densely.

        Computing ``similarity.compute(matrix)`` in one shot allocates a dense
        ``n x n`` array, and filtering it down to the top-k allocates a second
        one, even though only ``n * k`` entries survive.
        This method walks the rows in blocks, keeps the top-k of each block and
        accumulates the survivors directly in sparse form, so the peak
        allocation is one ``block x n`` slab plus the ``n * k`` result.

        ``torch.topk`` is row independent, so blocking returns exactly the same
        values and column indices as the dense path, ties included.

        Args:
            matrix (Any): The row-entity matrix to correlate with itself.
            similarity (Any): The similarity measure to apply.
            k (int): The number of neighbours to keep per row.
            block_bytes (int): Memory budget for a single dense block.

        Returns:
            csr_matrix: The {n x n} top-k similarity matrix.
        """
        n_rows = matrix.shape[0]

        # Safety check for k size, matching the dense implementation
        k = min(k, n_rows - 1)
        if k <= 0:
            return csr_matrix((n_rows, n_rows), dtype=matrix.dtype)

        rows_per_block = Recommender._similarity_block_rows(n_rows, block_bytes)

        total_nnz = n_rows * k
        rows = np.repeat(np.arange(n_rows, dtype=np.int32), k)
        cols = np.empty(total_nnz, dtype=np.int32)
        values: Optional[np.ndarray] = None

        for start in range(0, n_rows, rows_per_block):
            stop = min(start + rows_per_block, n_rows)

            # One {block x n} dense slab at a time instead of the full matrix
            block = torch.from_numpy(similarity.compute(matrix[start:stop], matrix))
            block_values, block_indices = torch.topk(block, k=k, dim=1)

            # The result dtype is decided by the similarity, not by the input
            block_values_np = block_values.numpy()
            if values is None:
                values = np.empty(total_nnz, dtype=block_values_np.dtype)

            cursor = start * k
            cols[cursor : stop * k] = block_indices.numpy().ravel()
            values[cursor : stop * k] = block_values_np.ravel()

        return csr_matrix((values, (rows, cols)), shape=(n_rows, n_rows))

    @staticmethod
    def _as_dense_tensor(predictions: Any) -> Tensor:
        """Convert a score matrix to a dense Tensor, sparse or dense alike.

        Similarity matrices may be stored sparsely, in which case the product
        with the training matrix stays sparse and has to be densified before
        it can be handed back as a Tensor.

        Args:
            predictions (Any): The computed scores, sparse or dense.

        Returns:
            Tensor: The dense score Tensor.
        """
        if issparse(predictions):
            predictions = predictions.toarray()
        return torch.from_numpy(np.asarray(predictions))

    @classmethod
    def _topk_similarity_size_mb(
        cls,
        side_len: int,
        k: int,
        data_dtype: Any,
        block_bytes: int = SIMILARITY_BLOCK_BYTES,
    ) -> tuple:
        """Size the sparse top-k similarity produced by the blocked builder.

        Args:
            side_len (int): The side of the square similarity matrix.
            k (int): The number of neighbours kept per row.
            data_dtype (Any): The dtype of the similarity values.
            block_bytes (int): Memory budget for a single dense block.

        Returns:
            tuple: The (peak, resident) footprint in MB.
        """
        nnz = min(side_len * side_len, side_len * k)

        resident_mb = cls._compressed_sparse_size_mb(
            nnz=nnz, ptr_len=side_len + 1, data_dtype=data_dtype
        )

        # While building: the COO staging arrays plus one dense block, then the
        # COO arrays plus the CSR they are converted into
        staging_mb = cls._coo_size_mb(nnz=nnz, data_dtype=data_dtype)
        block_mb = cls._dense_size_mb(
            (cls._similarity_block_rows(side_len, block_bytes), side_len), data_dtype
        )

        peak_mb = staging_mb + cls._peak_size_mb(block_mb, resident_mb)
        return peak_mb, resident_mb

    @classmethod
    def get_name_from_params(cls, params: dict) -> str:
        """Generates a deterministic coolname based on a dictionary of parameters."""
        # Add model name to params to ensure different models with same params get different names
        params_with_model = {"model": cls.__name__, **params}

        # Create a reproducible json dump of model parameters
        param_str = json.dumps(params_with_model, sort_keys=True, default=str)

        # Use the hash of the model hyperparameter as seed for the name generation
        hash_hex = hashlib.md5(
            param_str.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
        seed_int = int(hash_hex, 16)

        # Coolname uses the random seed
        current_state = random.getstate()
        fun_extension = "DefaultCoolName"  # Default name in case of failure

        # After the name generation the random seed
        # will be reverted to the experiment seed
        try:
            random.seed(seed_int)
            words = coolname.generate(3)
            fun_extension = "".join(word.capitalize() for word in words)
        finally:
            random.setstate(current_state)

        return f"{cls.__name__}_{fun_extension}"

    @property
    def name(self):
        """The name of the model."""
        return self.__class__.__name__

    @property
    def name_param(self):
        """The name of the model with a deterministic coolname extension.

        The name is generated based on the hash of the model's parameters,
        ensuring that the same parameters always yield the same name.
        """
        return self.get_name_from_params(self.get_params())

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


class IterativeRecommender(Recommender, L.LightningModule):
    """Interface for recommendation model that use
    an iterative approach to be trained.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        *args (Any): Argument for PyTorch LightningModule.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch LightningModule.

    Attributes:
        epochs (int): The number of epochs used to
            train the model.
        learning_rate (float): The learning rate using
            during optimization.
    """

    epochs: int
    learning_rate: float

    def __init__(
        self, params: dict, info: dict, *args: Any, seed: int = 42, **kwargs: Any
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)
        self.save_hyperparameters(params)

        # Optimization parameters
        self._optimizer_name: str = "Adam"
        self._optimizer_kwargs: Dict[str, Any] = {}
        self._lr_scheduler_name: Optional[str] = None
        self._lr_scheduler_kwargs: Dict[str, Any] = {}

    def set_optimization_parameters(
        self,
        optimizer_config: Optional[OptimizerConfig] = None,
        lr_scheduler_config: Optional[LRSchedulerConfig] = None,
    ):
        """Set the optimizer and scheduler used during training.

        Args:
            optimizer_config (Optional[OptimizerConfig]): The optimizer configuration.
            lr_scheduler_config (Optional[LRSchedulerConfig]): The scheduler configuration.
        """
        # Set the optimizer values
        if optimizer_config:
            self._optimizer_name = optimizer_config.name
            self._optimizer_kwargs = optimizer_config.params

        # Set the scheduler values
        if lr_scheduler_config:
            self._lr_scheduler_name = lr_scheduler_config.name
            self._lr_scheduler_kwargs = lr_scheduler_config.params

    def configure_optimizers(self):
        """Standard Lightning method to define optimizers.

        This method separates parameters into two groups:
        1. Decay Group:
           - Dense layers weights (Linear, Conv).
           - Structural embeddings (e.g., Positional Embeddings).
        2. No-Decay Group:
           - Sparse Entity Embeddings (User/Item) -> Handled manually by EmbLoss.
           - Biases -> Standard DL practice (no decay).
           - LayerNorm weights -> Standard Transformer practice (no decay).
        """
        # Identify parameters that belong to nn.Embedding modules
        embedding_param_ids = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                for param in module.parameters():
                    embedding_param_ids.add(id(param))

        # Separate parameters into groups
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # We disable optimizer weight decay for:
            # A. Biases (standard practice)
            # B. LayerNorm parameters (standard Transformer practice)
            # C. Sparse Embeddings (User/Item), because we use EmbLoss for them.
            #    EXCEPTION: Positional Embeddings should have weight decay applied.

            is_bias = "bias" in name
            is_layernorm = "layernorm" in name or "norm" in name
            is_embedding = id(param) in embedding_param_ids
            is_positional = "position" in name  # Heuristic to catch position_embedding

            if is_bias or is_layernorm or (is_embedding and not is_positional):
                no_decay_params.append(param)
            else:
                # Linear weights, Conv weights, and Positional Embeddings go here
                decay_params.append(param)

        # Finalize the Optimizer with correct groups
        decay = getattr(self, "weight_decay", 0.0)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        # Create the optimizer instance
        optimizer = optimizer_registry.get(
            name=self._optimizer_name,
            params=optimizer_grouped_parameters,
            lr=self.learning_rate,
            **self._optimizer_kwargs,
        )

        # No learning rate scheduler provided -> Return only the optimizer
        if self._lr_scheduler_name is None:
            return optimizer

        # Create the instance of the learning rate scheduler
        scheduler = lr_scheduler_registry.get(
            name=self._lr_scheduler_name,
            optimizer=optimizer,
            **self._lr_scheduler_kwargs,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

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

            # Xavier fills the whole matrix, padding row included. That row is
            # excluded from every gradient update, so a random value there would
            # never be corrected.
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

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
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Performs a single training step for a given batch.

        This is a standard method defined by PyTorch Lightning.

        Args:
            batch (Any): A single batch of data from the DataLoader.
            batch_idx (int): The current current batch index.

        Returns:
            Tensor: The computed loss for the batch.
        """

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """PyTorch Lightning needs this method to be implemented
        to correctly perform the on_validation_epoch_end callback.

        Args:
            batch (Any): A single batch of data from the DataLoader.
            batch_idx (int): The current current batch index.

        Returns:
            Any: The batch of data from the DataLoader.
        """
        return batch

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """PyTorch Lightning hook used during checkpoint saving.

        Args:
            checkpoint (Dict[str, Any]): The dictionary containing the
                checkpoint information.
        """
        checkpoint["name"] = self.name
        checkpoint["params"] = self.get_params()
        checkpoint["info"] = self.info


class ContextRecommenderUtils(nn.Module, ABC):
    # pylint: disable = too-many-instance-attributes  # this class is the state it holds
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
        **kwargs (Any): Arbitrary keyword arguments. ``transactions`` carries the
            row-oriented training records, passed through by the pipelines.

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
        context_types: dict = info.get("context_types", {})
        self.context_is_float = [
            context_types.get(name) == "float" for name in self.context_labels
        ]

        # Row-oriented source of training examples. Absent for a dataset with no
        # contextual columns, in which case the matrix view is used instead.
        self._transactions = kwargs.get("transactions")

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

        The rows are preferred over the matrix: a matrix cell cannot hold the
        same pair seen in several contexts, so sourcing from it would drop every
        record but one and leave the contexts attached to nothing.

        Args:
            interactions (Interactions): The interaction of users with items.
            sessions (Sessions): The sessions of the users.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            DataLoader: The appropriate dataloader for the training.
        """
        source: Any = self._transactions

        if source is None:
            logger.attention(
                f"{self.__class__.__name__} received no transactions and will read the "
                "interaction matrix instead. Contexts cannot be aligned this way."
            )
            source = interactions

        return source.get_pointwise_dataloader(
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
            linear_part += self._get_context_bias(contexts)

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
            global_indices = contexts.long() + self.context_offsets
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
        """Retrieve one embedding per context field.

        A categorical field looks its value up in the merged table. A numeric field
        owns a single row of that table and scales it by the value it carries, which
        is what lets a measurement keep its ordering instead of being spread over an
        invented vocabulary. Either way a field contributes exactly one vector, so
        the number of fields a model sees never changes.

        Args:
            contexts (Tensor): The context row, indices and values together.

        Returns:
            Optional[Tensor]: The per-field embeddings, or None without contexts.
        """
        if not self.context_dims or self.merged_context_embedding is None:
            return None

        offsets = self.context_offsets
        indices = contexts.long() + offsets
        if not any(self.context_is_float):
            return self.merged_context_embedding(indices)

        # A numeric field always sits at its own offset; only the scaling differs.
        numeric = torch.tensor(self.context_is_float, device=contexts.device)
        indices = torch.where(numeric, offsets.expand_as(indices), indices)
        embeddings = self.merged_context_embedding(indices)
        scale = torch.where(numeric, contexts, torch.ones_like(contexts))
        return embeddings * scale.unsqueeze(-1)

    def _get_context_bias(self, contexts: Tensor) -> Tensor:
        """Sum the first-order term contributed by the context fields.

        Args:
            contexts (Tensor): The context row, indices and values together.

        Returns:
            Tensor: The summed context bias, one value per row.
        """
        offsets = self.context_offsets
        indices = contexts.long() + offsets
        if not any(self.context_is_float):
            return self.merged_context_bias(indices).sum(dim=1).squeeze(-1)

        numeric = torch.tensor(self.context_is_float, device=contexts.device)
        indices = torch.where(numeric, offsets.expand_as(indices), indices)
        biases = self.merged_context_bias(indices).squeeze(-1)
        scale = torch.where(numeric, contexts, torch.ones_like(contexts))
        return (biases * scale).sum(dim=1)

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
        interactions (Interactions): The training interactions.
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
        interactions: Interactions,
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
        self.train_matrix = interactions.get_sparse()
        self.item_similarity = np.zeros(self.n_items)

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
        """
        # Compute predictions and convert to Tensor. The similarity matrix may
        # be stored sparsely, which keeps the product sparse until densified.
        predictions = self.train_matrix[user_indices.tolist(), :] @ self.item_similarity
        predictions = self._as_dense_tensor(predictions)

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
