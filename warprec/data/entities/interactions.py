from typing import Tuple, Any, Optional, List

import numpy as np
import narwhals as nw
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from narwhals.dataframe import DataFrame
from scipy.sparse import csr_matrix, coo_matrix

from warprec.data.entities.train_structures import (
    InteractionDataset,
    PointWiseDataset,
    ContrastiveDataset,
)
from warprec.utils.enums import RatingType


# Worker seed function for reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class Interactions:
    """Interactions class will handle the data of the transactions.

    Args:
        data (DataFrame[Any]): Transaction data in DataFrame format.
        original_dims (Tuple[int, int]):
            int: Number of users.
            int: Number of items.
        user_mapping (dict): Mapping of user ID -> user idx.
        item_mapping (dict): Mapping of item ID -> item idx.
        side_data (Optional[DataFrame[Any]]): The side information features in DataFrame format.
        user_cluster (Optional[dict]): The user cluster information.
        item_cluster (Optional[dict]): The item cluster information.
        batch_size (int): The batch size that will be used to
            iterate over the interactions.
        rating_type (RatingType): The type of rating to be used.
        rating_label (str): The label of the rating column.
        context_labels (Optional[List[str]]): The list of labels of the
            contextual data.
        precision (Any): The precision of the internal representation of the data.
    """

    def __init__(
        self,
        data: DataFrame[Any],
        original_dims: Tuple[int, int],
        user_mapping: dict,
        item_mapping: dict,
        side_data: Optional[DataFrame[Any]] = None,
        user_cluster: Optional[dict] = None,
        item_cluster: Optional[dict] = None,
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        rating_label: str = None,
        context_labels: Optional[List[str]] = None,
        precision: Any = np.float32,
    ) -> None:
        # Setup the variables
        self._inter_df = data
        self._inter_side = side_data.clone() if side_data is not None else None
        self._inter_user_cluster = user_cluster if user_cluster is not None else None
        self._inter_item_cluster = item_cluster if item_cluster is not None else None
        self.batch_size = batch_size
        self.rating_type = rating_type
        self.precision = precision

        # Setup the training variables
        self._inter_dict: Optional[dict] = None
        self._inter_sparse: csr_matrix = None
        self._inter_side_sparse: csr_matrix = None
        self._inter_side_tensor: Tensor = None
        self._inter_side_labels: List[str] = []
        self._history_matrix: Tensor = None
        self._history_lens: Tensor = None
        self._history_mask: Tensor = None

        # Set DataFrame labels
        self.user_label = data.columns[0]
        self.item_label = data.columns[1]
        self.rating_label = rating_label if rating_type == RatingType.EXPLICIT else None
        self.context_labels = context_labels if context_labels else []

        # Set mappings
        self._umap = user_mapping
        self._imap = item_mapping

        # Filter side information (if present)
        if self._inter_side is not None:
            valid_items = self._inter_df.select(self.item_label).unique()
            # We use inner join on unique items to filter
            self._inter_side = self._inter_side.join(
                valid_items, on=self.item_label, how="inner"
            )

            # Order side information to be in the same order of the dataset (by item index)
            # Create mapping DF for items
            imap_df = nw.from_dict(
                {
                    self.item_label: list(item_mapping.keys()),
                    "__order__": list(item_mapping.values()),
                },
                native_namespace=nw.get_native_namespace(self._inter_side),
            )

            # Join to get the order, sort, and drop temp column
            self._inter_side = (
                self._inter_side.join(imap_df, on=self.item_label, how="left")
                .sort("__order__")
                .drop("__order__")
            )

            # Construct lookup for side information features
            feature_cols = [c for c in self._inter_side.columns if c != self.item_label]

            # Create the lookup tensor for side information
            # Extract values to numpy
            side_values = self._inter_side.select(feature_cols).to_numpy()
            side_tensor = torch.tensor(side_values, dtype=torch.long)

            # Create the padding row (zeros)
            padding_row = torch.zeros((1, side_tensor.shape[1]), dtype=torch.long)

            # Concatenate padding row at the beginning (assuming index 0 is padding/unknown)
            self._inter_side_tensor = torch.cat([side_tensor, padding_row], dim=0)

            # Store the feature labels
            self._inter_side_labels = feature_cols

        # Definition of dimensions
        self._uid = self._inter_df.select(self.user_label).unique().to_numpy().flatten()
        self._nuid = self._inter_df.select(nw.col(self.user_label).n_unique()).item()
        self._niid = self._inter_df.select(nw.col(self.item_label).n_unique()).item()
        self._og_nuid, self._og_niid = original_dims
        self._transactions = self._inter_df.select(nw.len()).item()

    def _get_mapped_indices(self) -> Tuple[Tensor, Tensor]:
        """Retrieves mapped user and item indices directly from the sparse matrix structure.

        Returns:
            Tuple[Tensor, Tensor]: (user_indices, item_indices) aligned as LongTensors.
        """
        mat = self.get_sparse()

        if not mat.has_sorted_indices:
            mat.sort_indices()

        # Extract the positive items
        pos_items = mat.indices.astype(np.int64)

        # Reconstruct the users
        n_users = mat.shape[0]
        interactions_per_user = np.diff(mat.indptr)
        users = np.repeat(np.arange(n_users), interactions_per_user).astype(np.int64)

        # Return Tensors directly
        return torch.from_numpy(users), torch.from_numpy(pos_items)

    def get_dict(self) -> dict:
        """This method will return the transaction information in dict format.

        Returns:
            dict: The transaction information in the current
                representation {user ID: {item ID: rating}}.
        """
        if self._inter_dict is not None:
            return self._inter_dict

        u_vals = self._inter_df.select(self.user_label).to_numpy().flatten()
        i_vals = self._inter_df.select(self.item_label).to_numpy().flatten()

        self._inter_dict = {}

        if self.rating_type == RatingType.EXPLICIT:
            r_vals = self._inter_df.select(self.rating_label).to_numpy().flatten()
            for u, i, r in zip(u_vals, i_vals, r_vals):
                if u not in self._inter_dict:
                    self._inter_dict[u] = {}
                self._inter_dict[u][i] = r
        elif self.rating_type == RatingType.IMPLICIT:
            for u, i in zip(u_vals, i_vals):
                if u not in self._inter_dict:
                    self._inter_dict[u] = {}
                self._inter_dict[u][i] = 1

        return self._inter_dict

    def get_df(self) -> DataFrame[Any]:
        """This method will return the raw data.

        Returns:
            DataFrame[Any]: The raw data in tabular format.
        """
        return self._inter_df

    def get_sparse(self) -> csr_matrix:
        """This method retrieves the sparse representation of data.

        This method also checks if the sparse structure has
        already been created, if not then it also create it first.

        Returns:
            csr_matrix: Sparse representation of the transactions (CSR Format).
        """
        if isinstance(self._inter_sparse, csr_matrix):
            return self._inter_sparse
        return self._to_sparse()

    def get_sparse_by_rating(self, rating_value: float) -> coo_matrix:
        """Returns a sparse matrix (COO format) containing only the interactions
        that match a specific rating value.

        Args:
            rating_value (float): The rating value to filter by.

        Returns:
            coo_matrix: A sparse matrix of shape [num_users, num_items] for the given rating.

        Raises:
            ValueError: If interactions are not explicit or if
                rating label is None.
        """
        if self.rating_type != RatingType.EXPLICIT or self.rating_label is None:
            raise ValueError(
                "Filtering by rating is only supported for explicit feedback data."
            )

        # Filter original DataFrame for the specified rating value
        rating_df = self._inter_df.filter(nw.col(self.rating_label) == rating_value)

        # Edge case: No interactions with the specified rating
        if rating_df.select(nw.len()).item() == 0:
            return coo_matrix((self._og_nuid, self._og_niid), dtype=self.precision)

        umap_df = nw.from_dict(
            {
                self.user_label: list(self._umap.keys()),
                "__uidx__": list(self._umap.values()),
            },
            native_namespace=nw.get_native_namespace(rating_df),
        )

        imap_df = nw.from_dict(
            {
                self.item_label: list(self._imap.keys()),
                "__iidx__": list(self._imap.values()),
            },
            native_namespace=nw.get_native_namespace(rating_df),
        )

        # Join to map
        mapped_df = rating_df.join(umap_df, on=self.user_label, how="inner").join(
            imap_df, on=self.item_label, how="inner"
        )

        # Sort to ensure reproducibility
        mapped_df = mapped_df.sort(["__uidx__", "__iidx__"])

        # Extract indices
        users = mapped_df.select("__uidx__").to_numpy().flatten()
        items = mapped_df.select("__iidx__").to_numpy().flatten()

        # Values are all ones for the presence of interaction
        values = np.ones(len(users), dtype=self.precision)

        return coo_matrix(
            (values, (users, items)),
            shape=(self._og_nuid, self._og_niid),
            dtype=self.precision,
        )

    def get_side_sparse(self) -> csr_matrix:
        """This method retrieves the sparse representation of side data.

        This method also checks if the sparse structure has
        already been created, if not then it also create it first.

        Returns:
            csr_matrix: Sparse representation of the features (CSR Format).
        """
        if isinstance(self._inter_side_sparse, csr_matrix):
            return self._inter_side_sparse
        if self._inter_side is None:
            return None

        # Drop item label and convert to sparse
        side_features = self._inter_side.drop(self.item_label)
        # Convert to numpy first
        side_np = side_features.to_numpy()

        self._inter_side_sparse = csr_matrix(side_np, dtype=self.precision)
        return self._inter_side_sparse

    def get_side_tensor(self) -> Tensor:
        """This method retrieves the tensor representation of side data.

        Returns:
            Tensor: Tensor representation of the features if available.
        """
        return self._inter_side_tensor

    def get_interaction_dataloader(
        self,
        include_user_id: bool = False,
        batch_size: int = 1024,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a PyTorch DataLoader that yields dense tensors of interaction batches.

        This method retrieves the sparse interaction matrix, converts it into a PyTorch
        TensorDataset, and then wraps it in a DataLoader. The batches are provided as
        dense tensors of shape [batch_size, num_items].

        Args:
            include_user_id (bool): Whether to include user IDs in the output.
            batch_size (int): The batch size to be used for the DataLoader.
            shuffle (bool): Whether to shuffle the data when loading.
            **kwargs (Any): The additional keyword arguments to pass the Dataloader.

        Returns:
            DataLoader: A DataLoader that yields batches of dense interaction tensors.
        """
        # Get the sparse matrix, which is memory-efficient.
        sparse_matrix = self.get_sparse()

        # Create the lazy dataset which just holds a reference to the sparse matrix.
        lazy_dataset = InteractionDataset(
            sparse_matrix, include_user_id=include_user_id
        )
        return DataLoader(
            lazy_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
        )

    def get_pointwise_dataloader(
        self,
        neg_samples: int = 0,
        include_side_info: bool = False,
        include_context: bool = False,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a PyTorch DataLoader with implicit feedback and negative sampling.

        Args:
            neg_samples (int): Number of negative samples per user.
            include_side_info (bool): Whether to include side information features in the output.
            include_context (bool): Wether to include the context in the output.
            batch_size (int): The batch size that will be used to
            shuffle (bool): Whether to shuffle the data.
            seed (int): Seed for Numpy random number generator for reproducibility.
            **kwargs (Any): The additional keyword arguments to pass the Dataloader.

        Returns:
            DataLoader: Yields (user, item, rating) with negative samples or
                (user, item, rating, context) if flagged.
        """
        pos_users, pos_items = self._get_mapped_indices()

        # Prepare side information and context if requested
        side_info_tensor = None
        if include_side_info and self._inter_side_tensor is not None:
            side_info_tensor = self._inter_side_tensor

        context_tensor = None
        if include_context and self.context_labels:
            ctx_vals = self._inter_df.select(self.context_labels).to_numpy()
            context_tensor = torch.tensor(ctx_vals, dtype=torch.long)

        # Create the Dataset
        dataset = PointWiseDataset(
            user_ids=pos_users,
            item_ids=pos_items,
            sparse_matrix=self.get_sparse(),
            neg_samples=neg_samples,
            niid=self._niid,
            side_information=side_info_tensor,
            contexts=context_tensor,
        )

        # Set the generator for the Dataloader for reproducibility
        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
            **kwargs,
        )

    def get_contrastive_dataloader(
        self,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> DataLoader:
        """Create a PyTorch DataLoader with triplets for implicit feedback.

        Args:
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.
            seed (int): Seed for reproducibility.
            **kwargs (Any): The additional keyword arguments to pass the Dataloader.

        Returns:
            DataLoader: Yields triplets of (user, positive_item, negative_item).
        """
        pos_users, pos_items = self._get_mapped_indices()

        # Create the Dataset
        dataset = ContrastiveDataset(
            user_ids=pos_users,
            item_ids=pos_items,
            sparse_matrix=self.get_sparse(),
            niid=self._niid,
        )

        # Set the generator for the Dataloader for reproducibility
        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
            **kwargs,
        )

    def get_history(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the history representation as three Tensors.

        This method also checks if this representation has been already computed,
        if so then it just returns it without computing it again.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - Tensor: A matrix of dimension [num_user, max_chronology_length],
                    containing transaction information.
                - Tensor: An array of dimension [num_user], containing the
                    length of each chronology (before padding).
                - Tensor: A binary mask that identifies where the real
                    transaction information are, ignoring padding.
        """
        if (
            isinstance(self._history_matrix, Tensor)
            and isinstance(self._history_lens, Tensor)
            and isinstance(self._history_mask, Tensor)
        ):
            return self._history_matrix, self._history_lens, self._history_mask
        return self._to_history()

    def get_dims(self) -> Tuple[int, int]:
        """This method will return the dimensions of the data.

        Returns:
            Tuple[int, int]: A tuple containing:
                int: Number of unique users.
                int: Number of unique items.
        """
        return (self._nuid, self._niid)

    def get_transactions(self) -> int:
        """This method will return the number of transactions.

        Returns:
            int: Number of transactions.
        """
        return self._transactions

    def get_unique_ratings(self) -> np.ndarray:
        """Returns a sorted array of unique rating values present in the dataset.
        This is useful for models that operate on explicit feedback.

        Returns:
            np.ndarray: A sorted array of unique rating values.
        """
        if self.rating_type != RatingType.EXPLICIT or self.rating_label is None:
            return np.array([])

        return np.sort(
            self._inter_df.select(self.rating_label).unique().to_numpy().flatten()
        )

    def _to_sparse(self) -> csr_matrix:
        """This method will create the sparse representation of the data contained.

        This method must not be called if the sparse representation has already be defined.

        Returns:
            csr_matrix: Sparse representation of the transactions (CSR Format).
        """
        umap_df = nw.from_dict(
            {
                self.user_label: list(self._umap.keys()),
                "__uidx__": list(self._umap.values()),
            },
            native_namespace=nw.get_native_namespace(self._inter_df),
        )

        imap_df = nw.from_dict(
            {
                self.item_label: list(self._imap.keys()),
                "__iidx__": list(self._imap.values()),
            },
            native_namespace=nw.get_native_namespace(self._inter_df),
        )

        # Join
        mapped_df = self._inter_df.join(umap_df, on=self.user_label, how="inner").join(
            imap_df, on=self.item_label, how="inner"
        )

        # Sort to ensure reproducibility
        mapped_df = mapped_df.sort(["__uidx__", "__iidx__"])

        users = mapped_df.select("__uidx__").to_numpy().flatten()
        items = mapped_df.select("__iidx__").to_numpy().flatten()

        if self.rating_type == RatingType.EXPLICIT:
            ratings = mapped_df.select(self.rating_label).to_numpy().flatten()
        else:
            ratings = np.ones(len(users))

        self._inter_sparse = coo_matrix(
            (ratings, (users, items)),
            shape=(self._og_nuid, self._og_niid),
            dtype=self.precision,
        ).tocsr()

        return self._inter_sparse

    def _to_history(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Creates three Tensor which contains information of the
        transaction history for each user.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - Tensor: A matrix of dimension [num_user, max_chronology_length],
                    containing transaction information.
                - Tensor: An array of dimension [num_user], containing the
                    length of each chronology (before padding).
                - Tensor: A binary mask that identifies where the real
                    transaction information are, ignoring padding.
        """
        # Get sparse interaction matrix
        sparse_matrix = self.get_sparse()
        n_users = sparse_matrix.shape[0]
        n_items = sparse_matrix.shape[1]
        indptr = sparse_matrix.indptr
        indices = sparse_matrix.indices

        # Calculate lengths for each user
        lens = indptr[1:] - indptr[:-1]
        max_history_len = int(lens.max()) if len(lens) > 0 else 0

        # Initialize matrices
        self._history_matrix = torch.full(
            (n_users, max_history_len), fill_value=n_items, dtype=torch.long
        )
        self._history_lens = torch.from_numpy(lens.astype(np.int64))
        self._history_mask = torch.zeros(n_users, max_history_len, dtype=torch.float)

        # Populate matrices using slicing
        for u in range(n_users):
            start, end = indptr[u], indptr[u + 1]
            if end > start:
                length = end - start
                items = torch.from_numpy(indices[start:end].astype(np.int64))
                self._history_matrix[u, :length] = items
                self._history_mask[u, :length] = 1.0

        return self._history_matrix, self._history_lens, self._history_mask

    def __len__(self) -> int:
        """This method calculates the length of the interactions.

        Length will be defined as the number of ratings.

        Returns:
            int: number of ratings present in the structure.
        """
        return self._transactions
