import typing
from typing import Tuple, Any, Optional, Dict, List

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import narwhals as nw
from narwhals.typing import FrameT

from scipy.sparse import csr_matrix, coo_matrix

from warprec.data.entities.train_structures import (
    LazyInteractionDataset,
    LazyItemRatingDataset,
    LazyTripletDataset,
)
from warprec.utils.enums import RatingType


class Interactions:
    """Interactions class will handle the data of the transactions.

    Args:
        data (DataFrame): Transaction data in DataFrame format.
        original_dims (Tuple[int, int]):
            int: Number of users.
            int: Number of items.
        user_mapping (dict): Mapping of user ID -> user idx.
        item_mapping (dict): Mapping of item ID -> item idx.
        side_data (Optional[DataFrame]): The side information features in DataFrame format.
        user_cluster (Optional[dict]): The user cluster information.
        item_cluster (Optional[dict]): The item cluster information.
        batch_size (int): The batch size that will be used to
            iterate over the interactions.
        rating_type (RatingType): The type of rating to be used.
        rating_label (str): The label of the rating column.
        context_labels (Optional[List[str]]): The list of labels of the
            contextual data.
        precision (Any): The precision of the internal representation of the data.

    Raises:
        ValueError: If the rating type is not supported.
    """

    def __init__(
        self,
        data: FrameT,
        original_dims: Tuple[int, int],
        user_mapping: dict,
        item_mapping: dict,
        side_data: Optional[FrameT] = None,
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
        self._cached_dataset: Dict[str, TensorDataset] = {}
        self._cached_tensor: Dict[str, Tensor] = {}
        self._inter_dict: dict = {}
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
            self._inter_side = self._inter_side.join(valid_items, on=self.item_label, how="inner")

            # Order side information to be in the same order of the dataset (by item index)
            # Create mapping DF for items
            imap_df = nw.from_dict({
                self.item_label: list(item_mapping.keys()),
                "__order__": list(item_mapping.values())
            }, native_namespace=nw.get_native_namespace(self._inter_side))

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

        # Set the index
        self._index = 0
        
        u_vals = self._inter_df.select(self.user_label).to_numpy().flatten()
        i_vals = self._inter_df.select(self.item_label).to_numpy().flatten()

        # Define the interaction dictionary, based on the RatingType selected
        if rating_type == RatingType.EXPLICIT:
            r_vals = self._inter_df.select(self.rating_label).to_numpy().flatten()
            # Build dict: {user: {item: rating}}
            # Iterating numpy arrays is faster than pandas apply/groupby for this specific structure
            self._inter_dict = {}
            for u, i, r in zip(u_vals, i_vals, r_vals):
                if u not in self._inter_dict:
                    self._inter_dict[u] = {}
                self._inter_dict[u][i] = r
                
        elif rating_type == RatingType.IMPLICIT:
            self._inter_dict = {}
            for u, i in zip(u_vals, i_vals):
                if u not in self._inter_dict:
                    self._inter_dict[u] = {}
                self._inter_dict[u][i] = 1
        else:
            raise ValueError(f"Rating type {rating_type} not supported.")

    def clear_dataset_cache(self):
        """This method will clear the cached Dataset objects."""
        del self._cached_dataset
        self._cached_dataset = {}

        del self._cached_tensor
        self._cached_tensor = {}

    def get_dict(self) -> dict:
        """This method will return the transaction information in dict format.

        Returns:
            dict: The transaction information in the current
                representation {user ID: {item ID: rating}}.
        """
        return self._inter_dict

    def get_df(self) -> FrameT:
        """This method will return the raw data.

        Returns:
            DataFrame: The raw data in tabular format.
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

        umap_df = nw.from_dict({
            self.user_label: list(self._umap.keys()),
            "__uidx__": list(self._umap.values())
        }, native_namespace=nw.get_native_namespace(rating_df))
        
        imap_df = nw.from_dict({
            self.item_label: list(self._imap.keys()),
            "__iidx__": list(self._imap.values())
        }, native_namespace=nw.get_native_namespace(rating_df))

        # Join to map
        mapped_df = rating_df.join(umap_df, on=self.user_label, how="inner") \
                             .join(imap_df, on=self.item_label, how="inner")

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
        
        self._inter_side_sparse = csr_matrix(
            side_np, dtype=self.precision
        )
        return self._inter_side_sparse

    def get_side_tensor(self) -> Tensor:
        """This method retrieves the tensor representation of side data.

        Returns:
            Tensor: Tensor representation of the features if available.
        """
        return self._inter_side_tensor

    def get_interaction_loader(
        self,
        include_user_id: bool = False,
        batch_size: int = 1024,
        shuffle: bool = True,
        low_memory: bool = False,
    ) -> DataLoader:
        """Create a PyTorch DataLoader that yields dense tensors of interaction batches.

        This method retrieves the sparse interaction matrix, converts it into a PyTorch
        TensorDataset, and then wraps it in a DataLoader. The batches are provided as
        dense tensors of shape [batch_size, num_items].

        Args:
            include_user_id (bool): Whether to include user IDs in the output.
            batch_size (int): The batch size to be used for the DataLoader.
            shuffle (bool): Whether to shuffle the data when loading.
            low_memory (bool): Whether to create the dataloader with a lazy approach.

        Returns:
            DataLoader: A DataLoader that yields batches of dense interaction tensors.
        """
        if low_memory:
            # Get the sparse matrix, which is memory-efficient.
            sparse_matrix = self.get_sparse()

            # Create the lazy dataset which just holds a reference to the sparse matrix.
            lazy_dataset = LazyInteractionDataset(
                sparse_matrix, include_user_id=include_user_id
            )
            return DataLoader(lazy_dataset, batch_size=batch_size, shuffle=shuffle)

        # Check if interactions have been cached
        cache_key = f"interaction_user_{include_user_id}"
        if cache_key in self._cached_dataset:
            dataset = self._cached_dataset[cache_key]
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Get the sparse interaction matrix. This ensures it's created if it doesn't exist.
        sparse_matrix = self.get_sparse()

        # Convert the sparse matrix to a dense tensor.
        dense_tensor = torch.from_numpy(sparse_matrix.todense()).to(dtype=torch.float32)

        # Create a TensorDataset from the dense tensor.
        # If requested also the user indices tensor will be yielded
        if include_user_id:
            indices_tensor = torch.arange(sparse_matrix.shape[0], dtype=torch.long)
            dataset = TensorDataset(indices_tensor, dense_tensor)
        else:
            dataset = TensorDataset(dense_tensor)

        # Cache the dataset
        self._cached_dataset[cache_key] = dataset

        # Wrap the dataset in a DataLoader.
        # The DataLoader will handle batching and shuffling.
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_item_rating_dataloader(
        self,
        neg_samples: int = 0,
        include_side_info: bool = False,
        include_context: bool = False,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        low_memory: bool = False,
    ) -> DataLoader:
        """Create a PyTorch DataLoader with implicit feedback and negative sampling.

        Args:
            neg_samples (int): Number of negative samples per user.
            include_side_info (bool): Whether to include side information features in the output.
            include_context (bool): Wether to include the context in the output.
            batch_size (int): The batch size that will be used to
            shuffle (bool): Whether to shuffle the data.
            seed (int): Seed for Numpy random number generator for reproducibility.
            low_memory (bool): Whether to create the dataloader with a lazy approach.

        Returns:
            DataLoader: Yields (user, item, rating) with negative samples or
                (user, item, rating, context) if flagged.

        Raises:
            ValueError: If context flag has been set but no context
                information is present in the DataFrame.
        """
        
        # Helper to get mapped indices
        def get_mapped_indices():
            # Create mapping DFs
            umap_df = nw.from_dict({
                self.user_label: list(self._umap.keys()),
                "__uidx__": list(self._umap.values())
            }, native_namespace=nw.get_native_namespace(self._inter_df))
            
            imap_df = nw.from_dict({
                self.item_label: list(self._imap.keys()),
                "__iidx__": list(self._imap.values())
            }, native_namespace=nw.get_native_namespace(self._inter_df))

            # Join
            mapped_df = self._inter_df.join(umap_df, on=self.user_label, how="inner") \
                                      .join(imap_df, on=self.item_label, how="inner")
            
            u_idx = mapped_df.select("__uidx__").to_numpy().flatten().astype(np.int64)
            i_idx = mapped_df.select("__iidx__").to_numpy().flatten().astype(np.int64)
            return u_idx, i_idx
        
        # pylint: disable=too-many-branches, too-many-statements
        if low_memory:
            sparse_matrix = self.get_sparse()

            # Extract positive interaction information
            pos_users_np, pos_items_np = get_mapped_indices()
            
            # Prepare side information
            side_info_tensor = None
            if include_side_info:
                if self._inter_side_tensor is not None:
                    side_info_tensor = self._inter_side_tensor
                else:
                    raise ValueError(
                        "Requested side information but none provided in init."
                    )

            # Prepare the context
            context_tensor = None
            if include_context:
                if not self.context_labels:
                    raise ValueError(
                        "Requested context information but none provided in init."
                    )

                context_values = self._inter_df.select(self.context_labels).to_numpy()
                context_tensor = torch.tensor(context_values, dtype=torch.long)

            # Create the lazy dataset
            lazy_dataset = LazyItemRatingDataset(
                user_ids=pos_users_np,
                item_ids=pos_items_np,
                sparse_matrix=sparse_matrix,
                neg_samples=neg_samples,
                niid=self._niid,
                seed=seed,
                side_information=side_info_tensor,
                contexts=context_tensor,
            )

            # Edge case: No interactions
            if len(lazy_dataset) == 0:
                return DataLoader(
                    TensorDataset(
                        torch.LongTensor([]),
                        torch.LongTensor([]),
                        torch.FloatTensor([]),
                    )
                )

            return DataLoader(lazy_dataset, batch_size=batch_size, shuffle=shuffle)

        # Check if dataloader has been cached
        cache_key = f"item_rating_neg_{neg_samples}_context_{include_context}"
        if cache_key in self._cached_dataset:
            dataset = self._cached_dataset[cache_key]
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Extract the positive interactions
        pos_users_np, pos_items_np = get_mapped_indices()

        pos_users = torch.from_numpy(pos_users_np)
        pos_items = torch.from_numpy(pos_items_np)
        pos_ratings = torch.ones(len(pos_users), dtype=torch.float)

        # Extract side information if flagged
        pos_features = None
        if include_side_info:
            if self._inter_side_tensor is not None:
                pos_features = self._inter_side_tensor[pos_items]
            else:
                raise ValueError(
                    "Requested side information but none provided in init."
                )

        # Extract positive context data if flagged
        pos_contexts = None
        if include_context:
            if self.context_labels:
                ctx_vals = self._inter_df.select(self.context_labels).to_numpy()
                pos_contexts = torch.tensor(ctx_vals, dtype=torch.long)
            else:
                raise ValueError(
                    "Requested context information but none provided in init."
                )

        # Negative sampling
        if neg_samples > 0:
            np.random.seed(seed)
            num_positives = len(pos_users)
            total_neg = num_positives * neg_samples

            # Same users for the negative part
            neg_users = pos_users.repeat_interleave(neg_samples)
            neg_users_np = neg_users.numpy()  # View numpy for faster collision check

            # Sample initial candidates
            neg_items_np = np.random.randint(
                0, self._niid, size=total_neg, dtype=np.int64
            )

            # Create a set for faster lookup O(1)
            positive_pairs = set(zip(pos_users_np, pos_items_np))

            # Helper function to check for collisions
            def get_invalid_indices(indices_subset):
                return np.array(
                    [
                        idx
                        for idx in indices_subset
                        if (neg_users_np[idx], neg_items_np[idx]) in positive_pairs
                    ],
                    dtype=np.int64,
                )

            # First check on entire set
            all_indices = np.arange(total_neg)
            invalid_indices = get_invalid_indices(all_indices)

            while len(invalid_indices) > 0:
                num_invalid = len(invalid_indices)

                # Generate new candidates only for invalid samples
                new_candidates = np.random.randint(
                    0, self._niid, size=num_invalid, dtype=np.int64
                )
                neg_items_np[invalid_indices] = new_candidates

                # Check new samples candidates
                invalid_indices = get_invalid_indices(invalid_indices)

            # Convert final tensors
            neg_items = torch.from_numpy(neg_items_np)
            neg_ratings = torch.zeros(total_neg, dtype=torch.float)

            # Extract side information for negatives if flagged
            neg_features = None
            if include_side_info and pos_features is not None:
                neg_features = self._inter_side_tensor[neg_items]

            # Repeat the context if flagged
            neg_contexts = None
            if include_context and pos_contexts is not None:
                # Negatives have the same context as the positive
                neg_contexts = pos_contexts.repeat_interleave(neg_samples, dim=0)

            # Final concatenation
            all_users = torch.cat([pos_users, neg_users])
            all_items = torch.cat([pos_items, neg_items])
            all_ratings = torch.cat([pos_ratings, neg_ratings])

            # Construct the final dataset
            tensors = [all_users, all_items, all_ratings]
            if include_side_info:
                all_features = torch.cat([pos_features, neg_features])
                tensors.append(all_features)
            if include_context:
                all_contexts = torch.cat([pos_contexts, neg_contexts])
                tensors.append(all_contexts)

            dataset = TensorDataset(*tensors)

        else:
            # Only positive interactions
            tensors = [pos_users, pos_items, pos_ratings]
            if include_side_info:
                tensors.append(pos_features)
            if include_context:
                tensors.append(pos_contexts)

            dataset = TensorDataset(*tensors)

        # Cache the dataset and return the dataloader
        self._cached_dataset[cache_key] = dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @typing.no_type_check
    def get_pos_neg_dataloader(
        self,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        low_memory: bool = False,
    ) -> DataLoader:
        """Create a PyTorch DataLoader with triplets for implicit feedback.

        Args:
            batch_size (int): The batch size that will be used to
                iterate over the interactions.
            shuffle (bool): Whether to shuffle the data.
            seed (int): Seed for Numpy random number generator for reproducibility.
            low_memory (bool): Whether to create the dataloader with a lazy approach.

        Returns:
            DataLoader: Yields triplets of (user, positive_item, negative_item).
        """
        if low_memory:
            sparse_matrix = self.get_sparse()

            lazy_dataset = LazyTripletDataset(
                sparse_matrix=sparse_matrix,
                niid=self._niid,
                seed=seed,
            )

            # Edge case: No interactions
            if len(lazy_dataset) == 0:
                return DataLoader(
                    TensorDataset(
                        torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])
                    )
                )

            return DataLoader(lazy_dataset, batch_size=batch_size, shuffle=shuffle)

        # Check if dataloader has been cached
        cache_key = "pos_neg"
        if cache_key in self._cached_dataset:
            dataset = self._cached_dataset[cache_key]
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Define main variables
        sparse_matrix = self.get_sparse()
        num_users = self._nuid
        num_items = self._niid
        num_positives = sparse_matrix.nnz

        # Preallocate arrays for triplets
        users_triplet = np.empty(num_positives, dtype=np.int64)
        positives_triplet = np.empty_like(users_triplet)
        negatives_triplet = np.empty_like(users_triplet)

        # Single call of module for efficiency
        np.random.seed(seed)  # Set the seed for reproducibility
        rint = np.random.randint  # Storing the module call is ever so slightly faster

        current_idx = 0
        for u in range(num_users):
            # Using sparse CSR matrix, get the indices of nnz columns
            # these will be the positive items
            start_ptr = sparse_matrix.indptr[u]
            end_ptr = sparse_matrix.indptr[u + 1]

            # Get indices of items interacted (positive items)
            user_pos = sparse_matrix.indices[start_ptr:end_ptr]
            user_count = len(user_pos)
            if user_count == 0:  # Skip the user if it has 0 interactions
                continue

            user_pos_set = set(user_pos)  # Efficient control using sets

            # Edge case: the user interacted with all the items
            if user_count == num_items:
                continue  # Skip the user if it interacted with all items

            # Iter through all the positive items
            for pos_item in user_pos:
                # Assign user and positive item to arrays
                users_triplet[current_idx] = u
                positives_triplet[current_idx] = pos_item

                # Until we find a valid negative, keep searching
                while True:
                    candidate_neg_item = rint(0, num_items)

                    if (
                        candidate_neg_item not in user_pos_set
                    ):  # If found save and break loop
                        negatives_triplet[current_idx] = candidate_neg_item
                        break

                current_idx += 1

        # Trim length based on possible triplets skipped
        users_triplet_trimmed = users_triplet[:current_idx]
        positives_triplet_trimmed = positives_triplet[:current_idx]
        negatives_triplet_trimmed = negatives_triplet[:current_idx]

        # Create Tensors for efficient data loading
        users_tensor = torch.LongTensor(users_triplet_trimmed)
        positives_tensor = torch.LongTensor(positives_triplet_trimmed)
        negatives_tensor = torch.LongTensor(negatives_triplet_trimmed)

        # Create final dataset
        dataset = TensorDataset(users_tensor, positives_tensor, negatives_tensor)
        self._cached_dataset[cache_key] = dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

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

        return np.sort(self._inter_df.select(self.rating_label).unique().to_numpy().flatten())


    def _to_sparse(self) -> csr_matrix:
        """This method will create the sparse representation of the data contained.

        This method must not be called if the sparse representation has already be defined.

        Returns:
            csr_matrix: Sparse representation of the transactions (CSR Format).
        """
        umap_df = nw.from_dict({
            self.user_label: list(self._umap.keys()),
            "__uidx__": list(self._umap.values())
        }, native_namespace=nw.get_native_namespace(self._inter_df))
        
        imap_df = nw.from_dict({
            self.item_label: list(self._imap.keys()),
            "__iidx__": list(self._imap.values())
        }, native_namespace=nw.get_native_namespace(self._inter_df))

        # Join
        mapped_df = self._inter_df.join(umap_df, on=self.user_label, how="inner") \
                                  .join(imap_df, on=self.item_label, how="inner")

        # FIX: Ordinamento deterministico.
        # Polars non garantisce l'ordine dopo un join, Pandas spesso sì.
        # Ordinando per User Index e Item Index rendiamo i dati identici per entrambi i backend.
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
        n_items = sparse_matrix.shape[1]

        # Get user_ids and item_ids from interactions
        user_ids, item_ids = sparse_matrix.nonzero()

        # Create a dictionary to store history for each user
        user_history_dict: dict[int, list] = {}
        for u, i in zip(user_ids, item_ids):
            if u not in user_history_dict:
                user_history_dict[u] = []
            user_history_dict[u].append(i)

        # Determine max history length for padding
        max_history_len = 0
        for user_id in range(self._nuid):
            if user_id in user_history_dict:
                max_history_len = max(max_history_len, len(user_history_dict[user_id]))

        # Initialize matrices
        self._history_matrix = torch.full(
            (self._nuid, max_history_len), fill_value=n_items, dtype=torch.long
        )
        self._history_lens = torch.zeros(self._nuid, dtype=torch.long)
        self._history_mask = torch.zeros(self._nuid, max_history_len, dtype=torch.float)

        # Populate matrices
        for user_id in range(self._nuid):
            if user_id in user_history_dict:
                items = torch.tensor(user_history_dict[user_id], dtype=torch.long)
                self._history_matrix[user_id, : len(items)] = items
                self._history_lens[user_id] = len(items)
                self._history_mask[user_id, : len(items)] = 1.0

        return self._history_matrix, self._history_lens, self._history_mask

    def __iter__(self) -> "Interactions":
        """This method will return the iterator of the interactions.

        Returns:
            Interactions: The iterator of the interactions.
        """
        self._index = 0
        if not isinstance(self._inter_sparse, csr_matrix):
            self._to_sparse()
        return self

    def __next__(self) -> csr_matrix:
        """This method will iterate over the sparse data.

        Returns:
            csr_matrix: The csr representation of data.
        Raises:
            StopIteration: If the end of the data is reached.
            ValueError: If the sparse matrix is None.
        """
        if self._index >= self._og_nuid:
            raise StopIteration
        if self._inter_sparse is None:
            raise ValueError("The sparse matrix is None.")

        start = self._index
        end = min(start + self.batch_size, self._og_nuid)
        self._index = end
        return self._inter_sparse[start:end]

    def __len__(self) -> int:
        """This method calculates the length of the interactions.

        Length will be defined as the number of ratings.

        Returns:
            int: number of ratings present in the structure.
        """
        return sum(len(ir) for _, ir in self._inter_dict.items())
