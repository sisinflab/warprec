from typing import Tuple, Optional, List, Set, Any, Dict

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from pandas import DataFrame
from scipy.sparse import csr_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import RatingType
from warprec.utils.logger import logger


class EvaluationDataset(TorchDataset):
    """PyTorch Dataset to yield (train, eval, user_idx) in batch,
    converting sparse matrices in dense tensors.

    Args:
        eval_interactions (csr_matrix): Sparse matrix of evaluation interactions.
    """

    def __init__(
        self,
        eval_interactions: csr_matrix,
    ):
        self.num_users, self.num_items = eval_interactions.shape
        self.eval_interactions = eval_interactions

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        eval_row = self.eval_interactions.getrow(idx)
        eval_batch = torch.from_numpy(eval_row.toarray()).to(torch.float32).squeeze(0)

        return eval_batch, idx


class NegativeEvaluationDataset(TorchDataset):
    """PyTorch Dataset to yield (pos_item, neg_item, user_idx) in batch,
    converting sparse matrices in dense tensors.

    For each user, positive items are retrieve and a number of negatives
    is sampled randomly.

    Args:
        train_interactions (csr_matrix): Sparse matrix of training interactions.
        eval_interactions (csr_matrix): Sparse matrix of evaluation interactions.
        num_negatives (int): Number of negatives to sample per user.
        seed (int): Random seed for negative sampling.
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_interactions: csr_matrix,
        num_negatives: int = 99,
        seed: int = 42,
    ):
        super().__init__()
        self.num_users, self.num_items = train_interactions.shape
        self.num_negatives = num_negatives

        # Init lists for faster access
        self.users_with_positives = []
        self.positive_items_list = []
        self.negative_items_list = []

        # Get positive interactions in eval set
        eval_positives_by_user = [
            eval_interactions.indices[
                eval_interactions.indptr[u] : eval_interactions.indptr[u + 1]
            ]
            for u in range(self.num_users)
        ]

        # Set the random seed
        np.random.seed(seed)

        for u in range(self.num_users):
            # If no pos interaction in eval set, skip
            if len(eval_positives_by_user[u]) == 0:
                continue

            self.users_with_positives.append(u)
            self.positive_items_list.append(
                torch.tensor(eval_positives_by_user[u], dtype=torch.long)
            )

            # Obtain all positive (train/eval)
            train_positives = train_interactions[u].indices
            user_all_positives = np.union1d(eval_positives_by_user[u], train_positives)

            # Compute candidates
            candidate_negatives_count = self.num_items - len(user_all_positives)
            num_to_sample = min(self.num_negatives, candidate_negatives_count)

            # Randomly sample negatives until correctly sampled
            if num_to_sample > 0:
                negatives: Set[Any] = set()
                while len(negatives) < num_to_sample:
                    candidate = np.random.randint(0, self.num_items)
                    if candidate not in user_all_positives:
                        negatives.add(candidate)
                self.negative_items_list.append(
                    torch.tensor(list(negatives), dtype=torch.long)
                )
            else:
                self.negative_items_list.append(torch.tensor([], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.users_with_positives)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        """Yield data for a single user.

        Args:
            idx (int): Index of the user.

        Returns:
            Tuple[Tensor, Tensor, int]:
                - positive_items (Tensor): Tensor with positive items.
                - negative_items (Tensor): Tensor with negative items.
                - idx (int): User index.
        """
        # Fast access to pre-computed pos-neg interactions
        positive_items = self.positive_items_list[idx]
        negative_items = self.negative_items_list[idx]

        # Retrieve valid user idxs
        user_idx = self.users_with_positives[idx]

        return (
            positive_items,
            negative_items,
            user_idx,
        )


class EvaluationDataLoader(DataLoader):
    """Custom DataLoader to yield tuple (train, eval) in batch size."""

    def __init__(
        self,
        eval_interactions: csr_matrix,
        batch_size: int = 1024,
        **kwargs,
    ):
        dataset = EvaluationDataset(
            eval_interactions=eval_interactions,
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=False, **kwargs)


class NegativeEvaluationDataLoader(DataLoader):
    """DataLoader for evaluation with negative sampling."""

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_interactions: csr_matrix,
        num_negatives: int = 99,
        seed: int = 42,
        batch_size: int = 1024,
        **kwargs,
    ):
        self.num_items = train_interactions.shape[1]  # Used as padding index

        dataset = NegativeEvaluationDataset(
            train_interactions=train_interactions,
            eval_interactions=eval_interactions,
            num_negatives=num_negatives,
            seed=seed,
        )

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    def _collate_fn(
        self,
        batch: List[Tuple[Tensor, Tensor, int]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Custom collate_fn to handle negative sampling in evaluation."""
        positive_tensors, negative_tensors, user_indices = zip(*batch)

        # User indices will be a list of ints, so we convert it
        user_indices_tensor = torch.tensor(list(user_indices), dtype=torch.long)

        # We use the pad_sequence utility to pad item indices
        # in order to have all tensor of the same size
        positives_padded = pad_sequence(
            positive_tensors,  # type: ignore[arg-type]
            batch_first=True,
            padding_value=self.num_items,
        )
        negatives_padded = pad_sequence(
            negative_tensors,  # type: ignore[arg-type]
            batch_first=True,
            padding_value=self.num_items,
        )

        return positives_padded, negatives_padded, user_indices_tensor


class Dataset:
    """The definition of the Dataset class that will handle transaction data.

    Args:
        train_data (DataFrame): The train data.
        eval_data (Optional[DataFrame]): The evaluation data.
        side_data (Optional[DataFrame]): The side information data.
        user_cluster (Optional[DataFrame]): The user cluster data.
        item_cluster (Optional[DataFrame]): The item cluster data.
        batch_size (int): The batch size that will be used evaluation.
        rating_type (RatingType): The type of rating used.
        user_id_label (str): The label of the user id column.
        item_id_label (str): The label of the item id column.
        rating_label (str): The label of the rating column.
        timestamp_label (str): The label of the timestamp column.
        cluster_label (str): The label of the cluster column.
        context_labels (Optional[List[str]]): The list of labels of the
            contextual data.
        precision (Any): The precision of the internal representation of the data.
        evaluation_set (str): The type of evaluation set. Can either be 'Test'
            or 'Validation'.

    Attributes:
        train_set (Interactions): Training set used by recommendation models.
        eval_set (Interactions): Evaluation set used by recommendation models.
        train_session (Sessions): Training session used by sequential models.
        user_cluster (Optional[dict]): User cluster information.
        item_cluster (Optional[dict]): Item cluster information.

    Raises:
        ValueError: If the evaluation_set is not supported.
    """

    train_set: Interactions = None
    eval_set: Interactions = None
    train_session: Sessions = None
    user_cluster: Optional[dict] = None
    item_cluster: Optional[dict] = None

    def __init__(
        self,
        train_data: DataFrame,
        eval_data: Optional[DataFrame] = None,
        side_data: Optional[DataFrame] = None,
        user_cluster: Optional[DataFrame] = None,
        item_cluster: Optional[DataFrame] = None,
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        rating_label: str = None,
        timestamp_label: str = None,
        cluster_label: str = None,
        context_labels: Optional[List[str]] = None,
        precision: Any = np.float32,
        evaluation_set: str = "Test",
    ):
        # Check evaluation set
        if evaluation_set not in ["Test", "Validation"]:
            raise ValueError("Evaluation set must be either 'Test' or 'Validation'.")

        # Initializing variables
        self._nuid: int = 0
        self._niid: int = 0
        self._nfeat: int = 0
        self._max_seq_len: int = 0
        self._umap: dict[Any, int] = {}
        self._imap: dict[Any, int] = {}
        self._context_maps: dict[str, dict[Any, int]] = {}
        self._context_dims: dict[str, int] = {}
        self._feat_lookup: Tensor = None
        self._uc: Tensor = None
        self._ic: Tensor = None
        self._stash: Dict[
            str, Any
        ] = {}  # Stash will be used by the user for custom needs

        # Initialize the dataset dataloader
        self._precomputed_dataloader: Dict[str, Any] = {}

        # If side information data has been provided, we filter the main dataset
        if side_data is not None:
            train_data, eval_data = self._filter_data(
                train=train_data,
                eval=eval_data,
                filter_data=side_data,
                label=item_id_label,
            )

        # Define dimensions that will lead the experiment
        self._nuid = train_data[user_id_label].nunique()
        self._niid = train_data[item_id_label].nunique()
        self._nfeat = len(side_data.columns) - 1 if side_data is not None else 0
        self._batch_size = batch_size

        # Values that will be used to calculate mappings
        _uid = train_data[user_id_label].unique()
        _iid = train_data[item_id_label].unique()

        # Calculate mapping for users and items
        self._umap = {user: i for i, user in enumerate(_uid)}
        self._imap = {item: i for i, item in enumerate(_iid)}

        # Process contextual data
        if context_labels:
            train_data = self._process_context_data(
                train_data, context_labels, fit=True
            )

            if eval_data is not None:
                eval_data = self._process_context_data(
                    eval_data, context_labels, fit=False
                )

        # Process the side information data and filter not valid columns
        self.side = None
        if side_data is not None:
            self.side = self._process_side_data(side_data, item_id_label)

        # Save user and item cluster information inside the dataset
        self.user_cluster = (
            {
                self._umap[user_id]: cluster
                for user_id, cluster in zip(
                    user_cluster[user_id_label], user_cluster[cluster_label]
                )
                if user_id in self._umap
            }
            if user_cluster is not None
            else None
        )
        self.item_cluster = (
            {
                self._imap[item_id]: cluster
                for item_id, cluster in zip(
                    item_cluster[item_id_label], item_cluster[cluster_label]
                )
                if item_id in self._imap
            }
            if item_cluster is not None
            else None
        )

        # Pre compute lookup tensors for user clusters
        if self.user_cluster is not None:
            unique_user_clusters = sorted(set(self.user_cluster.values()))
            user_cluster_remap = {
                cud: idx + 1 for idx, cud in enumerate(unique_user_clusters)
            }  # Use appropriate indexes for clusters
            self._uc = torch.zeros(self._nuid, dtype=torch.long)
            for u, c in self.user_cluster.items():
                self._uc[u] = user_cluster_remap[c]
        else:
            self._uc = torch.ones(self._nuid, dtype=torch.long)

        # Pre compute lookup tensors for item clusters
        if self.item_cluster is not None:
            unique_item_clusters = sorted(set(self.item_cluster.values()))
            item_cluster_remap = {
                cid: idx + 1 for idx, cid in enumerate(unique_item_clusters)
            }  # Use appropriate indexes for clusters
            self._ic = torch.zeros(self._niid, dtype=torch.long)
            for i, c in self.item_cluster.items():
                self._ic[i] = item_cluster_remap[c]
        else:
            self._ic = torch.ones(self._niid, dtype=torch.long)

        # Create the main data structures
        self.train_set = self._create_inner_set(
            train_data,
            side_data=self.side,
            user_cluster=self.user_cluster,
            item_cluster=self.item_cluster,
            batch_size=batch_size,
            rating_type=rating_type,
            rating_label=rating_label,
            context_labels=context_labels,
            precision=precision,
        )

        if eval_data is not None:
            self.eval_set = self._create_inner_set(
                eval_data,
                side_data=self.side,
                user_cluster=self.user_cluster,
                item_cluster=self.item_cluster,
                header_msg=evaluation_set,
                batch_size=batch_size,
                rating_type=rating_type,
                rating_label=rating_label,
                context_labels=context_labels,
                precision=precision,
            )

        # Save side information inside the dataset
        if self.side is not None:
            # Create the lookup tensor for side information features
            self._feat_lookup = torch.tensor(
                self.train_set._inter_side.iloc[:, 1:].values
            ).float()

        # Sequential recommendation sessions
        self.train_session = Sessions(
            train_data,
            self._umap,
            self._imap,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            timestamp_label=timestamp_label,
            context_labels=context_labels,
        )

    def _filter_data(
        self,
        train: DataFrame,
        eval: DataFrame,
        filter_data: DataFrame,
        label: str,
    ) -> Tuple[DataFrame, DataFrame]:
        """Filter the data based on a given additional information set and label.

        Args:
            train (DataFrame): The train set.
            eval (DataFrame): The evaluation set.
            filter_data (DataFrame): The additional information dataset.
            label (str): The label used to filter the data.

        Returns:
            Tuple[DataFrame, DataFrame]:
                - DataFrame: The filtered train set.
                - DataFrame: The filtered evaluation set.
        """
        # Compute shared data points first
        shared_data = set(train[label]).intersection(filter_data[label])

        # Count the number of data points before filtering
        train_data_before_filter = train[label].nunique()

        # Filter all the data based on data points present in both train data and filter.
        # This procedure is fundamental because we need dimensions to match
        train = train[train[label].isin(shared_data)]
        eval = eval[eval[label].isin(shared_data)]
        filter_data = filter_data[filter_data[label].isin(shared_data)]

        # Count the number of data points after filtering
        train_data_after_filter = train[label].nunique()

        logger.attention(
            ""
            f"Filtered out {train_data_before_filter - train_data_after_filter} {label}."
        )

        return train, eval

    def _create_inner_set(
        self,
        data: DataFrame,
        side_data: Optional[DataFrame] = None,
        user_cluster: Optional[dict] = None,
        item_cluster: Optional[dict] = None,
        header_msg: str = "Train",
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        rating_label: str = None,
        context_labels: Optional[List[str]] = None,
        precision: Any = np.float32,
    ) -> Interactions:
        """Functionality to create Interaction data from DataFrame.

        Args:
            data (DataFrame): The data used to create the interaction object.
            side_data (Optional[DataFrame]): The side data information about the dataset.
            user_cluster (Optional[dict]): The user cluster information.
            item_cluster (Optional[dict]): The item cluster information.
            header_msg (str): The header of the logger output.
            batch_size (int): The batch size of the interaction.
            rating_type (RatingType): The type of rating used.
            rating_label (str): The label of the rating column.
            context_labels (Optional[List[str]]): The list of labels of the
                contextual data.
            precision (Any): The precision that will be used to store interactions.

        Returns:
            Interactions: The final interaction object.
        """
        inter_set = Interactions(
            data,
            (self._nuid, self._niid),
            self._umap,
            self._imap,
            side_data=side_data,
            user_cluster=user_cluster,
            item_cluster=item_cluster,
            batch_size=batch_size,
            rating_type=rating_type,
            rating_label=rating_label,
            context_labels=context_labels,
            precision=precision,
        )
        nuid, niid = inter_set.get_dims()
        transactions = inter_set.get_transactions()
        logger.stat_msg(
            (
                f"Number of users: {nuid}      "
                f"Number of items: {niid}      "
                f"Transactions: {transactions}"
            ),
            f"{header_msg} set",
        )

        return inter_set

    def _process_context_data(
        self, df: DataFrame, context_labels: List[str], fit: bool = False
    ) -> DataFrame:
        """Processes context columns: creates mappings (if fit=True) and converts
        values to integers.

        Strategy:
        - Index 0 is reserved for <UNK> (Unknown) or Padding.
        - Actual values start from index 1.

        Args:
            df (DataFrame): The DataFrame containing transaction data
                with contextual information.
            context_labels (List[str]): The labels of the columns containing
                the contextual information.
            fit (bool): Wether or not to fit the dataset on the DataFrame
                data. Usually used on the train set.

        Returns:
            DataFrame: The processed data.

        Raises:
            ValueError: If the contextual column names are not
                present the the DataFrame.
        """
        df_processed = df.copy()

        for col in context_labels:
            if col not in df_processed.columns:
                raise ValueError(f"Context label '{col}' not found in DataFrame.")

            if fit:
                # Create mapping based on unique values in this column
                uniques = df_processed[col].unique()
                # Start from 1, reserve 0 for Unknown
                mapping = {val: i + 1 for i, val in enumerate(uniques)}

                self._context_maps[col] = mapping
                # Dimension is len(uniques) + 1 (for the UNK token)
                self._context_dims[col] = len(uniques) + 1

                logger.msg(f"Context '{col}': found {len(uniques)} unique values.")
            else:
                # Use existing mapping
                mapping = self._context_maps.get(col)
                if mapping is None:
                    raise ValueError(
                        f"Mapping for context '{col}' not found. Fit on train first."
                    )

            # Apply mapping
            # Values not in mapping become NaN, then fill with 0 (UNK), then cast to int
            df_processed[col] = df_processed[col].map(mapping).fillna(0).astype(int)

            # Optional: Check for OOV (Out Of Vocabulary) in Eval
            if not fit:
                num_unk = (df_processed[col] == 0).sum()
                if num_unk > 0:
                    logger.attention(
                        f"Context '{col}': found {num_unk} unknown values in Eval set (mapped to 0)."
                    )

        return df_processed

    def _process_side_data(
        self, side_data: DataFrame, item_id_label: str
    ) -> Optional[DataFrame]:
        """Process side information data and filter out invalid columns

        Args:
            side_data (DataFrame): The side data DataFrame.
            item_id_label (str): The label of the item ID.

        Returns:
            Optional[DataFrame]: In case the method fails in advance or
                cleaned DataFrame with valid columns.

        Raises:
            ValueError: When the item ID is not found in side data.
        """
        # Check for item ID label
        if item_id_label not in side_data.columns:
            raise ValueError("Item ID label not found inside side information data.")

        # View only the needed columns
        feature_cols = side_data.drop(columns=[item_id_label])

        # Find all the numeric columns
        numeric_cols = []
        filtered_cols = []
        for col in feature_cols.columns:
            series_with_nan = feature_cols[col].replace("", np.nan).copy()

            # Count NaNs before and after conversion
            nulls_before = series_with_nan.isnull().sum()
            coerced_series = pd.to_numeric(series_with_nan, errors="coerce")
            nulls_after = coerced_series.isnull().sum()

            # If the number of nulls remain un-changed, this will be considered
            # a numeric column, else we filter it
            if nulls_after > nulls_before:
                filtered_cols.append(col)
            else:
                numeric_cols.append(col)

        # Check if there are valid columns
        if not numeric_cols:
            logger.negative(
                "No valid columns found. Side information will not be available."
            )
            return None

        # Log the filtered out columns
        if filtered_cols:
            logger.attention(
                "Side information contains non-numeric values. "
                f"The following columns will be ignored: {filtered_cols}"
            )

        # Update the inner DataFrame with valid data
        valid_columns = [item_id_label] + numeric_cols
        return (
            side_data[valid_columns].copy().fillna(0)
        )  # Missing values will be filled with zeros

    def get_evaluation_dataloader(self) -> EvaluationDataLoader:
        """Retrieve the EvaluationDataLoader for the dataset.

        Returns:
            EvaluationDataLoader: DataLoader that yields batches of interactions
                (eval_batch, user_indices).
        """
        key = "full"
        if key not in self._precomputed_dataloader:
            eval_sparse = self.eval_set.get_sparse()

            self._precomputed_dataloader[key] = EvaluationDataLoader(
                eval_interactions=eval_sparse,
                batch_size=self._batch_size,
            )

        return self._precomputed_dataloader[key]

    def get_neg_evaluation_dataloader(
        self,
        num_negatives: int = 99,
        seed: int = 42,
    ) -> NegativeEvaluationDataLoader:
        """Retrieve the NegativeEvaluationDataLoader for the dataset.

        Args:
            num_negatives (int): Number of negative samples per user.
            seed (int): Random seed for negative sampling.

        Returns:
            NegativeEvaluationDataLoader: DataLoader that yields batches
                of interactions (pos_items, neg_items, user_indices)
        """
        key = f"neg_{num_negatives}_{seed}"

        if key not in self._precomputed_dataloader:
            train_sparse = self.train_set.get_sparse()
            eval_sparse = self.eval_set.get_sparse()

            self._precomputed_dataloader[key] = NegativeEvaluationDataLoader(
                train_interactions=train_sparse,
                eval_interactions=eval_sparse,
                num_negatives=num_negatives,
                batch_size=self._batch_size,
                seed=seed,
            )

        return self._precomputed_dataloader[key]

    def get_dims(self) -> Tuple[int, int]:
        """Returns the dimensions of the data.

        Returns:
            Tuple[int, int]:
                int: Number of unique user_ids.
                int: Number of unique item_ids.
        """
        return (self._nuid, self._niid)

    def get_context_dims(self) -> Dict[str, int]:
        """Returns the dimensions (vocab size) of each context feature.

        Returns:
            Dict[str, int]: A dictionary containing the name of the
                context feature and the vocab size.
        """
        return self._context_dims

    def get_mappings(self) -> Tuple[dict, dict]:
        """Returns the mapping used for this dataset.

        Returns:
            Tuple[dict, dict]:
                dict: Mapping of user_id -> user_idx.
                dict: Mapping of item_id -> item_idx.
        """
        return (self._umap, self._imap)

    def get_inverse_mappings(self) -> Tuple[dict, dict]:
        """Returns the inverse of the mapping.

        Returns:
            Tuple[dict, dict]:
                dict: Mapping of user_idx -> user_id.
                dict: Mapping of item_idxs -> item_id.
        """
        return {v: k for k, v in self._umap.items()}, {
            v: k for k, v in self._imap.items()
        }

    def get_features_lookup(self) -> Optional[Tensor]:
        """This method retrieves the lookup tensor for side information features.

        Returns:
            Optional[Tensor]: Lookup tensor for side information features.
        """
        return self._feat_lookup

    def get_user_cluster(self) -> Tensor:
        """This method retrieves the lookup tensor for user clusters.

        Returns:
            Tensor: Lookup tensor for user clusters.
        """
        return self._uc

    def get_item_cluster(self) -> Tensor:
        """This method retrieves the lookup tensor for item clusters.

        Returns:
            Tensor: Lookup tensor for item clusters.
        """
        return self._ic

    def get_stash(self) -> Dict[str, Any]:
        """This method retrieves the stash dictionary
        containing all user custom structures.

        Returns:
            Dict[str, Any]: The stash dictionary.
        """
        return self._stash

    def info(self) -> dict:
        """This method returns the main information of the
        dataset in dict format.

        Returns:
            dict: The dictionary with the main information of
                the dataset.
        """
        base_info = {
            "n_items": self._niid,
            "n_users": self._nuid,
            "n_features": self._nfeat,
            "item_mapping": self._imap,
            "user_mapping": self._umap,
        }

        # Optionally add contextual dimensions if present
        if self._context_dims:
            base_info["context_dims"] = self._context_dims

        return base_info

    def add_to_stash(self, key: str, value: Any):
        """Add a custom structure to the stash.

        Args:
            key (str): The key of the custom structure.
            value (Any): The value of the custom structure.
        """
        self._stash[key] = value

    def update_mappings(self, user_mapping: dict, item_mapping: dict):
        """Update the mappings of the dataset.

        Args:
            user_mapping (dict): The mapping of user_id -> user_idx.
            item_mapping (dict): The mapping of item_id -> item_idx.
        """
        self.umap = user_mapping
        self.imap = item_mapping

    def clear_cache(self):
        """Clear the cache of inner data structures."""
        self.train_set.clear_dataset_cache()
        self.train_session.clear_dataset_cache()
