# pylint: disable = too-many-branches, too-many-statements
from typing import Tuple, Optional, List, Any, Dict

import torch
import numpy as np
from torch import Tensor

import narwhals as nw
from narwhals.typing import FrameT

from warprec.data.entities import Interactions, Sessions
from warprec.data.eval_loaders import (
    EvaluationDataLoader,
    SampledEvaluationDataLoader,
    ContextualEvaluationDataLoader,
    SampledContextualEvaluationDataLoader,
)
from warprec.utils.enums import RatingType
from warprec.utils.logger import logger


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
        train_data: FrameT,
        eval_data: Optional[FrameT] = None,
        side_data: Optional[FrameT] = None,
        user_cluster: Optional[FrameT] = None,
        item_cluster: Optional[FrameT] = None,
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
        
        # Convert inputs to Narwhals
        train_data = nw.from_native(train_data, pass_through=True)
        eval_data = nw.from_native(eval_data, pass_through=True) if eval_data is not None else None
        side_data = nw.from_native(side_data, pass_through=True) if side_data is not None else None
        user_cluster = nw.from_native(user_cluster, pass_through=True) if user_cluster is not None else None
        item_cluster = nw.from_native(item_cluster, pass_through=True) if item_cluster is not None else None

        # Initializing variables
        self._nuid: int = 0
        self._niid: int = 0
        self._nfeat: int = 0
        self._max_seq_len: int = 0
        self._umap: dict[Any, int] = {}
        self._imap: dict[Any, int] = {}
        self._feature_maps: dict[str, dict[Any, int]] = {}
        self._feature_dims: dict[str, int] = {}
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
                train_set=train_data,
                eval_set=eval_data,
                filter_data=side_data,
                label=item_id_label,
            )

        # Define dimensions that will lead the experiment
        self._nuid = train_data.select(nw.col(user_id_label).n_unique()).item()
        self._niid = train_data.select(nw.col(item_id_label).n_unique()).item()
        self._nfeat = (len(side_data.columns) - 1) if side_data is not None else 0
        self.batch_size = batch_size

        # Values that will be used to calculate mappings
        _uid = train_data.select(user_id_label).unique().to_dict(as_series=False)[user_id_label]
        _iid = train_data.select(item_id_label).unique().to_dict(as_series=False)[item_id_label]

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
        if user_cluster is not None:
            uc_dict = user_cluster.select(user_id_label, cluster_label).to_dict(as_series=False)
            self.user_cluster = {
                self._umap[user_id]: cluster
                for user_id, cluster in zip(uc_dict[user_id_label], uc_dict[cluster_label])
                if user_id in self._umap
            }
        else:
            self.user_cluster = None

        if item_cluster is not None:
            ic_dict = item_cluster.select(item_id_label, cluster_label).to_dict(as_series=False)
            self.item_cluster = {
                self._imap[item_id]: cluster
                for item_id, cluster in zip(ic_dict[item_id_label], ic_dict[cluster_label])
                if item_id in self._imap
            }
        else:
            self.item_cluster = None

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

        # Add padding value to item clusters
        padding_value = torch.tensor([0], dtype=torch.long)
        self._ic = torch.cat((self._ic, padding_value), dim=0)

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
            native_ns = nw.get_native_namespace(self.side)
            
            # Creiamo il DF di mapping: ItemID -> ItemIndex
            mapping_data = {
                item_id_label: list(self._imap.keys()),
                "__item_idx__": list(self._imap.values())
            }
            mapping_df = nw.from_dict(mapping_data, native_namespace=native_ns)

            # 2. Eseguiamo un LEFT JOIN partendo dal mapping.
            # Questo garantisce che:
            #   a) Abbiamo una riga per ogni item nel sistema (anche se non ha side info)
            #   b) Possiamo ordinare per __item_idx__
            aligned_side = mapping_df.join(self.side, on=item_id_label, how="left")

            # 3. Ordiniamo esplicitamente per l'indice dell'item
            aligned_side = aligned_side.sort("__item_idx__")

            # 4. Selezioniamo le colonne delle feature
            # Nota: escludiamo item_id_label e la colonna temporanea __item_idx__
            feat_cols = [c for c in self.side.columns if c != item_id_label]
            
            # 5. Riempiamo eventuali buchi (se un item non aveva side info) con 0
            # e estraiamo i valori ordinati
            side_values = aligned_side.select(feat_cols).with_columns(
                nw.all().fill_null(0)
            ).to_numpy()
            
            feature_lookup = torch.tensor(side_values).float()

            # Add a padding row (per l'indice 0 riservato al padding/unknown se usato dal modello)
            # Nota: Se il tuo mapping parte da 0, questa riga è extra. 
            # Se il tuo mapping parte da 1 (come spesso accade in NLP/RecSys per riservare 0), serve.
            # Assumendo che self._imap parta da 0, questo aggiunge una riga in fondo.
            padding_row = torch.zeros((1, feature_lookup.size(1)))
            self._feat_lookup = torch.cat([feature_lookup, padding_row], dim=0)

        # Sequential recommendation sessions
        # self.train_session = Sessions(
        #     train_data,
        #     self._umap,
        #     self._imap,
        #     user_id_label=user_id_label,
        #     item_id_label=item_id_label,
        #     timestamp_label=timestamp_label,
        #     context_labels=context_labels,
        # )

    def _filter_data(
        self,
        train_set: FrameT,
        eval_set: FrameT,
        filter_data: FrameT,
        label: str,
    ) -> Tuple[FrameT, FrameT]:
        """Filter the data based on a given additional information set and label.

        Args:
            train_set (DataFrame): The train set.
            eval_set (DataFrame): The evaluation set.
            filter_data (DataFrame): The additional information dataset.
            label (str): The label used to filter the data.

        Returns:
            Tuple[DataFrame, DataFrame]:
                - DataFrame: The filtered train set.
                - DataFrame: The filtered evaluation set.
        """
        train_vals = set(train_set.select(label).unique().to_dict(as_series=False)[label])
        filter_vals = set(filter_data.select(label).unique().to_dict(as_series=False)[label])
        
        shared_data = list(train_vals.intersection(filter_vals))

        # Count the number of data points before filtering
        train_data_before_filter = train_set.select(nw.col(label).n_unique()).item()

        # Filter all the data based on data points present in both train data and filter.
        # Narwhals: use .filter(nw.col(label).is_in(shared_data))
        train_set = train_set.filter(nw.col(label).is_in(shared_data))
        if eval_set is not None:
            eval_set = eval_set.filter(nw.col(label).is_in(shared_data))
        
        # Note: The original code also filtered 'filter_data', but didn't return it.
        # If side_data is modified in place in original pandas, it matters. 
        # Here we return modified train/eval. Side data is usually static ref.
        # If we need to filter side_data for internal consistency:
        # filter_data = filter_data.filter(nw.col(label).is_in(shared_data)) 
        # (But this function returns only train/eval)

        # Count the number of data points after filtering
        train_data_after_filter = train_set.select(nw.col(label).n_unique()).item()

        logger.attention(
            ""
            f"Filtered out {train_data_before_filter - train_data_after_filter} {label}."
        )

        return train_set, eval_set

    def _create_inner_set(
        self,
        data: FrameT,
        side_data: Optional[FrameT] = None,
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

    def _process_side_data(
        self, side_data: FrameT, item_id_label: str
    ) -> Optional[FrameT]:
        """Process side information data.

        It maps categorical/numerical values to integer indices suitable for Embeddings.

        Args:
            side_data (DataFrame): The side data DataFrame.
            item_id_label (str): The label of the item ID.

        Returns:
            Optional[DataFrame]: The processed DataFrame with integer indices.

        Raises:
            ValueError: If the item ID label is not found in side_data.
        """
        # Check for item ID label
        if item_id_label not in side_data.columns:
            raise ValueError("Item ID label not found inside side information data.")

        df_processed = side_data.clone()

        # Identify feature columns (all except item_id_label)
        feature_cols = [c for c in df_processed.columns if c != item_id_label]

        if not feature_cols:
            logger.negative("No feature columns found in side data.")
            return None

        # Iterate over each feature column to create mappings and transform data
        for col in feature_cols:
            # Fill missing data with a placeholder
            df_processed = df_processed.with_columns(nw.col(col).fill_null("UNK"))

            # Create a mapping from unique values to integers
            unique_vals = df_processed.select(col).unique().to_dict(as_series=False)[col]

            # Map: value -> index (starting from 1, reserving 0 for unknown/padding)
            mapping = {val: i + 1 for i, val in enumerate(unique_vals)}

            # Save the mapping
            self._feature_maps[col] = mapping

            # Save the dimension (vocab size + 1 for UNK)
            self._feature_dims[col] = len(unique_vals) + 1

            # Apply the mapping to the columns
            map_df = nw.from_dict({
                col: list(mapping.keys()), 
                f"{col}_idx": list(mapping.values())
            }, native_namespace=nw.get_native_namespace(df_processed))
            
            # Join and replace
            df_processed = df_processed.join(map_df, on=col, how="left")
            
            # Fill nulls (if any value wasn't in mapping, though here we built mapping from data)
            # and replace original column
            df_processed = df_processed.with_columns(
                nw.col(f"{col}_idx").fill_null(0).cast(nw.Int64).alias(col)
            ).drop(f"{col}_idx")

            logger.msg(f"Side Feature '{col}': found {len(unique_vals)} unique values.")

        return df_processed

    def _process_context_data(
        self, df: FrameT, context_labels: List[str], fit: bool = False
    ) -> FrameT:
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
        df_processed = df.clone()

        for col in context_labels:
            if col not in df_processed.columns:
                raise ValueError(f"Context label '{col}' not found in DataFrame.")

            if fit:
                # Create mapping based on unique values in this column
                uniques = df_processed.select(col).unique().to_dict(as_series=False)[col]
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
            map_df = nw.from_dict({
                col: list(mapping.keys()), 
                f"{col}_idx": list(mapping.values())
            }, native_namespace=nw.get_native_namespace(df_processed))

            # Join
            df_processed = df_processed.join(map_df, on=col, how="left")

            # Optional: Check for OOV (Out Of Vocabulary) in Eval
            if not fit:
                # Check for nulls in the mapped column before filling
                num_unk = df_processed.select(nw.col(f"{col}_idx").is_null().sum()).item()
                if num_unk > 0:
                    logger.attention(
                        f"Context '{col}': found {num_unk} unknown values in Eval set (mapped to 0)."
                    )

            # Finalize column
            df_processed = df_processed.with_columns(
                nw.col(f"{col}_idx").fill_null(0).cast(nw.Int64).alias(col)
            ).drop(f"{col}_idx")

        return df_processed

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
                batch_size=self.batch_size,
            )

        return self._precomputed_dataloader[key]

    def get_sampled_evaluation_dataloader(
        self,
        num_negatives: int = 99,
        seed: int = 42,
    ) -> SampledEvaluationDataLoader:
        """Retrieve the SampledEvaluationDataLoader for the dataset.

        Args:
            num_negatives (int): Number of negative samples per user.
            seed (int): Random seed for negative sampling.

        Returns:
            SampledEvaluationDataLoader: DataLoader that yields batches
                of interactions (pos_items, neg_items, user_indices)
        """
        key = f"neg_{num_negatives}_{seed}"

        if key not in self._precomputed_dataloader:
            train_sparse = self.train_set.get_sparse()
            eval_sparse = self.eval_set.get_sparse()

            self._precomputed_dataloader[key] = SampledEvaluationDataLoader(
                train_interactions=train_sparse,
                eval_interactions=eval_sparse,
                num_negatives=num_negatives,
                batch_size=self.batch_size,
                seed=seed,
            )

        return self._precomputed_dataloader[key]

    def get_contextual_evaluation_dataloader(self) -> ContextualEvaluationDataLoader:
        """Retrieve the ContextualEvaluationDataLoader for the dataset.

        This loader is specific for Context-Aware Recommender Systems.
        It iterates over transactions (User, Item, Context) instead of Users.

        Returns:
            ContextualEvaluationDataLoader: The contextual data loader.
        """
        key = "full_contextual"
        if key not in self._precomputed_dataloader:
            eval_data = self.eval_set.get_df()

            # Retrieve labels to pre-compute eval data
            user_label = self.eval_set.user_label
            item_label = self.eval_set.item_label
            context_labels = self.eval_set.context_labels

            # Map the evaluation dataset to ensure consistency
            # Create mapping DFs
            u_map_df = nw.from_dict({
                user_label: list(self._umap.keys()), 
                f"{user_label}_idx": list(self._umap.values())
            }, native_namespace=nw.get_native_namespace(eval_data))
            
            i_map_df = nw.from_dict({
                item_label: list(self._imap.keys()), 
                f"{item_label}_idx": list(self._imap.values())
            }, native_namespace=nw.get_native_namespace(eval_data))

            # Apply mappings via Join
            eval_data = eval_data.join(u_map_df, on=user_label, how="inner") \
                                 .join(i_map_df, on=item_label, how="inner")

            # Replace columns with mapped indices
            eval_data = eval_data.with_columns(
                nw.col(f"{user_label}_idx").alias(user_label),
                nw.col(f"{item_label}_idx").alias(item_label)
            ).drop(f"{user_label}_idx", f"{item_label}_idx")

            # dropna is implicit in inner join for the mapped keys, 
            # but if we need to drop other nulls:
            # eval_data = eval_data.drop_nulls(subset=[user_label, item_label])

            self._precomputed_dataloader[key] = ContextualEvaluationDataLoader(
                eval_data=eval_data,
                user_id_label=user_label,
                item_id_label=item_label,
                context_labels=context_labels,
                batch_size=self.batch_size,
            )

        return self._precomputed_dataloader[key]

    def get_sampled_contextual_evaluation_dataloader(
        self,
        num_negatives: int = 99,
        seed: int = 42,
    ) -> SampledContextualEvaluationDataLoader:
        """Retrieve the SampledContextualEvaluationDataLoader for the dataset.

        Args:
            num_negatives (int): Number of negative samples per transaction.
            seed (int): Random seed.

        Returns:
            SampledContextualEvaluationDataLoader: The sampled contextual loader.
        """
        key = f"sampled_contextual_{num_negatives}_{seed}"

        if key not in self._precomputed_dataloader:
            train_sparse = self.train_set.get_sparse()
            eval_data = self.eval_set.get_df()

            # Retrieve labels to pre-compute eval data
            user_label = self.eval_set.user_label
            item_label = self.eval_set.item_label
            context_labels = self.eval_set.context_labels

            # Map the evaluation dataset
            u_map_df = nw.from_dict({
                user_label: list(self._umap.keys()), 
                f"{user_label}_idx": list(self._umap.values())
            }, native_namespace=nw.get_native_namespace(eval_data))
            
            i_map_df = nw.from_dict({
                item_label: list(self._imap.keys()), 
                f"{item_label}_idx": list(self._imap.values())
            }, native_namespace=nw.get_native_namespace(eval_data))

            eval_data = eval_data.join(u_map_df, on=user_label, how="inner") \
                                 .join(i_map_df, on=item_label, how="inner")

            eval_data = eval_data.with_columns(
                nw.col(f"{user_label}_idx").alias(user_label),
                nw.col(f"{item_label}_idx").alias(item_label)
            ).drop(f"{user_label}_idx", f"{item_label}_idx")

            self._precomputed_dataloader[key] = SampledContextualEvaluationDataLoader(
                train_interactions=train_sparse,
                eval_data=eval_data,
                user_id_label=user_label,
                item_id_label=item_label,
                context_labels=context_labels,
                num_items=self._niid,
                num_negatives=num_negatives,
                seed=seed,
                batch_size=self.batch_size,
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

    def get_feature_dims(self) -> Dict[str, int]:
        """Returns the dimensions (vocab size) of each side information feature.

        Returns:
            Dict[str, int]: A dictionary containing the name of the
                feature and the vocab size.
        """
        return self._feature_dims

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

        # Optionally add feature dimensions if present
        if self._feature_dims:
            base_info["feature_dims"] = self._feature_dims

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
        self._umap = user_mapping
        self._imap = item_mapping

    def clear_cache(self):
        """Clear the cache of inner data structures."""
        self.train_set.clear_dataset_cache()
        self.train_session.clear_dataset_cache()
