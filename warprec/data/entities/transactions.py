from typing import Any, Dict, List, Optional, Tuple

import narwhals as nw
import numpy as np
import torch
from narwhals.dataframe import DataFrame
from scipy.sparse import coo_matrix, csr_matrix
from torch import Tensor
from torch.utils.data import DataLoader

from warprec.data.entities.interactions import seed_worker
from warprec.data.entities.train_structures import PointWiseDataset
from warprec.utils.enums import RatingType


class Transactions:
    # pylint: disable = too-many-instance-attributes  # this class is the state it holds
    """Row-oriented view of the interaction records of a split.

    ``Interactions`` represents a split as a sparse user x item matrix. A matrix
    holds one cell per pair, so it cannot represent the same pair occurring in
    several contexts, and collapsing the rows into it both loses those records
    and leaves any per-row information with nothing to attach to. This entity
    keeps every row exactly as it was read, so a (user, item, context) triple
    survives intact.

    Every array is extracted from a single joined frame in one pass. Users,
    items, ratings and contexts therefore share one ordering by construction and
    cannot desynchronise.

    Args:
        data (DataFrame[Any]): Transaction data in DataFrame format.
        original_dims (Tuple[int, int]): The number of users and items.
        user_mapping (dict): Mapping of user ID -> user idx.
        item_mapping (dict): Mapping of item ID -> item idx.
        context_labels (Optional[List[str]]): The labels of the contextual columns.
        field_types (Optional[Dict[str, str]]): The declared type of each context
            field. Stored for the models that consume them; every field is
            currently emitted as a categorical index.
        side_tensor (Optional[Tensor]): The item feature lookup, indexed by item.
        rating_type (RatingType): The type of rating to be used.
        rating_label (Optional[str]): The label of the rating column.
        timestamp_label (Optional[str]): The label of the timestamp column.
        batch_size (int): The default batch size for the dataloaders.
        negative_sampling (str): How negatives are drawn, 'uniform' or 'popularity'.
    """

    def __init__(
        self,
        data: DataFrame[Any],
        original_dims: Tuple[int, int],
        user_mapping: dict,
        item_mapping: dict,
        context_labels: Optional[List[str]] = None,
        field_types: Optional[Dict[str, str]] = None,
        side_tensor: Optional[Tensor] = None,
        rating_type: RatingType = RatingType.IMPLICIT,
        rating_label: Optional[str] = None,
        timestamp_label: Optional[str] = None,
        batch_size: int = 1024,
        negative_sampling: str = "uniform",
    ) -> None:
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        # Each argument is a distinct part of the data schema.
        self._df = data
        self._umap = user_mapping
        self._imap = item_mapping
        self._og_nuid, self._og_niid = original_dims
        self._side_tensor = side_tensor

        self.user_label = data.columns[0]
        self.item_label = data.columns[1]
        self.context_labels = context_labels if context_labels else []
        self.field_types = field_types if field_types else {}
        self.rating_type = rating_type
        self.rating_label = rating_label if rating_type == RatingType.EXPLICIT else None
        self.timestamp_label = timestamp_label
        self.batch_size = batch_size
        self.negative_sampling = negative_sampling

        namespace = nw.get_native_namespace(data)
        umap_df = nw.from_dict(
            {
                self.user_label: list(user_mapping.keys()),
                "__uidx__": list(user_mapping.values()),
            },
            native_namespace=namespace,
        )
        imap_df = nw.from_dict(
            {
                self.item_label: list(item_mapping.keys()),
                "__iidx__": list(item_mapping.values()),
            },
            native_namespace=namespace,
        )

        # One frame in, every array out: this is what keeps the contexts
        # attached to the interaction they describe.
        mapped = (
            data.join(umap_df, on=self.user_label, how="inner")
            .join(imap_df, on=self.item_label, how="inner")
            .sort(["__uidx__", "__iidx__"])
        )

        self._users = mapped.select("__uidx__").to_numpy().flatten().astype(np.int64)
        self._items = mapped.select("__iidx__").to_numpy().flatten().astype(np.int64)

        if self.rating_label is not None:
            self._ratings = (
                mapped.select(self.rating_label).to_numpy().flatten().astype(np.float32)
            )
        else:
            self._ratings = np.ones(len(self._users), dtype=np.float32)

        self._contexts: Optional[np.ndarray] = None
        if self.context_labels:
            self._contexts = (
                mapped.select(self.context_labels).to_numpy().astype(np.int64)
            )

        self._seen_index: Optional[csr_matrix] = None
        self._context_index: Optional[
            Tuple[Dict[Tuple[int, int], np.ndarray], Dict[tuple, int]]
        ] = None

    def __len__(self) -> int:
        """The number of interaction records, duplicates included.

        Returns:
            int: The number of rows.
        """
        return len(self._users)

    def get_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return the row arrays, all sharing one ordering.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
                The users, items, ratings and contexts. The contexts are None
                when the dataset carries none.
        """
        return self._users, self._items, self._ratings, self._contexts

    def get_dims(self) -> Tuple[int, int]:
        """Return the dimensions of the index space.

        Returns:
            Tuple[int, int]: The number of users and the number of items.
        """
        return self._og_nuid, self._og_niid

    def _get_seen_index(self) -> csr_matrix:
        """Binary user x item membership index, used only by negative sampling.

        This is deliberately not the source of training rows: it exists so that
        the sampler can ask whether a user has interacted with an item in
        constant time. Building training examples from it is what collapses the
        records this entity exists to preserve.

        Returns:
            csr_matrix: The binary membership matrix.
        """
        if self._seen_index is None:
            index = coo_matrix(
                (
                    np.ones(len(self._users), dtype=np.float32),
                    (self._users, self._items),
                ),
                shape=(self._og_nuid, self._og_niid),
            ).tocsr()
            index.sort_indices()
            self._seen_index = index
        return self._seen_index

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
        """Create a DataLoader of (user, item, rating) rows with negative sampling.

        Args:
            neg_samples (int): Number of negative samples per positive row.
            include_side_info (bool): Whether to include item features.
            include_context (bool): Whether to include the contexts.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.
            seed (int): Seed for reproducibility.
            **kwargs (Any): Additional keyword arguments for the DataLoader.

        Returns:
            DataLoader: Yields (user, item, rating[, features][, context]).
        """
        side_info_tensor = None
        if include_side_info and self._side_tensor is not None:
            side_info_tensor = self._side_tensor

        context_tensor = None
        if include_context and self._contexts is not None:
            context_tensor = torch.from_numpy(self._contexts)

        dataset = PointWiseDataset(
            user_ids=torch.from_numpy(self._users),
            item_ids=torch.from_numpy(self._items),
            sparse_matrix=self._get_seen_index(),
            neg_samples=neg_samples,
            niid=self._og_niid,
            side_information=side_info_tensor,
            contexts=context_tensor,
            negative_sampling=self.negative_sampling,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=generator,
            **kwargs,
        )

    def get_context_index(
        self,
    ) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[tuple, int]]:
        """Index the items each user interacted with, per context.

        The evaluator uses it to ask whether an item was already seen *in this
        context*, rather than seen at all. A dataset without contexts yields a
        single context id, so the question degenerates to the plain one.

        Returns:
            Tuple[Dict[Tuple[int, int], np.ndarray], Dict[tuple, int]]:
                A mapping of (user, context id) to the items seen, and a mapping
                of context vector to context id.
        """
        if self._context_index is not None:
            return self._context_index

        if self._contexts is None:
            keys = np.zeros(len(self._users), dtype=np.int64)
            context_ids: Dict[tuple, int] = {(): 0}
        else:
            uniques, keys = np.unique(self._contexts, axis=0, return_inverse=True)
            keys = keys.astype(np.int64).ravel()
            context_ids = {tuple(row.tolist()): idx for idx, row in enumerate(uniques)}

        grouped: Dict[Tuple[int, int], List[int]] = {}
        for user, context_id, item in zip(self._users, keys, self._items):
            grouped.setdefault((int(user), int(context_id)), []).append(int(item))

        index = {
            key: np.unique(np.asarray(values, dtype=np.int64))
            for key, values in grouped.items()
        }

        self._context_index = (index, context_ids)
        return self._context_index
