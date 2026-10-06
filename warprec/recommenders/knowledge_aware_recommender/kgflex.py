# pylint: disable = R0801, E1102
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, hstack
from torch import Tensor, nn

from warprec.data.entities import Interactions, KnowledgeGraph, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.knowledge_aware_recommender.knowledge_utils import (
    KnowledgeRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry

Selection = Tuple[np.ndarray, np.ndarray, np.ndarray]


@model_registry.register(name="KGFlex")
class KGFlex(KnowledgeRecommenderUtils, IterativeRecommender):
    """Implementation of KGFlex algorithm from
        Sparse Feature Factorization for Recommender Systems with Knowledge
        Graphs (RecSys 2021)

    A user is not described by one vector but by the few features their
    choices depend on. Those are found before training: a feature matters to a
    user as much as it tells the items they took from as many they did not,
    measured by information gain. Only the features that tell them apart are
    kept, and the gain becomes the weight the feature carries for that user.

    Every kept feature has a global embedding and bias shared by all users,
    and a personal embedding for each user who kept it. An item is scored by
    summing, over the features it shares with the user, the gain times the
    agreement of the two embeddings plus the bias, so a user only ever trains
    the features they are expert about.

    The original implementation is KGFlex in Elliot. Three of its behaviours
    are not reproduced: its prediction paired a user's embeddings with the
    wrong features, so evaluation scored a different formula than training; it
    skipped a sample whose negative shared no feature with the user; and it
    raised on a first-order limit without a second-order one.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        knowledge (Optional[KnowledgeGraph]): The facts about the items.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the feature embeddings.
        first_order_limit (int): How many first-order features each user keeps.
        second_order_limit (int): How many second-order features each user keeps.
        min_feature_items (int): How many items must carry a feature for it to
            be considered.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        user_offsets (Tensor): Where each user's features start among the pairs.
        pair_feature (Tensor): The feature of each (user, feature) pair.
        pair_weight (Tensor): The information gain of each pair.
        item_rows (Tensor): The item of each (item, feature) fact.
        item_columns (Tensor): The feature of each (item, feature) fact.
        item_keys (Tensor): The (item, feature) facts encoded and sorted.

    Raises:
        ValueError: If the interactions were not provided, or no feature
            informs any user.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    first_order_limit: int
    second_order_limit: int
    min_feature_items: int
    batch_size: int
    epochs: int
    learning_rate: float

    # Registered buffers, annotated so that they read as the tensors they are.
    user_offsets: Tensor
    pair_feature: Tensor
    pair_weight: Tensor
    item_rows: Tensor
    item_columns: Tensor
    item_keys: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        knowledge: Optional[KnowledgeGraph] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        # pylint: disable = too-many-arguments, too-many-positional-arguments, too-many-locals
        super().__init__(params, info, *args, knowledge=knowledge, seed=seed, **kwargs)

        if interactions is None:
            raise ValueError(
                "KGFlex chooses each user's features from the items they "
                "interacted with, so it needs the interactions at construction."
            )

        first, first_labels = self._features(1, self.first_order_limit)
        second, second_labels = self._features(2, self.second_order_limit)
        first_items, second_items = self._to_scipy(first), self._to_scipy(second)

        history = interactions.get_sparse().tocsr().astype(np.float32)
        history.data[:] = 1.0

        generator = torch.Generator()
        generator.manual_seed(seed)
        negatives = self._draw_negatives(history, generator)

        first_kept = self._select(
            history, negatives, first_items, self.first_order_limit
        )
        second_kept = self._select(
            history, negatives, second_items, self.second_order_limit
        )
        users = np.concatenate([first_kept[0], second_kept[0]])
        columns = np.concatenate([first_kept[1], second_kept[1] + first.size(1)])
        gains = np.concatenate([first_kept[2], second_kept[2]])

        if users.size == 0:
            raise ValueError(
                "No feature of the knowledge graph tells any user's items apart "
                "from the rest, so KGFlex has nothing to learn. Lower "
                "min_feature_items or raise the feature limits."
            )

        # Only the features some user kept are given an embedding.
        used, columns = np.unique(columns, return_inverse=True)
        labels = first_labels + second_labels
        self.feature_labels: List[Tuple[Any, ...]] = [labels[c] for c in used.tolist()]
        self.n_features = int(used.size)

        order = np.lexsort((columns, users))
        users, columns, gains = users[order], columns[order], gains[order]
        offsets = np.zeros(self.n_users + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(np.bincount(users, minlength=self.n_users))

        facts = hstack([first_items, second_items]).tocsc()[:, used].tocoo()
        rows = torch.as_tensor(facts.row, dtype=torch.long)
        cols = torch.as_tensor(facts.col, dtype=torch.long)

        self.register_buffer("user_offsets", torch.from_numpy(offsets))
        self.register_buffer("pair_feature", torch.as_tensor(columns, dtype=torch.long))
        self.register_buffer("pair_weight", torch.as_tensor(gains, dtype=torch.float))
        self.register_buffer("item_rows", rows)
        self.register_buffer("item_columns", cols)
        self.register_buffer(
            "item_keys", torch.sort(rows * self.n_features + cols).values
        )

        # The personal embedding of each (user, feature) pair, and the global
        # embedding and bias of each feature, drawn as Elliot draws them.
        self.user_feature_embedding = nn.Embedding(users.size, self.embedding_size)
        self.feature_embedding = nn.Embedding(self.n_features, self.embedding_size)
        self.feature_bias = nn.Embedding(self.n_features, 1)
        for embedding in (
            self.user_feature_embedding,
            self.feature_embedding,
            self.feature_bias,
        ):
            nn.init.normal_(embedding.weight, std=0.1)

        self.rec_loss = BPRLoss()

        silent = int((offsets[1:] == offsets[:-1]).sum())
        logger.stat_msg(
            f"Features: {self.n_features}      User-feature pairs: {users.size}      "
            f"Users without an informative feature: {silent}/{self.n_users}",
            "KGFlex",
        )

    def _features(self, order: int, limit: int) -> Tuple[Tensor, List[Tuple[Any, ...]]]:
        """The item features of one order, or none when no user may keep any.

        Walking two facts from every item is the expensive part of a large
        graph, so it is not done for features that would all be discarded.

        Args:
            order (int): 1 for first-order features, 2 for second-order ones.
            limit (int): How many features of this order each user keeps.

        Returns:
            Tuple[Tensor, List[Tuple[Any, ...]]]: The sparse {item x feature}
                matrix and the feature labels.
        """
        if limit == 0:
            empty = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0),
                (self.n_items, 0),
            )
            return empty.coalesce(), []

        return self._graph.item_features(order=order, min_items=self.min_feature_items)

    def get_extra_state(self) -> Dict[str, Any]:
        """What a checkpoint carries besides tensors: the name of each feature.

        Returns:
            Dict[str, Any]: The feature labels, in embedding order.
        """
        return {"feature_labels": self.feature_labels}

    def set_extra_state(self, state: Any):
        """Restore the feature names a checkpoint was trained with.

        Args:
            state (Any): What get_extra_state returned.
        """
        self.feature_labels = list(state["feature_labels"])

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        """Take the features a checkpoint was trained on, not the ones drawn here.

        The selection depends on the seed the model was built with, and the
        pipelines rebuild a model from its checkpoint without it. The shapes
        are therefore taken from the checkpoint before its values are copied
        in, so the weights land on the pairs they were trained for.

        Args:
            state_dict (Dict[str, Any]): The state being loaded.
            prefix (str): The prefix of this module's keys.
            local_metadata (Dict[str, Any]): The metadata of this module.
            strict (bool): Whether every key must match.
            missing_keys (List[str]): Collects the keys the state lacks.
            unexpected_keys (List[str]): Collects the keys the model lacks.
            error_msgs (List[str]): Collects the loading errors.
        """
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        for name in (
            "user_offsets",
            "pair_feature",
            "pair_weight",
            "item_rows",
            "item_columns",
            "item_keys",
        ):
            saved = state_dict.get(prefix + name)
            if saved is not None:
                current = getattr(self, name)
                setattr(self, name, torch.empty_like(saved, device=current.device))

        for name in ("user_feature_embedding", "feature_embedding", "feature_bias"):
            saved = state_dict.get(f"{prefix}{name}.weight")
            current = getattr(self, name).weight
            if saved is not None and saved.shape != current.shape:
                rows, width = saved.shape
                setattr(self, name, nn.Embedding(rows, width, device=current.device))

        saved = state_dict.get(prefix + "feature_embedding.weight")
        if saved is not None:
            self.n_features = int(saved.size(0))

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _to_scipy(features: Tensor) -> csr_matrix:
        """The same binary matrix, in the form scipy multiplies.

        Args:
            features (Tensor): A coalesced sparse {item x feature} tensor.

        Returns:
            csr_matrix: The matrix as a scipy CSR.
        """
        rows, columns = features.indices().numpy()
        return coo_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, columns)),
            shape=tuple(features.shape),
        ).tocsr()

    def _draw_negatives(
        self, history: csr_matrix, generator: torch.Generator
    ) -> csr_matrix:
        """As many items a user did not take as they took, with replacement.

        A user who took every item has nothing to be contrasted with, so none
        is drawn for them and they end up without features.

        Args:
            history (csr_matrix): The binary {user x item} training matrix.
            generator (torch.Generator): The stream to draw from.

        Returns:
            csr_matrix: How often each item was drawn for each user.
        """
        taken = np.diff(history.indptr)
        owners = np.repeat(
            np.arange(self.n_users), np.where(taken < self.n_items, taken, 0)
        )
        seen = np.sort(
            np.repeat(np.arange(self.n_users), taken).astype(np.int64) * self.n_items
            + history.indices
        )

        def is_seen(users: np.ndarray, items: np.ndarray) -> np.ndarray:
            keys = users.astype(np.int64) * self.n_items + items
            position = np.searchsorted(seen, keys).clip(max=seen.size - 1)
            return seen[position] == keys

        items = torch.randint(self.n_items, (owners.size,), generator=generator).numpy()
        pending = np.flatnonzero(is_seen(owners, items))
        while pending.size:
            items[pending] = torch.randint(
                self.n_items, (pending.size,), generator=generator
            ).numpy()
            pending = pending[is_seen(owners[pending], items[pending])]

        return coo_matrix(
            (np.ones(owners.size, dtype=np.float32), (owners, items)),
            shape=history.shape,
        ).tocsr()

    @staticmethod
    def _entropy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """The binary entropy of two counts, in bits, with 0 log 0 = 0.

        Args:
            first (np.ndarray): The first count.
            second (np.ndarray): The second count.

        Returns:
            np.ndarray: The entropy of each pair of counts.
        """
        total = first + second
        safe = np.where(total > 0, total, 1.0)

        entropy = np.zeros_like(total, dtype=np.float64)
        for part in (first, second):
            share = part / safe
            entropy -= np.where(
                share > 0, share * np.log2(np.where(share > 0, share, 1.0)), 0.0
            )
        return entropy

    @staticmethod
    def information_gain(
        positive: np.ndarray, negative: np.ndarray, taken: np.ndarray
    ) -> np.ndarray:
        """How much knowing a feature tells a user's items from the rest.

        The user's n items are set against n they did not take, so the prior
        uncertainty is one bit. A feature carried by ``positive`` of the first
        and ``negative`` of the second splits the 2n items into those carrying
        it and those not, and the gain is the bit minus the entropy left in the
        two groups, each weighted by its size.

        Args:
            positive (np.ndarray): How many of the user's items carry the feature.
            negative (np.ndarray): How many of the drawn negatives carry it.
            taken (np.ndarray): How many items the user took, n.

        Returns:
            np.ndarray: The information gain, in bits.
        """
        carried = positive + negative
        missing = 2 * taken - carried
        return (
            1.0
            - carried / (2 * taken) * KGFlex._entropy(positive, negative)
            - missing
            / (2 * taken)
            * KGFlex._entropy(taken - positive, taken - negative)
        )

    def _select(
        self,
        history: csr_matrix,
        negatives: csr_matrix,
        features: csr_matrix,
        limit: int,
    ) -> Selection:
        """The features of one order that tell each user's items apart.

        Args:
            history (csr_matrix): The binary {user x item} training matrix.
            negatives (csr_matrix): The {user x item} drawn negatives.
            features (csr_matrix): The binary {item x feature} matrix.
            limit (int): How many features each user keeps; -1 keeps all.

        Returns:
            Selection: The users, features and gains kept, gain above zero.
        """
        empty: Selection = (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
        if limit == 0 or features.shape[1] == 0:
            return empty

        positive = (history @ features).tocoo()
        taken = np.diff(history.indptr).astype(np.float64)

        # A user who took every item had no negatives to be contrasted with.
        eligible = (taken < self.n_items)[positive.row]
        users, columns = positive.row[eligible], positive.col[eligible]
        if users.size == 0:
            return empty

        counts = positive.data[eligible].astype(np.float64)
        drawn = np.asarray((negatives @ features).tocsr()[users, columns]).ravel()
        gains = self.information_gain(counts, drawn, taken[users])

        informative = gains > 0
        users, columns, gains = (
            users[informative],
            columns[informative],
            gains[informative],
        )

        if limit > 0:
            order = np.lexsort((columns, -gains, users))
            users, columns, gains = users[order], columns[order], gains[order]
            rank = np.arange(users.size) - np.searchsorted(users, users)
            keep = rank < limit
            users, columns, gains = users[keep], columns[keep], gains[keep]

        return (
            users.astype(np.int64),
            columns.astype(np.int64),
            gains.astype(np.float32),
        )

    def _user_pairs(self, user: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """What each feature of the given users adds to a score it is shared in.

        ``k_uf (p_uf . g_f + b_f)`` depends on the user and the feature but not
        on the item, which is what lets a catalogue be scored with one product.

        Args:
            user (Tensor): The user indices.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The position of each pair's user in
                the batch, its feature and its contribution.
        """
        start = self.user_offsets[user]
        degree = self.user_offsets[user + 1] - start

        rows = torch.repeat_interleave(
            torch.arange(user.numel(), device=user.device), degree
        )
        run_start = torch.cumsum(degree, dim=0) - degree
        pairs = (
            start[rows]
            + torch.arange(rows.numel(), device=user.device)
            - run_start[rows]
        )

        features = self.pair_feature[pairs]
        agreement = (
            self.user_feature_embedding(pairs) * self.feature_embedding(features)
        ).sum(dim=-1)
        contribution = self.pair_weight[pairs] * (
            agreement + self.feature_bias(features).squeeze(-1)
        )
        return rows, features, contribution

    def _shared(self, pairs: Tuple[Tensor, Tensor, Tensor], item: Tensor) -> Tensor:
        """Sum each user's contributions over the features the item carries.

        Args:
            pairs (Tuple[Tensor, Tensor, Tensor]): The output of _user_pairs.
            item (Tensor): One item per user of the batch.

        Returns:
            Tensor: One score per (user, item); 0 when nothing is shared.
        """
        rows, features, contribution = pairs
        keys = item[rows] * self.n_features + features
        position = torch.searchsorted(self.item_keys, keys).clamp(
            max=self.item_keys.numel() - 1
        )
        shared = (self.item_keys[position] == keys).to(contribution.dtype)

        return torch.zeros(
            item.numel(), device=contribution.device, dtype=contribution.dtype
        ).index_add_(0, rows, contribution * shared)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, positive, negative = batch

        pairs = self._user_pairs(user)

        # The paper adds no regularizer.
        loss = self.rec_loss(
            self._shared(pairs, positive), self._shared(pairs, negative)
        )

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Score each given user against the item beside it.

        Args:
            user (Tensor): The user indices.
            item (Tensor): The item indices.

        Returns:
            Tensor: One score per pair.
        """
        return self._shared(self._user_pairs(user), item)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from each user's features, over the items carrying them.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        rows, features, contribution = self._user_pairs(user_indices)

        weights = torch.zeros(
            user_indices.numel(),
            self.n_features,
            device=contribution.device,
            dtype=contribution.dtype,
        )
        weights.index_put_((rows, features), contribution, accumulate=True)

        catalogue = torch.sparse_coo_tensor(
            torch.stack([self.item_rows, self.item_columns]),
            torch.ones(self.item_rows.numel(), device=contribution.device),
            (self.n_items, self.n_features),
        )
        scores = torch.sparse.mm(catalogue, weights.t()).t()

        if item_indices is None:
            return scores

        # Sampled candidates are padded with the index one past the catalogue.
        return nn.functional.pad(scores, (0, 1)).gather(1, item_indices)
