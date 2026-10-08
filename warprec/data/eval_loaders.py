from typing import Tuple, List, Any, Optional

import torch
import numpy as np

from narwhals.dataframe import DataFrame

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix

from warprec.data.entities.train_structures.interaction_structures import (
    popularity_cumulative,
)


def draw_candidates(
    rng: np.random.RandomState,
    cumulative: Optional[np.ndarray],
    num_items: int,
    size: int,
) -> np.ndarray:
    """Draw item candidates from the configured negative-sampling distribution.

    Args:
        rng (np.random.RandomState): The stream the draws come from.
        cumulative (Optional[np.ndarray]): The cumulative popularity weights, or
            None to draw every item with equal probability.
        num_items (int): The size of the catalogue.
        size (int): How many candidates to draw.

    Returns:
        np.ndarray: The drawn item indices, which may repeat and may be items the
            user has already seen; the caller rejects those.
    """
    if cumulative is None:
        return rng.randint(0, num_items, size=size)
    drawn = np.searchsorted(cumulative, rng.random_sample(size))
    return np.minimum(drawn, num_items - 1)


def drawable_items(cumulative: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Which items a popularity draw can return at all.

    Args:
        cumulative (Optional[np.ndarray]): The cumulative popularity weights, or
            None for a uniform draw.

    Returns:
        Optional[np.ndarray]: A mask of the items with a weight, or None when
            every item can be drawn.
    """
    if cumulative is None:
        return None
    return np.diff(cumulative, prepend=0.0) > 0


def draw_negatives(
    rng: np.random.RandomState,
    cumulative: Optional[np.ndarray],
    drawable: Optional[np.ndarray],
    num_items: int,
    seen: np.ndarray,
    num_negatives: int,
    user: int,
) -> np.ndarray:
    """Draw distinct items a user has not seen, in the configured distribution.

    Candidates are drawn one after another and a repeat or a seen item is
    rejected, so the result is a draw without replacement from the user's unseen
    items: uniform over them, or in proportion to their popularity. The order of
    the draws is kept. Sorting the candidates and keeping the first ones, as
    the loaders used to, favoured the items with the lowest indices.

    Args:
        rng (np.random.RandomState): The stream the draws come from.
        cumulative (Optional[np.ndarray]): The cumulative popularity weights, or
            None to draw every item with equal probability.
        drawable (Optional[np.ndarray]): The items the popularity draw can
            return, from drawable_items; None for a uniform draw.
        num_items (int): The size of the catalogue.
        seen (np.ndarray): The items the user may not be offered.
        num_negatives (int): How many items to draw.
        user (int): The user drawn for, named in the error.

    Returns:
        np.ndarray: Exactly num_negatives distinct unseen items, in draw order.

    Raises:
        ValueError: If the user has fewer items left to draw than num_negatives.
    """
    seen = np.unique(np.asarray(seen, dtype=np.int64))
    if drawable is None:
        available = num_items - len(seen)
        reason = ""
    else:
        # An item no one interacted with has no weight and is never drawn.
        available = int(drawable.sum() - drawable[seen].sum())
        reason = " that popularity sampling can draw"
    # Every user must yield the same number of candidates, or the evaluator
    # cannot line the batch up, and drawing for a user who cannot supply them
    # would never return.
    if available < num_negatives:
        raise ValueError(
            f"Sampled evaluation asks for {num_negatives} negatives per user, but "
            f"user {user} has only {available} of the {num_items} items left "
            f"unseen{reason}. Lower 'evaluation.num_negatives' or evaluate on the "
            "full catalogue."
        )

    chosen = np.empty(0, dtype=np.int64)
    while len(chosen) < num_negatives:
        # Twice as many as needed, so that one round is usually enough.
        candidates = draw_candidates(rng, cumulative, num_items, 2 * num_negatives)
        candidates = candidates[np.isin(candidates, seen, invert=True)]
        merged = np.concatenate([chosen, candidates.astype(np.int64)])
        # Each item where it was first drawn, in the order of the draws.
        _, first = np.unique(merged, return_index=True)
        chosen = merged[np.sort(first)]
    return chosen[:num_negatives]


class EvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, item_indices, values)

    The ground truth stays sparse until it reaches the device, where the
    Evaluator densifies it once. Expanding a row here would cost the full item
    catalogue for a handful of values, and send it through the DataLoader.
    """

    def __init__(
        self,
        eval_interactions: csr_matrix,
    ):
        self.eval_interactions = eval_interactions
        self.users_with_eval = [
            u
            for u in range(eval_interactions.shape[0])
            if eval_interactions.indptr[u + 1] - eval_interactions.indptr[u] > 0
        ]

    def __len__(self) -> int:
        return len(self.users_with_eval)

    def __getitem__(self, idx: int) -> Tuple[int, Tensor, Tensor]:
        user_idx = self.users_with_eval[idx]
        start = self.eval_interactions.indptr[user_idx]
        end = self.eval_interactions.indptr[user_idx + 1]
        cols = torch.from_numpy(
            self.eval_interactions.indices[start:end].astype(np.int64)
        )
        vals = torch.from_numpy(
            self.eval_interactions.data[start:end].astype(np.float32)
        )

        return user_idx, cols, vals


class ContextualEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, target_item_idx, context_vector)
    """

    def __init__(
        self,
        eval_data: DataFrame[Any],
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
    ):
        # Pre-convert DataFrames to torch tensor to reduce overhead
        self.user_indices = torch.from_numpy(
            eval_data.select(user_id_label).to_numpy().flatten().astype(np.int64)
        )
        self.item_indices = torch.from_numpy(
            eval_data.select(item_id_label).to_numpy().flatten().astype(np.int64)
        )
        # Categorical fields store an index and numeric fields a value, so the
        # evaluation contexts are read the same way the training ones are.
        self.context_features = torch.from_numpy(
            eval_data.select(context_labels).to_numpy().astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.user_indices[idx],
            self.item_indices[idx],
            self.context_features[idx],
        )


class SampledEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, pos_item, neg_items_vector)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_interactions: csr_matrix,
        num_negatives: int = 99,
        seed: int = 42,
        negative_sampling: str = "uniform",
        neg_alpha: float = 0.75,
    ):
        super().__init__()
        self.num_users, self.num_items = train_interactions.shape

        # Pre-calculate all positives (Train + Eval) per user,
        # we use a list of arrays for fast access
        self.all_positives = []
        for u in range(self.num_users):
            train_indices = train_interactions.indices[
                train_interactions.indptr[u] : train_interactions.indptr[u + 1]
            ]
            eval_indices = eval_interactions.indices[
                eval_interactions.indptr[u] : eval_interactions.indptr[u + 1]
            ]
            self.all_positives.append(np.union1d(train_indices, eval_indices))

        # Identify users who actually have evaluation data
        self.users_with_eval = [
            u
            for u in range(self.num_users)
            if eval_interactions.indptr[u + 1] - eval_interactions.indptr[u] > 0
        ]

        self.positive_items_list = []
        self.negative_items_list = []

        # The sampler owns its stream: seeding the global one made the draws
        # depend on whatever else had touched it. RandomState is the same
        # algorithm, so the uniform draws are unchanged.
        rng = np.random.RandomState(seed)
        cumulative = (
            popularity_cumulative(train_interactions, self.num_items, neg_alpha)
            if negative_sampling == "popularity"
            else None
        )
        drawable = drawable_items(cumulative)

        for u in self.users_with_eval:
            # Store positives
            eval_pos = eval_interactions.indices[
                eval_interactions.indptr[u] : eval_interactions.indptr[u + 1]
            ]
            self.positive_items_list.append(torch.tensor(eval_pos, dtype=torch.long))

            # Compute seen items
            seen_items = self.all_positives[u]
            n_seen = len(seen_items)

            # If user has seen almost everything, return empty or partial
            if self.num_items - n_seen <= 0:
                self.negative_items_list.append(torch.tensor([], dtype=torch.long))
                continue

            self.negative_items_list.append(
                torch.from_numpy(
                    draw_negatives(
                        rng,
                        cumulative,
                        drawable,
                        self.num_items,
                        seen_items,
                        num_negatives,
                        u,
                    )
                )
            )

    def __len__(self) -> int:
        return len(self.users_with_eval)

    def __getitem__(self, idx: int) -> Tuple[int, Tensor, Tensor]:
        user_idx = self.users_with_eval[idx]
        return (
            user_idx,
            self.positive_items_list[idx],
            self.negative_items_list[idx],
        )

    def collate_fn(
        self, batch: List[Tuple[int, Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Collate the sampled evaluation rows into padded batches.

        Args:
            batch (List[Tuple[int, Tensor, Tensor]]): The rows to collate.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The users, positives and negatives.
        """
        user_indices, positive_tensors, negative_tensors = zip(*batch)
        user_indices_tensor = torch.tensor(list(user_indices), dtype=torch.long)

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
        return user_indices_tensor, positives_padded, negatives_padded


class SampledContextualEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, pos_item, neg_items_vector, context_vector)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_data: DataFrame[Any],
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
        num_items: int,
        num_negatives: int = 99,
        seed: int = 42,
        negative_sampling: str = "uniform",
        neg_alpha: float = 0.75,
    ):
        self.num_negatives = num_negatives
        self.num_items = num_items

        # Pre-convert DataFrames to torch tensor to reduce overhead
        self.user_indices = torch.from_numpy(
            eval_data.select(user_id_label).to_numpy().flatten().astype(np.int64)
        )
        self.pos_item_indices = torch.from_numpy(
            eval_data.select(item_id_label).to_numpy().flatten().astype(np.int64)
        )
        # Categorical fields store an index and numeric fields a value, so the
        # evaluation contexts are read the same way the training ones are.
        self.context_features = torch.from_numpy(
            eval_data.select(context_labels).to_numpy().astype(np.float32)
        )

        n_train_users = train_interactions.shape[0]

        self.negatives_list: list[Tensor] = []
        rng = np.random.RandomState(seed)
        cumulative = (
            popularity_cumulative(train_interactions, self.num_items, neg_alpha)
            if negative_sampling == "popularity"
            else None
        )
        drawable = drawable_items(cumulative)

        for idx, user_idx_tensor in enumerate(self.user_indices):
            u = int(user_idx_tensor.item())
            target_item = int(self.pos_item_indices[idx].item())

            # Retrieve training history
            if u < n_train_users:
                train_items = train_interactions.indices[
                    train_interactions.indptr[u] : train_interactions.indptr[u + 1]
                ]
            else:
                train_items = np.array([], dtype=np.int64)

            # Compute seen items
            seen_items = np.append(train_items, target_item)

            self.negatives_list.append(
                torch.from_numpy(
                    draw_negatives(
                        rng,
                        cumulative,
                        drawable,
                        self.num_items,
                        seen_items,
                        self.num_negatives,
                        u,
                    )
                )
            )

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.user_indices[idx],
            self.pos_item_indices[idx],
            self.negatives_list[idx],
            self.context_features[idx],
        )

    def collate_fn(
        self, batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Collate the sampled contextual evaluation rows into batches.

        Args:
            batch (List[Tuple[Tensor, Tensor, Tensor, Tensor]]): The rows to collate.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The users, positives, negatives
                and contexts.
        """
        user_indices, pos_items, neg_items, context_features = zip(*batch)

        tensor_user_indices = torch.stack(user_indices)
        tensor_pos_items = torch.stack(pos_items).unsqueeze(1)
        tensor_neg_items = torch.stack(neg_items)
        tensor_context_features = torch.stack(context_features)

        return (
            tensor_user_indices,
            tensor_pos_items,
            tensor_neg_items,
            tensor_context_features,
        )


def sparse_eval_collate(
    samples: List[Tuple[int, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate the sparse rows of EvaluationDataset into flat COO arrays.

    Four tensors rather than three: three would collide with the contextual
    batch shape that `Evaluator._parse_batch` dispatches on.

    Args:
        samples (List[Tuple[int, Tensor, Tensor]]): The per-user rows.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: The user indices, the row index
            of each value within the batch, the item indices, and the values.
    """
    users = torch.tensor([s[0] for s in samples], dtype=torch.long)
    counts = torch.tensor([s[1].numel() for s in samples], dtype=torch.long)
    rows = torch.repeat_interleave(torch.arange(len(samples), dtype=torch.long), counts)
    cols = torch.cat([s[1] for s in samples])
    vals = torch.cat([s[2] for s in samples])

    return users, rows, cols, vals
