from typing import Tuple, List

import torch
import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix


class EvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, dense_ground_truth)
    """

    def __init__(
        self,
        eval_interactions: csr_matrix,
    ):
        self.num_users = eval_interactions.shape[0]
        self.eval_interactions = eval_interactions

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, idx: int) -> Tuple[int, Tensor]:
        eval_row = self.eval_interactions.getrow(idx)
        ground_truth = torch.from_numpy(eval_row.toarray()).to(torch.float32).squeeze(0)

        return idx, ground_truth


class ContextualEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, target_item_idx, context_vector)
    """

    def __init__(
        self,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
    ):
        # Pre-convert DataFrames to torch tensor to reduce overhead
        self.user_indices = torch.from_numpy(
            eval_data[user_id_label].values.astype(np.int64)
        )
        self.item_indices = torch.from_numpy(
            eval_data[item_id_label].values.astype(np.int64)
        )
        self.context_features = torch.from_numpy(
            eval_data[context_labels].values.astype(np.int64)
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
        # (We only want to iterate over these)
        self.users_with_eval = [
            u
            for u in range(self.num_users)
            if eval_interactions.indptr[u + 1] - eval_interactions.indptr[u] > 0
        ]

        self.positive_items_list = []
        self.negative_items_list = []

        np.random.seed(seed)

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

            # Sample one time 2x the number of negatives
            # NOTE: In most cases this will skip the while loop
            num_to_generate = num_negatives * 2
            candidates = np.random.randint(0, self.num_items, size=num_to_generate)

            # Fast filtering using numpy boolean masking
            # Note: For very large item sets, np.isin can be slow.
            # If num_items > 1M, consider Bloom Filters or just the 'while' loop.
            mask = np.isin(candidates, seen_items, invert=True)
            valid_negatives = candidates[mask]

            # Remove duplicates in candidates if necessary
            valid_negatives = np.unique(valid_negatives)

            # If we don't have enough, fallback to a loop
            if len(valid_negatives) < num_negatives:
                final_negs = list(valid_negatives)
                while len(final_negs) < num_negatives:
                    cand = np.random.randint(0, self.num_items)
                    if cand not in seen_items and cand not in final_negs:
                        final_negs.append(cand)
                valid_negatives = np.array(final_negs)

            # Take exactly num_negatives
            self.negative_items_list.append(
                torch.tensor(valid_negatives[:num_negatives], dtype=torch.long)
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


class SampledContextualEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, pos_item, neg_items_vector, context_vector)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
        num_items: int,
        num_negatives: int = 99,
        seed: int = 42,
    ):
        # pylint: disable = too-many-nested-blocks
        self.num_negatives = num_negatives
        self.num_items = num_items

        # Pre-convert DataFrames to torch tensor to reduce overhead
        self.user_indices = torch.from_numpy(
            eval_data[user_id_label].values.astype(np.int64)
        )
        self.pos_item_indices = torch.from_numpy(
            eval_data[item_id_label].values.astype(np.int64)
        )
        self.context_features = torch.from_numpy(
            eval_data[context_labels].values.astype(np.int64)
        )

        n_train_users = train_interactions.shape[0]

        self.negatives_list: list[Tensor] = []
        np.random.seed(seed)

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

            # Generate 2x candidates to avoid loops in most cases
            num_to_generate = self.num_negatives * 2
            candidates = np.random.randint(0, self.num_items, size=num_to_generate)

            # Filter out seen items using optimized numpy boolean masking
            mask = np.isin(candidates, seen_items, invert=True)
            valid_negatives = candidates[mask]

            # Remove duplicates
            valid_negatives = np.unique(valid_negatives)

            # If we don't have enough, fallback to a loop
            if len(valid_negatives) < self.num_negatives:
                final_negs = list(valid_negatives)
                while len(final_negs) < self.num_negatives:
                    cand = np.random.randint(0, self.num_items)
                    if cand not in final_negs:
                        if cand != target_item:
                            if cand not in train_items:
                                final_negs.append(cand)  # type: ignore[arg-type]
                valid_negatives = np.array(final_negs)

            # Take exactly num_negatives
            self.negatives_list.append(
                torch.tensor(valid_negatives[: self.num_negatives], dtype=torch.long)
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


class EvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, ground_truths)
    """

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


class ContextualEvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, target_items, contexts)
    """

    def __init__(
        self,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
        batch_size: int = 1024,
        **kwargs,
    ):
        dataset = ContextualEvaluationDataset(
            eval_data=eval_data,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            context_labels=context_labels,
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=False, **kwargs)


class SampledEvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, pos_items, neg_items)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_interactions: csr_matrix,
        num_negatives: int = 99,
        seed: int = 42,
        batch_size: int = 1024,
        **kwargs,
    ):
        self.num_items = train_interactions.shape[1]

        dataset = SampledEvaluationDataset(
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
        batch: List[Tuple[int, Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        user_indices, positive_tensors, negative_tensors = zip(*batch)

        user_indices_tensor = torch.tensor(list(user_indices), dtype=torch.long)

        # Stack sequences and pad them ensure same length
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


class SampledContextualEvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, pos_items, neg_items, contexts)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
        num_items: int,
        num_negatives: int = 99,
        seed: int = 42,
        batch_size: int = 1024,
        **kwargs,
    ):
        dataset = SampledContextualEvaluationDataset(
            train_interactions=train_interactions,
            eval_data=eval_data,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            context_labels=context_labels,
            num_items=num_items,
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
        batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
