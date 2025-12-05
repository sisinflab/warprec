from typing import Tuple, List, Set, Any

import torch
import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix


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


class ContextualEvaluationDataset(TorchDataset):
    """Dataset optimized for Full Contextual Evaluation.

    It converts DataFrame columns directly into Tensors for fast access.
    Each item is a single transaction: (user, item, context).

    Args:
        eval_data (DataFrame): The evaluation DataFrame.
        user_id_label (str): The user ID label in the DataFrame.
        item_id_label (str): The item ID label in the DataFrame.
        context_labels (List[str]): The list of labels of context
            information in the DataFrame.
    """

    def __init__(
        self,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
    ):
        # Convert DataFrame columns to Tensors immediately.
        self.user_indices = torch.from_numpy(
            eval_data[user_id_label].values.astype(np.int64)
        )
        self.item_indices = torch.from_numpy(
            eval_data[item_id_label].values.astype(np.int64)
        )

        # Context is a matrix [total_interactions, context_features]
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


class SampledContextualEvaluationDataset(TorchDataset):
    """Dataset optimized for Sampled Contextual Evaluation.

    For each transaction in the test set, it provides:
    - User ID
    - Positive Item ID (Target)
    - N Negative Item IDs
    - Context Vector

    Negatives are pre-computed in __init__ for maximum evaluation speed.
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
        self.num_negatives = num_negatives
        self.num_items = num_items

        # Store Context and User/Item info as Tensors
        self.user_indices = torch.from_numpy(
            eval_data[user_id_label].values.astype(np.int64)
        )
        self.pos_item_indices = torch.from_numpy(
            eval_data[item_id_label].values.astype(np.int64)
        )
        self.context_features = torch.from_numpy(
            eval_data[context_labels].values.astype(np.int64)
        )

        # Group eval items by user for fast lookup
        eval_items_per_user = (
            eval_data.groupby(user_id_label)[item_id_label].apply(set).to_dict()
        )

        np.random.seed(seed)
        self.negatives_list = []

        # Iterate over each transaction to generate specific negatives
        for _, user_idx in enumerate(self.user_indices.numpy()):
            u = int(user_idx)

            # Get train interactions
            train_items = train_interactions.indices[
                train_interactions.indptr[u] : train_interactions.indptr[u + 1]
            ]

            # Get test interactions
            test_items = eval_items_per_user.get(u, set())

            # Combine exclusions
            seen_items = set(train_items).union(test_items)

            # Sample Negatives
            negatives: list[int] = []
            while len(negatives) < num_negatives:
                cand = np.random.randint(0, num_items)
                if cand not in seen_items and cand not in negatives:
                    negatives.append(cand)

            self.negatives_list.append(torch.tensor(negatives, dtype=torch.long))

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


class ContextualEvaluationDataLoader(DataLoader):
    """DataLoader for Full Contextual Evaluation."""

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


class SampledContextualEvaluationDataLoader(DataLoader):
    """DataLoader for Sampled Contextual Evaluation."""

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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Custom collate to stack negatives correctly."""
        user_indices, pos_items, neg_items, context_features = zip(*batch)

        tensor_user_indices = torch.stack(user_indices)
        tensor_pos_items = torch.stack(pos_items)
        tensor_neg_items = torch.stack(neg_items)
        tensor_context_features = torch.stack(context_features)

        # Concatenate Positive and Negatives into a single Candidates tensor
        candidates = torch.cat([tensor_pos_items.unsqueeze(1), tensor_neg_items], dim=1)

        return tensor_user_indices, candidates, tensor_context_features
