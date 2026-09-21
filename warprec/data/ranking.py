from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

from warprec.data.entities.context import context_key


def mask_seen_pairs(predictions: Tensor, seen: csr_matrix) -> None:
    """Exclude from the ranking every item a user has already interacted with.

    Args:
        predictions (Tensor): The score matrix, modified in place.
        seen (csr_matrix): The rows of the training matrix for this batch of users.
    """
    predictions[seen.nonzero()] = -torch.inf


def mask_seen_in_context(
    predictions: Tensor,
    user_indices: Tensor,
    context_rows: np.ndarray,
    context_index: Dict[Tuple[int, int], np.ndarray],
    context_ids: Dict[tuple, int],
    target_items: Optional[Tensor] = None,
) -> int:
    """Exclude the items a user has seen *in this situation*, rather than at all.

    A context-aware run asks a different question of the training history: an item
    the user watched on a weekday morning is a fair recommendation for a Saturday
    night. A row whose context never appeared in training has nothing to exclude.

    Args:
        predictions (Tensor): The score matrix, modified in place.
        user_indices (Tensor): The users of this batch.
        context_rows (np.ndarray): The context of each row of the batch.
        context_index (Dict[Tuple[int, int], np.ndarray]): The items each user saw,
            per context id.
        context_ids (Dict[tuple, int]): The id of each known context vector.
        target_items (Optional[Tensor]): The ground-truth item of each row, when the
            evaluation has one. Used only to count how often the answer was a
            repetition the run has just masked away.

    Returns:
        int: How many rows had their ground-truth item masked as already seen.
    """
    repeated = 0
    for row, user in enumerate(user_indices.tolist()):
        key = context_ids.get(context_key(context_rows[row]), -1)
        if key < 0:
            continue

        seen = context_index.get((user, key))
        if seen is None:
            continue

        predictions[row, seen] = -torch.inf
        if target_items is not None:
            repeated += int(int(target_items[row]) in seen)

    return repeated


def resolve_mask_policy(policy: str, transactions: Any) -> str:
    """Decide which seen-item rule a run follows.

    Args:
        policy (str): The configured policy, one of 'auto', 'context', 'pair' or
            'none'.
        transactions (Any): The row-oriented training records, or None when the
            dataset has no contextual columns.

    Returns:
        str: The resolved policy, never 'auto'.
    """
    if policy != "auto":
        return policy
    return (
        "context"
        if transactions is not None and transactions.context_labels
        else "pair"
    )


def cold_item_candidates(train_set: csr_matrix, candidates: str) -> Optional[Tensor]:
    """Which items a run is allowed to rank, under a cold-start protocol.

    An item nobody interacted with during training is a cold one. Ranking over the
    whole catalogue mixes the two populations, and since the warm items are both
    far more numerous and far better served by a collaborative model, a cold-start
    result computed over everything mostly measures warm-item ranking instead.

    Args:
        train_set (csr_matrix): The training interaction matrix.
        candidates (str): Which population to keep, 'all', 'cold' or 'warm'.

    Returns:
        Optional[Tensor]: A boolean mask over the items, or None when every item
            is eligible and there is nothing to restrict.

    Raises:
        ValueError: If the candidate set is not one WarpRec knows.
    """
    if candidates == "all":
        return None

    if candidates not in ("cold", "warm"):
        raise ValueError(
            f"Candidate set '{candidates}' is not supported. "
            "Use 'all', 'cold' or 'warm'."
        )

    is_cold = torch.as_tensor(train_set.getnnz(axis=0) == 0)
    return is_cold if candidates == "cold" else ~is_cold


def restrict_to_candidates(predictions: Tensor, candidates: Tensor) -> None:
    """Exclude from the ranking every item outside the candidate set.

    Args:
        predictions (Tensor): The score matrix, modified in place.
        candidates (Tensor): A boolean mask over the items, True to keep.
    """
    predictions[:, ~candidates] = -torch.inf


def top_k_breaking_ties(
    predictions: Tensor, k: int, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor]:
    """Take the top k scores, deciding ties by chance rather than by item id.

    ``torch.topk`` resolves equal scores by position, which is not a neutral rule:
    a model that scores a whole population alike then always returns its lowest
    item ids, and in most catalogues those are the oldest and best known entries.
    Under a cold-start protocol that is the difference between a model that has
    learned nothing scoring at the random floor and appearing to beat it.

    Shuffling the columns before the selection and mapping the indices back makes
    tied items equally likely while leaving any genuine ordering untouched.

    Args:
        predictions (Tensor): The score matrix.
        k (int): The cutoff.
        generator (Optional[torch.Generator]): The generator that makes the
            shuffle reproducible. Without one the ordering falls back to the
            deterministic behaviour.

    Returns:
        Tuple[Tensor, Tensor]: The top-k values and the item indices they belong to.
    """
    if generator is None:
        return torch.topk(predictions, k, dim=1)

    order = torch.randperm(
        predictions.size(1), generator=generator, device=predictions.device
    )
    values, shuffled = torch.topk(predictions[:, order], k, dim=1)
    return values, order[shuffled]
