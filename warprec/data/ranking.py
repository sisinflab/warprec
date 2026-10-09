from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

from warprec.data.entities.context import context_key
from warprec.utils.logger import logger


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


def resolve_recommendation_mask(policy: str, transactions: Any) -> str:
    """Which seen-item rule the written recommendations can follow.

    Evaluation asks a question about a moment: this user, in this situation, was
    shown this item. Writing recommendations asks a question about a user alone,
    so there is no situation to compare a history against and a contextual policy
    has nothing to resolve. It falls back to excluding everything the user has
    seen, which is the conservative reading, and says so rather than leaving the
    written list quietly filtered by a rule the configuration did not ask for.

    'none' and 'pair' mean the same here as they do in evaluation, so a run that
    asks for either gets it.

    Args:
        policy (str): The configured policy, one of 'auto', 'context', 'pair' or
            'none'.
        transactions (Any): The row-oriented training records, or None when the
            dataset has no contextual columns.

    Returns:
        str: The rule to apply, either 'pair' or 'none'.
    """
    resolved = resolve_mask_policy(policy, transactions)
    if resolved != "context":
        return resolved

    logger.attention(
        "Recommendations are written for a user rather than for a user in a "
        "situation, so the contextual seen-item rule cannot be applied to them. "
        "Every item the user has already interacted with is excluded instead."
    )
    return "pair"


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

    Every user's ties are drawn on their own: each item of an ambiguous row gets a
    random key, and the row is ranked by its score first and by that key among
    equal scores. So a model that ties a whole population gives each user an
    independent uniform draw of it, in a uniformly random order, and its score is
    an average over the users rather than the luck of one draw shared by a batch.
    A higher score still always ranks above a lower one.

    Only the rows whose ranking is genuinely ambiguous get keys, and a row with no
    ties comes back exactly as ``torch.topk`` returns it. Finding them costs one
    position past the cutoff; on real model scores almost no row is ambiguous.

    Args:
        predictions (Tensor): The score matrix.
        k (int): The cutoff.
        generator (Optional[torch.Generator]): The generator that makes the
            draw reproducible. Without one the ordering falls back to the
            deterministic behaviour of ``torch.topk``.

    Returns:
        Tuple[Tensor, Tensor]: The top-k values, in descending order, and the
            item indices they belong to.
    """
    if generator is None:
        return torch.topk(predictions, k, dim=1)

    # One extra position past the cutoff is enough to find the rows that need a
    # draw: a row with no two equal scores among them has both its membership
    # and its order already settled.
    probe = min(k + 1, predictions.size(1))
    values, indices = torch.topk(predictions, probe, dim=1)

    ambiguous = (values[:, :-1] == values[:, 1:]).any(dim=1)
    values, indices = values[:, :k], indices[:, :k]
    if not bool(ambiguous.any()):
        return values, indices

    rows = ambiguous.nonzero(as_tuple=True)[0]
    scores = predictions if rows.numel() == predictions.size(0) else predictions[rows]

    # The k-th score splits each row in three: the items above it are in the
    # list whatever the draw, the items below it are out, and the items equal to
    # it compete for the places left.
    kth = values[rows, k - 1 :]

    # Drawing the keys is the dominant cost, so they are drawn only for the
    # items that can still make some list. When the ties sit on a known
    # population, such as the cold items of a cold-start run, that is a small
    # fraction of the catalogue. When it is most of it, narrowing would only
    # add a copy of the scores.
    columns = (scores >= kth).any(dim=0).nonzero(as_tuple=True)[0]
    narrowed = columns.numel() <= scores.size(1) // 2
    if narrowed:
        scores = scores[:, columns]

    # Keying the items above the k-th score over every random key, and the ones
    # below it under, lets one top-k make the choice for every row at once, with
    # a fresh key per user and item. Drawn where the generator lives, which
    # torch requires, then moved to the scores: the evaluator's generator is on
    # the CPU even when the scores are not.
    keys = torch.rand(scores.shape, generator=generator, device=generator.device).to(
        predictions.device
    )
    keys.masked_fill_(scores < kth, -1.0)
    keys[scores > kth] += 2.0
    _, chosen = torch.topk(keys, k, dim=1)
    del keys

    # The draw settled which items are in, and in key order. Sorting them by
    # score, stably, puts a higher score first and leaves equal scores in the
    # order their keys drew.
    tied_values, order = torch.sort(
        scores.gather(1, chosen), dim=1, descending=True, stable=True
    )
    chosen = chosen.gather(1, order)
    if narrowed:
        chosen = columns[chosen]

    values, indices = values.clone(), indices.clone()
    values[rows] = tied_values
    indices[rows] = chosen
    return values, indices
