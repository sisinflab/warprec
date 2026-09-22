from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from torch import Tensor


class Reranker(ABC):
    """Reorders the head of a ranking before anything else looks at it.

    A re-ranker answers a different question from the model: not "what is this
    user most likely to interact with" but "which list serves them best". It runs
    over a pool of the highest-scoring candidates rather than the whole catalogue,
    because the objectives are greedy and quadratic in the number of candidates,
    and because there is nothing to gain from reconsidering items the model has
    already ranked far below the cut.

    The original scores travel through unchanged, only reordered, so a written
    recommendation still carries the number the model produced.

    Args:
        item_features (Tensor): The {item x feature} content matrix, dense.
        pool (int): How many of the top candidates are reconsidered.
        **kwargs (Any): Ignored, so that a registry may pass shared arguments.

    Raises:
        ValueError: If the pool is smaller than one.
    """

    def __init__(self, item_features: Tensor, pool: int = 100, **kwargs: Any):
        if pool < 1:
            raise ValueError(
                f"A re-ranking pool must hold at least one item, got {pool}."
            )

        self.item_features = item_features.float()
        self.pool = pool

    @abstractmethod
    def _select(
        self,
        relevance: Tensor,
        candidates: Tensor,
        k: int,
        user_indices: Optional[Tensor],
    ) -> Tensor:
        """Choose k of the pooled candidates, in the order they should appear.

        Args:
            relevance (Tensor): The pooled scores, [batch_size, pool].
            candidates (Tensor): The pooled item indices, [batch_size, pool].
            k (int): How many to choose.
            user_indices (Optional[Tensor]): The users of this batch.

        Returns:
            Tensor: Positions within the pool, [batch_size, k], in order.
        """

    @staticmethod
    def _normalised(relevance: Tensor) -> Tensor:
        """Scale each row's relevance onto [0, 1].

        The objectives trade relevance off against a similarity or a divergence,
        both of which live on [0, 1]. Without this the trade-off weight would mean
        something different for every model and every dataset.

        Args:
            relevance (Tensor): The pooled scores.

        Returns:
            Tensor: The scores, per row, on a common scale.
        """
        finite = torch.where(
            torch.isfinite(relevance), relevance, torch.zeros_like(relevance)
        )
        lowest = finite.min(dim=1, keepdim=True).values
        highest = finite.max(dim=1, keepdim=True).values
        return (finite - lowest) / (highest - lowest).clamp(min=1e-12)

    def __call__(
        self, predictions: Tensor, k: int, user_indices: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Reorder the head of each row.

        Args:
            predictions (Tensor): The score matrix.
            k (int): The cutoff.
            user_indices (Optional[Tensor]): The users of this batch, needed by the
                objectives that read a user's history.

        Returns:
            Tuple[Tensor, Tensor]: The scores and the item indices they belong to,
                in the order the re-ranker chose.
        """
        pool = min(self.pool, predictions.size(1))
        cut = min(k, pool)

        relevance, candidates = torch.topk(predictions, pool, dim=1)
        chosen = self._select(relevance, candidates, cut, user_indices)

        values = relevance.gather(1, chosen)
        indices = candidates.gather(1, chosen)

        # A cutoff deeper than the pool keeps the ranking the model produced, so
        # asking for more than was reconsidered never loses items.
        if cut < k:
            values = torch.cat([values, relevance[:, cut:k]], dim=1)
            indices = torch.cat([indices, candidates[:, cut:k]], dim=1)

        return values, indices


class GreedyReranker(Reranker):
    """A re-ranker that builds its list one item at a time.

    Every objective here is submodular and is optimised the same way: start empty
    and repeatedly add whichever remaining candidate improves the list most. The
    subclasses differ only in what "most" means.
    """

    @abstractmethod
    def _marginal_gain(
        self,
        relevance: Tensor,
        candidates: Tensor,
        selected: Tensor,
        step: int,
        state: dict,
    ) -> Tensor:
        """Score every candidate by what it would add to the list so far.

        Args:
            relevance (Tensor): The pooled relevance, already normalised.
            candidates (Tensor): The pooled item indices.
            selected (Tensor): A mask of the candidates already taken.
            step (int): How many items have been chosen so far.
            state (dict): Whatever the subclass carries between steps.

        Returns:
            Tensor: The gain of each candidate, [batch_size, pool].
        """

    @abstractmethod
    def _start(
        self, relevance: Tensor, candidates: Tensor, user_indices: Optional[Tensor]
    ) -> dict:
        """Prepare whatever the objective needs before the first step.

        Args:
            relevance (Tensor): The pooled relevance.
            candidates (Tensor): The pooled item indices.
            user_indices (Optional[Tensor]): The users of this batch.

        Returns:
            dict: The state handed to each step.
        """

    def _advance(self, state: dict, candidates: Tensor, picked: Tensor) -> None:
        """Fold the chosen item into the state.

        Args:
            state (dict): The state carried between steps.
            candidates (Tensor): The pooled item indices.
            picked (Tensor): The position chosen for each row.
        """

    def _select(
        self,
        relevance: Tensor,
        candidates: Tensor,
        k: int,
        user_indices: Optional[Tensor],
    ) -> Tensor:
        """Build the list greedily.

        Args:
            relevance (Tensor): The pooled scores.
            candidates (Tensor): The pooled item indices.
            k (int): How many to choose.
            user_indices (Optional[Tensor]): The users of this batch.

        Returns:
            Tensor: Positions within the pool, in the order chosen.
        """
        scaled = self._normalised(relevance)
        state = self._start(scaled, candidates, user_indices)

        rows, pool = relevance.shape
        taken = torch.zeros(rows, pool, dtype=torch.bool, device=relevance.device)
        order = torch.empty(rows, k, dtype=torch.long, device=relevance.device)

        for step in range(k):
            gain = self._marginal_gain(scaled, candidates, taken, step, state)
            gain = gain.masked_fill(taken, -torch.inf)

            picked = gain.argmax(dim=1)
            order[:, step] = picked
            taken.scatter_(1, picked.unsqueeze(1), True)
            self._advance(state, candidates, picked)

        return order
