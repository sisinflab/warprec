from typing import Any, Optional

import torch
from torch import Tensor

from warprec.recommenders.reranking.base import GreedyReranker
from warprec.utils.registry import reranker_registry


@reranker_registry.register("MMR")
class MMR(GreedyReranker):
    """Maximal Marginal Relevance: trade relevance off against redundancy.

    At each step the item chosen is the one that is both well scored and unlike
    what the list already holds, which is what stops a ranking made of ten
    near-identical items. Similarity is read from the item content matrix, the
    same space the content-based models score in.

    For further details, please refer to Carbonell and Goldstein, *The Use of MMR,
    Diversity-Based Reranking for Reordering Documents and Producing Summaries*
    (SIGIR 1998).

    Args:
        item_features (Tensor): The {item x feature} content matrix.
        pool (int): How many of the top candidates are reconsidered.
        diversity (float): How much of the objective is redundancy rather than
            relevance. 0 leaves the ranking untouched, 1 ignores the model.
        **kwargs (Any): Ignored.

    Raises:
        ValueError: If the diversity weight falls outside [0, 1].
    """

    def __init__(
        self,
        item_features: Tensor,
        pool: int = 100,
        diversity: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__(item_features, pool, **kwargs)

        if not 0.0 <= diversity <= 1.0:
            raise ValueError(f"MMR diversity must lie in [0, 1], got {diversity}.")

        self.diversity = diversity

        # Normalised once, so that every pooled similarity is a plain dot product.
        norms = self.item_features.norm(dim=1, keepdim=True).clamp(min=1e-12)
        self.unit_features = self.item_features / norms

    def _start(
        self, relevance: Tensor, candidates: Tensor, user_indices: Optional[Tensor]
    ) -> dict:
        """Precompute the similarity between the pooled items.

        Args:
            relevance (Tensor): The pooled relevance.
            candidates (Tensor): The pooled item indices.
            user_indices (Optional[Tensor]): Unused.

        Returns:
            dict: The pooled similarity and the running redundancy.
        """
        pooled = self.unit_features[candidates]
        similarity = torch.bmm(pooled, pooled.transpose(1, 2))

        return {
            "similarity": similarity,
            "redundancy": torch.zeros_like(relevance),
        }

    def _marginal_gain(
        self,
        relevance: Tensor,
        candidates: Tensor,
        selected: Tensor,
        step: int,
        state: dict,
    ) -> Tensor:
        """Weigh each candidate's relevance against its closest chosen neighbour.

        Args:
            relevance (Tensor): The pooled relevance, normalised.
            candidates (Tensor): The pooled item indices.
            selected (Tensor): A mask of the candidates already taken.
            step (int): How many items have been chosen so far.
            state (dict): The pooled similarity and running redundancy.

        Returns:
            Tensor: The marginal relevance of each candidate.
        """
        if step == 0:
            # Nothing has been chosen, so there is no redundancy to answer for.
            return relevance

        return (1.0 - self.diversity) * relevance - self.diversity * state["redundancy"]

    def _advance(self, state: dict, candidates: Tensor, picked: Tensor) -> None:
        """Raise each candidate's redundancy to account for the new neighbour.

        Args:
            state (dict): The pooled similarity and running redundancy.
            candidates (Tensor): The pooled item indices.
            picked (Tensor): The position chosen for each row.
        """
        gather = picked.view(-1, 1, 1).expand(-1, 1, state["similarity"].size(2))
        to_picked = state["similarity"].gather(1, gather).squeeze(1)
        state["redundancy"] = torch.maximum(state["redundancy"], to_picked)
