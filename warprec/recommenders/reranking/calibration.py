from typing import Any, Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

from warprec.recommenders.reranking.base import GreedyReranker
from warprec.utils.registry import reranker_registry


@reranker_registry.register("Calibration")
class Calibration(GreedyReranker):
    """Keep a list's make-up close to the one the user's own history shows.

    A model trained to maximise accuracy drifts towards whatever a user consumes
    most and drops the rest: someone who watches four fifths drama and one fifth
    comedy is served ten dramas. Calibration asks instead that the proportions of
    the list resemble the proportions of the history, which restores the tail of
    a user's taste without abandoning relevance.

    The divergence between the two distributions is the Kullback-Leibler one, over
    a list distribution smoothed towards the user's so that a genre the list has
    yet to cover does not make it infinite.

    For further details, please refer to Steck, *Calibrated Recommendations*
    (RecSys 2018).

    Args:
        item_features (Tensor): The {item x feature} content matrix, read as the
            make-up of each item.
        pool (int): How many of the top candidates are reconsidered.
        weight (float): How much of the objective is calibration rather than
            relevance. 0 leaves the ranking untouched, 1 ignores the model.
        smoothing (float): How far the list distribution is pulled towards the
            user's before the divergence is taken.
        user_history (Optional[csr_matrix]): The {user x item} training matrix, from
            which each user's own make-up is read. It stays sparse: only the
            product against the item make-up is kept, which is one row per user
            over the features rather than over the catalogue.
        **kwargs (Any): Ignored.

    Raises:
        ValueError: If the weight falls outside [0, 1], or no history was given.
    """

    def __init__(
        self,
        item_features: Tensor,
        pool: int = 100,
        weight: float = 0.5,
        smoothing: float = 0.01,
        user_history: Optional[csr_matrix] = None,
        **kwargs: Any,
    ):
        super().__init__(item_features, pool, **kwargs)

        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Calibration weight must lie in [0, 1], got {weight}.")

        if user_history is None:
            raise ValueError(
                "Calibration compares a list against the user's own history and "
                "therefore needs the training interactions."
            )

        self.weight = weight
        self.smoothing = smoothing

        # The make-up of one item, as a distribution over the features it carries.
        totals = self.item_features.sum(dim=1, keepdim=True).clamp(min=1e-12)
        self.item_mix = self.item_features / totals

        # The make-up of a user, as the average over what they interacted with.
        # The product is taken against the sparse matrix and lands on the features,
        # so what is held is one short row per user rather than the catalogue.
        totals_per_user = np.asarray(user_history.sum(axis=1)).ravel()
        mixed = user_history @ self.item_mix.numpy()
        self.user_mix = torch.as_tensor(
            mixed / np.clip(totals_per_user, 1e-12, None)[:, None], dtype=torch.float
        )

    def _start(
        self, relevance: Tensor, candidates: Tensor, user_indices: Optional[Tensor]
    ) -> dict:
        """Read the target make-up and open an empty list.

        Args:
            relevance (Tensor): The pooled relevance.
            candidates (Tensor): The pooled item indices.
            user_indices (Optional[Tensor]): The users of this batch.

        Returns:
            dict: The target distribution and the running list distribution.

        Raises:
            ValueError: If the batch did not say which users it holds.
        """
        if user_indices is None:
            raise ValueError("Calibration needs to know which users it is ranking for.")

        target = self.user_mix[user_indices]
        return {
            "target": target,
            "accumulated": torch.zeros_like(target),
            "pooled_mix": self.item_mix[candidates],
        }

    def _marginal_gain(
        self,
        relevance: Tensor,
        candidates: Tensor,
        selected: Tensor,
        step: int,
        state: dict,
    ) -> Tensor:
        """Weigh each candidate's relevance against the list it would produce.

        Args:
            relevance (Tensor): The pooled relevance, normalised.
            candidates (Tensor): The pooled item indices.
            selected (Tensor): A mask of the candidates already taken.
            step (int): How many items have been chosen so far.
            state (dict): The target and accumulated distributions.

        Returns:
            Tensor: The objective of each candidate.
        """
        target = state["target"].unsqueeze(1)
        # The list each candidate would produce if it were the next one taken.
        candidate_mix = state["accumulated"].unsqueeze(1) + state["pooled_mix"]
        candidate_mix = candidate_mix / (step + 1)

        smoothed = (1 - self.smoothing) * candidate_mix + self.smoothing * target
        divergence = (
            target * (target.clamp(min=1e-12) / smoothed.clamp(min=1e-12)).log()
        ).sum(dim=2)

        return (1.0 - self.weight) * relevance - self.weight * divergence

    def _advance(self, state: dict, candidates: Tensor, picked: Tensor) -> None:
        """Fold the chosen item's make-up into the list.

        Args:
            state (dict): The target and accumulated distributions.
            candidates (Tensor): The pooled item indices.
            picked (Tensor): The position chosen for each row.
        """
        gather = picked.view(-1, 1, 1).expand(-1, 1, state["pooled_mix"].size(2))
        state["accumulated"] = state["accumulated"] + state["pooled_mix"].gather(
            1, gather
        ).squeeze(1)
