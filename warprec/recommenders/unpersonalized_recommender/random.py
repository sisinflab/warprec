# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import Tensor

from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry


@model_registry.register(name="Random")
class Random(Recommender):
    """Definition of Random unpersonalized model.
    This model will recommend items based on a random number generator.
    """

    @classmethod
    def estimate_space(
        cls,
        params: dict,
        info: dict,
        interactions: Optional[Interactions] = None,
        **kwargs: Any,
    ) -> dict:
        return {
            "train_ram_mb": 0.0,
            "notes": "Random analytical train-space estimate",
        }

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using a normalized popularity value.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Drawn from the model's own seed rather than the global stream, so that
        # the baseline one run is compared against is the baseline the next run
        # is compared against. The draw follows a user's position in the batch
        # rather than their identity, which is enough for a baseline: the point
        # is an arbitrary ranking that does not move between runs of the same
        # configuration.
        generator = torch.Generator(device=user_indices.device)
        generator.manual_seed(self.seed)

        shape = (user_indices.size(0), self.n_items)
        scores = torch.rand(shape, generator=generator, device=user_indices.device)

        if item_indices is None:
            # Case 'full': prediction on all items
            return scores  # [batch_size, n_items]

        # Case 'sampled': taken from the same draw, so that asking about a few
        # items gives what ranking them all would have given.
        return scores.gather(1, item_indices.clamp(max=self.n_items - 1))
