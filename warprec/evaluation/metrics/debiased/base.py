from typing import Any, Optional, Set, Tuple

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock


class InversePropensityMetric(UserAverageTopKMetric):
    """Common machinery for the estimators that correct for exposure bias.

    Each of them divides a relevant item's contribution by the probability that
    the item was observed at all, which is what removes the logging policy's
    popularity from the result. They differ only in what they divide the total
    by: the plain estimators use the number of relevant items, the self-
    normalised ones use the sum of the weights, which trades a little bias for a
    large reduction in variance.

    Attributes:
        propensity (Tensor): The observation probability of each item.

    Args:
        k (int): The cutoff.
        num_users (int): Number of users in the training set.
        *args (Any): The argument list.
        propensity (Optional[Tensor]): One observation probability per item.
            Without it the metric cannot correct anything and says so.
        **kwargs (Any): Additional keyword arguments for the parent class.

    Raises:
        ValueError: If no propensity was provided by the evaluation configuration.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    # Registered as a buffer, so it follows the metric onto whatever device the
    # evaluation runs on. Annotated because a buffer is otherwise typed loosely.
    propensity: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        *args: Any,
        propensity: Optional[Tensor] = None,
        **kwargs: Any,
    ):
        super().__init__(k, num_users, *args, **kwargs)

        if propensity is None:
            raise ValueError(
                f"{type(self).__name__} corrects for exposure bias and therefore "
                "needs propensities. Set 'evaluation.propensity.estimator' to "
                "'popularity'."
            )

        self.register_buffer("propensity", propensity.clone().float())

    def weighted_hits(self, top_k_rel: Tensor, top_k_indices: Tensor) -> Tensor:
        """The relevance of each retrieved item, divided by its propensity.

        Args:
            top_k_rel (Tensor): The relevance of the top-k items.
            top_k_indices (Tensor): The indices of the top-k items.

        Returns:
            Tensor: The corrected relevance, [batch_size, k].
        """
        return top_k_rel.float() / self.propensity[top_k_indices]

    def weight_total(self, target: Tensor) -> Tensor:
        """The sum of the weights of every relevant item a user has.

        This is the self-normalising constant: dividing by it rather than by the
        number of relevant items is what makes an estimator 'self-normalised'.

        Args:
            target (Tensor): The relevance of every item, per user.

        Returns:
            Tensor: The summed weights, [batch_size].
        """
        return ((target > 0).float() / self.propensity).sum(dim=1)

    def unpack_inputs(
        self, preds: Tensor, **kwargs: Any
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Retrieve the relevance, the valid users and the top-k relevance.

        Args:
            preds (Tensor): The prediction tensor.
            **kwargs (Any): The precomputed blocks.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The target, the valid users and the
                top-k relevance.
        """
        target = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )
        return target, users, top_k_rel

    def top_k_indices(self, preds: Tensor, **kwargs: Any) -> Tensor:
        """The indices of the retrieved items, precomputed or derived.

        Args:
            preds (Tensor): The prediction tensor.
            **kwargs (Any): The precomputed blocks.

        Returns:
            Tensor: The top-k indices.
        """
        indices = kwargs.get(f"top_{self.k}_indices")
        if indices is None:
            _, indices = torch.topk(preds, self.k, dim=1)
        return indices
