from typing import Any, Set

from torch import Tensor

from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("NewReleaseShare")
class NewReleaseShare(UserAverageTopKMetric):
    """The share of each user's top-k released in or after a given year.

    It reads the release year of every item from the dataset's stash, where the
    guide's callback puts it under 'item_year'. Items without a release year
    count as old.

    Args:
        k (int): The cutoff.
        num_users (int): The number of users in the training set.
        item_year (Tensor): The release year of every item index, from the stash.
        *args (Any): Passed on to the base class.
        since (int): The first year that counts as a new release.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Everything else the evaluator passes, ignored here.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
    }

    def __init__(
        self,
        k: int,
        num_users: int,
        item_year: Tensor,
        *args: Any,
        since: int = 1997,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, num_users, *args, dist_sync_on_step=dist_sync_on_step)
        self.since = since
        # NaN >= since is False, so an unknown year counts as old.
        self.register_buffer("is_new", (item_year >= since).float())

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        """The share of new releases in every user's top-k.

        Args:
            preds (Tensor): The scores, unused here.
            target (Tensor): The binary relevance, unused here.
            top_k_rel (Tensor): The relevance of the top-k, unused here.
            **kwargs (Any): The other blocks, among them the top-k indices and,
                under sampled evaluation, the candidates they index into.

        Returns:
            Tensor: One value per user, shape [batch].
        """
        top_k = self.remap_indices(
            kwargs[f"top_{self.k}_indices"], kwargs.get("item_indices")
        )
        return self.is_new[top_k].mean(dim=1)

    @property
    def name(self) -> str:
        """NewReleaseShare from the default year, NewReleaseShare[since=...] otherwise."""
        if self.since == 1997:
            return "NewReleaseShare"
        return f"NewReleaseShare[since={self.since}]"
