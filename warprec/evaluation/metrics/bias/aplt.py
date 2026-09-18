from torch import Tensor

from warprec.evaluation.metrics.bias.long_tail import LongTailMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("APLT")
class APLT(LongTailMetric):
    """APLT (Average Proportion of Long-Tail items) is a metric that evaluates
    the proportion of long-tail items present in the top-k recommendations.
    Unlike ACLT, which focuses on the number of long-tail recommendations, APLT normalizes
    by the total number of recommended items, providing a proportional measure.

    This metric helps analyze how well a recommendation system balances diversity
    by incorporating less popular items into recommendations while maintaining relevance.
    """

    def _aggregate(self, long_tail_hits: Tensor) -> Tensor:
        """Take the share of each user's top-k that falls in the long tail.

        Args:
            long_tail_hits (Tensor): The per-user long-tail membership of the top-k.

        Returns:
            Tensor: The proportion of long-tail recommendations per user.
        """
        return long_tail_hits.sum(dim=1).float() / self.k
