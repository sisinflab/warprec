from torch import Tensor

from warprec.evaluation.metrics.bias.long_tail import LongTailMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("ACLT")
class ACLT(LongTailMetric):
    """ACLT (Average Coverage of Long-Tail items) is a metric that evaluates the
    extent to which a recommendation system provides recommendations from the long-tail
    of item popularity. The long-tail is determined based on a given popularity percentile threshold.

    This metric is designed to assess recommendation diversity by measuring the
    proportion of recommended long-tail items relative to all recommendations. A higher
    ACLT value indicates a system that effectively recommends less popular items.
    """

    def _aggregate(self, long_tail_hits: Tensor) -> Tensor:
        """Count the long-tail items recommended to each user.

        Args:
            long_tail_hits (Tensor): The per-user long-tail membership of the top-k.

        Returns:
            Tensor: The count of long-tail recommendations per user.
        """
        return long_tail_hits.sum(dim=1).float()
