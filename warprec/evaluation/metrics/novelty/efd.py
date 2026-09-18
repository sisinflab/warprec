from typing import ClassVar

from warprec.evaluation.metrics.novelty.novelty_profile import NoveltyProfileMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("EFD")
class EFD(NoveltyProfileMetric):
    """Expected Free Discovery at K metric.

    This metric measures the recommender system's ability to suggest items
    that the user has not already seen (i.e., not present in the training set).
    """

    _LOG_DISCOUNT: ClassVar[bool] = True
