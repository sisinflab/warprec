from typing import ClassVar

from warprec.evaluation.metrics.novelty.novelty_profile import NoveltyProfileMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("EPC")
class EPC(NoveltyProfileMetric):
    """Expected Popularity Complement at K metric.

    This metric measures the capability of the recommender system
    to suggest items that are not popular.
    """

    _LOG_DISCOUNT: ClassVar[bool] = False
