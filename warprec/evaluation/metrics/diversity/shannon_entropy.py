# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ShannonEntropy")
class ShannonEntropy(TopKMetric):
    """Shannon Entropy measures the diversity of recommendations by calculating
    the information entropy over item recommendation frequencies.

    The metric formula is defines as:
        ShannonEntropy = -sum(p_i * log(p_i))

    where:
        -p_i is the probability of item i being recommended.

    Matrix computation of the metric:
        PREDS
    +---+---+---+---+
    | 8 | 2 | 7 | 2 |
    | 5 | 4 | 3 | 9 |
    +---+---+---+---+

    We extract the top-k predictions and get their column index. Let's assume k=2:
      TOP-K
    +---+---+
    | 0 | 2 |
    | 3 | 0 |
    +---+---+

    then we compute the item counts using the column indices:
         COUNTS
    +---+---+---+---+
    | 0 | 0 | 1 | 1 |
    +---+---+---+---+

    The probability distribution is calculated by dividing the counts by the total number of recommendations:
           PROBS
    +---+---+-----+-----+
    | 0 | 0 | .25 | .25 |
    +---+---+-----+-----+

    For further details, please refer to this `book <https://link.springer.com/referenceworkentry/10.1007/978-1-4939-7131-2_110158>`_.

    Attributes:
        item_counts (Tensor): Cumulative count of each item's recommendations
        users (Tensor): Total number of users evaluated

    Args:
        k (int): Recommendation list cutoff
        num_items (int): Number of items in the training set.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
    }

    item_counts: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        num_items: int,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_items = num_items

        self.add_state(
            "item_counts", default=torch.zeros(self.num_items), dist_reduce_fx="sum"
        )

        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )
        if "item_indices" in kwargs and kwargs.get("item_indices") is not None:
            top_k_indices = torch.gather(kwargs.get("item_indices"), 1, top_k_indices)

        # Flatten recommendations and count occurrences
        flattened = top_k_indices.flatten().long()

        # Update state
        self.item_counts += torch.bincount(flattened, minlength=self.num_items)
        self.users += users

    def compute(self):
        """Calculate final entropy value."""
        if self.users == 0:
            return torch.tensor(0.0)

        # Calculate probability distribution
        total_recs = self.users * self.k
        probs = self.item_counts / total_recs

        # Compute entropy with numerical stability
        shannon_entropy = -torch.sum(
            probs * torch.log(probs + 1e-12)
        ).item()  # Avoid log(0)
        return {self.name: shannon_entropy}
