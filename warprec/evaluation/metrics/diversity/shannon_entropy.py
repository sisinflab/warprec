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
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {MetricBlock.TOP_K_INDICES}

    item_counts: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        num_items: int,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_items = num_items
        self.add_state(
            "item_counts", default=torch.zeros(self.num_items), dist_reduce_fx="sum"
        )
        self.add_state("total_recs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        top_k_indices = kwargs.get(f"top_{self.k}_indices")
        item_indices = kwargs.get("item_indices")

        # Handle sampled item indices if provided (map local batch indices to global item IDs)
        item_indices = kwargs.get("item_indices")
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        # Flatten recommendations and count occurrences
        flattened = top_k_indices.flatten().long()

        # Safety check for bounds
        flattened = flattened[flattened < self.num_items]

        # Update state
        self.item_counts += torch.bincount(flattened, minlength=self.num_items)
        self.total_recs += flattened.numel()

    def compute(self):
        # Avoid division by zero
        if self.total_recs == 0:
            return {self.name: torch.tensor(0.0)}

        # Calculate probability distribution
        probs = self.item_counts / self.total_recs

        # Filter out zero probabilities to avoid log(0)
        probs = probs[probs > 0]

        # Compute entropy
        shannon_entropy = -torch.sum(probs * torch.log(probs)).item()
        return {self.name: shannon_entropy}
