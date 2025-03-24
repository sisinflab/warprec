# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


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
        train_set (csr_matrix): The training interaction data.
        dist_sync_on_step (bool): Synchronize metric state across devices
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    item_counts: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        dist_sync_on_step: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_items = train_set.shape[1]

        self.add_state(
            "item_counts", default=torch.zeros(self.num_items), dist_reduce_fx="sum"
        )

        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        # Get top-k recommended item indices
        top_k = torch.topk(preds, self.k, dim=1).indices

        # Flatten recommendations and count occurrences
        flattened = top_k.flatten().long()
        batch_counts = torch.bincount(flattened, minlength=self.num_items)

        # Update state
        self.item_counts += batch_counts
        self.users += preds.shape[0]  # Track total users

    def compute(self):
        """Calculate final entropy value."""
        if self.users == 0:
            return torch.tensor(0.0)

        # Calculate probability distribution
        total_recs = self.users * self.k
        probs = self.item_counts / total_recs

        # Compute entropy with numerical stability
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))  # Avoid log(0)
        return entropy

    def reset(self):
        """Reset metric state."""
        self.item_counts.zero_()
        self.users.zero_()
