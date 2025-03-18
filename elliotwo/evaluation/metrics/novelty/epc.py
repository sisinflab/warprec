from typing import Any
import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("EPC")
class EPC(TopKMetric):
    """
    Expected Popularity Complement at K metric.

    Attributes:
        epc (Tensor): The EPC value for every user.
        users (Tensor): Number of users evaluated.

    Args:
        k (int): The cutoff for recommendations.
        novelty_profile (Tensor): The novelty profile tensor that measures popularity.
        dist_sync_on_step (bool): Torchmetrics parameter for distributed synchronization.
        *args (Any): Additional arguments.
        **kwargs (Any): Additional keyword arguments.
    """

    epc: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        novelty_profile: Tensor,
        dist_sync_on_step: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.novelty_profile = novelty_profile.unsqueeze(0)
        self.add_state("epc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with a new batch of predictions."""
        target = target.clone()
        target[target > 0] = 1
        top_k = torch.topk(preds, self.k, dim=1, largest=True, sorted=True).indices
        rel = torch.gather(target, 1, top_k).float()

        # Extract novelty values
        batch_novelty = self.novelty_profile.repeat(
            target.shape[0], 1
        )  # [batch_size x items]
        novelty = torch.gather(batch_novelty, 1, top_k)  # [batch_size x top_k]

        # Update
        self.epc += self.dcg(rel * novelty).sum()
        self.users += target.shape[0]

    def compute(self):
        """Computes the final value of the metric."""
        return (
            self.epc / (self.users * self.discounted_sum(self.k))
            if self.users > 0
            else torch.tensor(0.0)
        )
