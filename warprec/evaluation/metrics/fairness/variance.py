from typing import Any, Dict
import inspect
import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("Variance")
class Variance(TopKMetric):
    """
    Computes the Variance of a specified metric across all users.

    This metric evaluates the consistency of the recommendation quality.
    A high variance suggests that the model performs very well for some users
    but poorly for others.

    Example string format: Variance[Precision]
    """

    def __init__(
            self,
            k: int,
            num_users: int,
            num_items: int,
            metric_name: str,
            dist_sync_on_step: bool = False,
            **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_users = num_users
        self.metric_name = metric_name

        # Dynamically load the target metric to calculate the variance of
        self.target_metric = metric_registry.get(
            metric_name,
            k=k,
            num_users=num_users,
            num_items=num_items,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )

        # Inherit required components from the target metric
        self._REQUIRED_COMPONENTS = self.target_metric._REQUIRED_COMPONENTS

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """
        Updates the target metric state by dynamically inspecting its signature.
        """
        sig = inspect.signature(self.target_metric.update).parameters
        available_data = {
            "preds": preds,
            "user_indices": user_indices,
            **kwargs
        }

        # Filter arguments based on what the target metric's update method expects
        args_to_pass = {k: v for k, v in available_data.items() if k in sig}

        # Handle generic **kwargs if present in the target metric
        if any(p.kind == p.VAR_KEYWORD for p in sig.values()):
            args_to_pass.update(kwargs)

        self.target_metric.update(**args_to_pass)

    def compute(self) -> Dict[str, Tensor]:
        """
        Computes the statistical variance of the metric scores across all users.
        """
        # 1. Get the individual scores for each user
        res = self.target_metric.compute()
        scores = res.get(self.metric_name, torch.zeros(self.num_users))

        # 2. Calculate the variance across the user dimension
        # We use population variance (unbiased=False) or sample variance (unbiased=True)
        # Most evaluation frameworks use the unbiased sample variance.
        variance_val = torch.var(scores.nan_to_num(0), unbiased=True)

        # 3. Return the global variance broadcasted to all users for framework compatibility
        return {self.name: variance_val.repeat(self.num_users)}

    @property
    def name(self) -> str:
        """The name of the metric identifying the target metric."""
        return f"Variance[{self.metric_name}]"