from typing import Any, List
import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("EucDistance")
class EucDistance(TopKMetric):
    """Computes the Euclidean Distance between the model's performance and an Utopia Point (Ideal Point).

    Formula: sqrt(sum((score_i - utopia_i)^2))

    Usually, a lower Euclidean Distance indicates a better performance as it is closer
    to the ideal objectives.

    Attributes:
        utopia_tensor (Tensor): Tensor of utopia points of sub metrics.

    Args:
        k (int): The number of top recommendations to consider (cutoff).
        num_users (int): Number of users in the training set.
        num_items (int): Number of items in the training set.
        metric_names (List[str]): List of sub metrics to compute.
        utopia_points (List[float]): List of utopia points of the sub metrics.
        *args (Any): Additional arguments to pass to the parent class.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.

    Raises:
        ValueError: If the provided lists have different lengths.
    """

    utopia_tensor: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        num_items: int,
        metric_names: List[str],
        utopia_points: List[float],
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_users = num_users

        if not (len(metric_names) == len(utopia_points)):
            raise ValueError(
                "The lists (names, utopia points) must have the same length."
            )

        # Register utopia points as a buffer to ensure they are moved to the correct device (CPU/GPU) automatically
        self.register_buffer(
            "utopia_tensor", torch.tensor(utopia_points, dtype=torch.float32)
        )

        # Dynamically load the component metrics from the registry
        self.sub_metrics: List[BaseMetric] = []
        required_blocks = set()

        for m_name in metric_names:
            m_inst = metric_registry.get(
                m_name,
                k=k,
                num_users=num_users,
                num_items=num_items,
                dist_sync_on_step=dist_sync_on_step,
                **kwargs,
            )
            self.sub_metrics.append(m_inst)
            # Collect and union all required data components from sub-metrics
            required_blocks |= m_inst._REQUIRED_COMPONENTS

        self._REQUIRED_COMPONENTS = required_blocks

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        for metric in self.sub_metrics:
            metric.update(preds, user_indices, **kwargs)

    def compute(self):
        """Computes the Euclidean distance for each user towards the Utopia Point."""
        sum_squared_diff = torch.zeros(self.num_users, device=self.utopia_tensor.device)

        for i, metric in enumerate(self.sub_metrics):
            # Compute the sub-metric score
            res = metric.compute()

            # Extract the specific score tensor for the current metric
            score = res.get(
                metric.name,
                torch.zeros(self.num_users, device=self.utopia_tensor.device),
            )

            utopia = self.utopia_tensor[i]

            # Compute squared difference
            # NOTE: Optimization direction is technically handled by the Utopia Point value itself
            # (e.g., Utopia is 1.0 for Precision, 0.0 for Error), but we keep the logic consistent.
            diff = score - utopia
            sum_squared_diff += torch.pow(diff, 2)

        # Final Euclidean distance: sqrt(sum(diff^2))
        euc_dist = torch.sqrt(sum_squared_diff)

        return {self.name: euc_dist}

    @property
    def name(self) -> str:
        """Generates a dynamic name for the metric identifying its components,
        utopia points, and directions."""
        m_str = ", ".join([m.name for m in self.sub_metrics])
        u_str = ", ".join(map(str, self.utopia_tensor.tolist()))
        return f"EucDistance[{m_str}][{u_str}]"
