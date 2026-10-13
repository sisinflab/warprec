from typing import Any, List
import torch
from torch import Tensor, nn
from warprec.errors import ConfigurationError, EvaluationError
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("EucDistance")
class EucDistance(TopKMetric):
    """Computes the Euclidean Distance between the model's performance and an Utopia Point (Ideal Point).

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
        ConfigurationError: If the provided lists have different lengths.
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

        if len(metric_names) != len(utopia_points):
            raise ConfigurationError(
                "The lists (names, utopia points) must have the same length."
            )

        # Register utopia points as a buffer to ensure they are moved to the correct device (CPU/GPU) automatically
        self.register_buffer(
            "utopia_tensor", torch.tensor(utopia_points, dtype=torch.float32)
        )

        # Dynamically load the component metrics from the registry
        # A ModuleList rather than a list, so that moving the composite moves its
        # sub-metrics' states with it. It holds BaseMetric instances only.
        self.sub_metrics: List[BaseMetric] = nn.ModuleList()  # type: ignore[assignment]
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
            required_blocks |= m_inst.components

        self._REQUIRED_COMPONENTS = required_blocks

    def reset(self) -> None:
        """Reset the sub-metrics along with the composite.

        torchmetrics resets a metric's own states only, and the composite keeps
        none: without this its sub-metrics would carry every evaluation that
        reused the evaluator into the next one.
        """
        super().reset()
        for metric in self.sub_metrics:
            metric.reset()

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions.

        The user indices travel in the keyword arguments, as the evaluator passes
        them: a user-averaged sub-metric reads them by name, a global one ignores
        them.

        Args:
            preds (Tensor): The prediction tensor.
            **kwargs (Any): The keyword arguments the evaluator gives every metric.
        """
        for metric in self.sub_metrics:
            metric.update(preds, **kwargs)

    def compute(self) -> dict:
        """Computes the Euclidean distance for each user towards the Utopia Point.

        A global sub-metric gives one score that every user shares, so the distance
        is per user when any sub-metric is user-averaged and a single float, as
        other global metrics give, when none is.

        Returns:
            dict: The distance under the metric's name.

        Raises:
            EvaluationError: If a sub-metric gives no single value under its name.
        """
        sum_squared_diff = torch.zeros((), device=self.utopia_tensor.device)

        for i, metric in enumerate(self.sub_metrics):
            # Compute the sub-metric score
            res = metric.compute()

            # A metric reporting several values (one per cluster pair, say) has
            # no single score to combine; a zero in its place would pass silently
            if metric.name not in res:
                raise EvaluationError(
                    f"EucDistance cannot combine {metric.name}: it reports "
                    f"{', '.join(res)} rather than one value under its name. "
                    "Pick sub-metrics that give a single score."
                )
            score = res[metric.name]

            utopia = self.utopia_tensor[i]

            # Compute squared difference
            # NOTE: Optimization direction is technically handled by the Utopia Point value itself
            # (e.g., Utopia is 1.0 for Precision, 0.0 for Error), but we keep the logic consistent.
            diff = score - utopia
            sum_squared_diff = sum_squared_diff + torch.pow(diff, 2)

        # Final Euclidean distance: sqrt(sum(diff^2))
        euc_dist = torch.sqrt(sum_squared_diff)

        if euc_dist.dim() == 0:
            return {self.name: euc_dist.item()}
        return {self.name: euc_dist}

    @property
    def name(self) -> str:
        """Generates a dynamic name for the metric identifying its components,
        utopia points, and directions."""
        m_str = ", ".join([m.name for m in self.sub_metrics])
        u_str = ", ".join(map(str, self.utopia_tensor.tolist()))
        return f"EucDistance[{m_str}][{u_str}]"
