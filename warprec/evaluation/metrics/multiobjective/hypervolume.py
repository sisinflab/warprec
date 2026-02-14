from typing import Any, List
import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("Hypervolume")
class Hypervolume(TopKMetric):
    """Computes the multi-objective Hypervolume (HV) metric based on a list of individual metrics.

    The Hypervolume measures the volume of the objective space dominated by the model's
    performance relative to a reference (nadir) point. In this implementation, it is
    calculated as the product of the distances between the scores and the nadir points.

    Attributes:
        nadir_tensor (Tensor): Tensor of nadir points of sub metrics.

    Args:
        k (int): The number of top recommendations to consider (cutoff).
        num_users (int): Number of users in the training set.
        num_items (int): Number of items in the training set.
        metric_names (List[str]): List of sub metrics to compute.
        nadir_points (List[float]): List of nadir points of the sub metrics.
        higher_is_better (List[bool]): List of booleans that defines wether
            a sub metric wants to be maximized.
        *args (Any): Additional arguments to pass to the parent class.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.

    Raises:
        ValueError: If the provided lists have different lengths.
    """

    nadir_tensor: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        num_items: int,
        metric_names: List[str],
        nadir_points: List[float],
        higher_is_better: List[bool],
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_users = num_users
        self.metric_names = metric_names
        self.higher_better = higher_is_better

        # Ensure all input lists have the same length to avoid indexing errors
        if not (len(metric_names) == len(nadir_points) == len(higher_is_better)):
            raise ValueError(
                "The lists (names, nadir points, directions) must have the same length."
            )

        # Register nadir points as a buffer to ensure they are moved to the correct device (CPU/GPU) automatically
        self.register_buffer(
            "nadir_tensor", torch.tensor(nadir_points, dtype=torch.float32)
        )

        # Dynamically load the component metrics from the registry
        self.sub_metrics: List[BaseMetric] = []
        required_blocks = set()

        for m_name in self.metric_names:
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
        """Computes the Hypervolume for each user.

        HV is calculated as the product of positive distances from the nadir point
        across all defined objectives.
        """
        hv_per_user = None

        for i, metric in enumerate(self.sub_metrics):
            # Compute the sub-metric score
            res = metric.compute()

            # Extract the specific score tensor for the current metric
            score = res.get(
                metric.name,
                torch.zeros(self.num_users, device=self.nadir_tensor.device),
            )
            nadir = self.nadir_tensor[i]

            # Calculate the distance from the nadir based on the optimization direction
            if self.higher_better[i]:
                # Maximization: distance = score - nadir (we want to be above the nadir)
                dist = score - nadir
            else:
                # Minimization: distance = nadir - score (we want to be below the nadir)
                dist = nadir - score

            # Clamp distance to 0: if the score is worse than the nadir, the volume contribution is zero
            dist = torch.clamp(dist, min=0.0)

            # Cumulative product: HV is the volume of the hyper-rectangle
            if hv_per_user is None:
                hv_per_user = dist
            else:
                hv_per_user = hv_per_user * dist

        return {self.name: hv_per_user}

    @property
    def name(self) -> str:
        """Generates a dynamic name for the metric identifying its components,
        nadir points, and directions.
        """
        m_str = ", ".join([m.name for m in self.sub_metrics])
        n_str = ", ".join(map(str, self.nadir_tensor.tolist()))
        h_str = ", ".join([str(h).lower() for h in self.higher_better])
        return f"Hypervolume[{m_str}][{n_str}][{h_str}]"
