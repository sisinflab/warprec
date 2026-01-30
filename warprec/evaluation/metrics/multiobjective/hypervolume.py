from typing import Any, List, Dict
import inspect
import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("Hypervolume")
class Hypervolume(TopKMetric):
    """
    Computes the multi-objective Hypervolume (HV) metric based on a list of individual metrics.

    The Hypervolume measures the volume of the objective space dominated by the model's
    performance relative to a reference (nadir) point. In this implementation, it is
    calculated as the product of the distances between the scores and the nadir points.

    Example string format: Hypervolume[Precision, Recall](0.0, 0.0)(true, true)
    """

    def __init__(
            self,
            k: int,
            num_users: int,
            num_items: int,
            metric_names: List[str],
            nadir_points: List[float],
            higher_is_better: List[bool],
            dist_sync_on_step: bool = False,
            **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_users = num_users
        self.metric_names = metric_names
        self.higher_better = higher_is_better

        # Ensure all input lists have the same length to avoid indexing errors
        if not (len(metric_names) == len(nadir_points) == len(higher_is_better)):
            raise ValueError("The lists (names, nadir points, directions) must have the same length.")

        # Register nadir points as a buffer to ensure they are moved to the correct device (CPU/GPU) automatically
        self.register_buffer("nadir_tensor", torch.tensor(nadir_points, dtype=torch.float32))

        # Dynamically load the component metrics from the registry
        self.sub_metrics = torch.nn.ModuleList()
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
            # Collect and union all required data components (e.g., target, items) from sub-metrics
            required_blocks |= m_inst._REQUIRED_COMPONENTS

        self._REQUIRED_COMPONENTS = required_blocks

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """
        Updates the state of sub-metrics by dynamically filtering parameters based on their signatures.
        This handles metrics with different 'update' inputs (e.g., some requiring user_indices, others not).
        """
        for metric in self.sub_metrics:
            # Get the parameters accepted by the sub-metric's update method
            sig = inspect.signature(metric.update).parameters

            # Consolidate all available data into a single dictionary
            available_data = {
                "preds": preds,
                "user_indices": user_indices,
                **kwargs  # May include optional data like 'target', 'labels', etc.
            }

            args_to_pass = {}

            # Case 1: Match arguments by explicit name (e.g., 'preds', 'user_indices', or 'target')
            for param_name in sig.keys():
                if param_name in available_data:
                    args_to_pass[param_name] = available_data[param_name]

            # Case 2: Handle metrics that accept a generic **kwargs (VAR_KEYWORD)
            has_kwargs_catchall = any(p.kind == p.VAR_KEYWORD for p in sig.values())
            if has_kwargs_catchall:
                for k, v in available_data.items():
                    if k not in args_to_pass:
                        args_to_pass[k] = v

            # Call update with the filtered arguments safely
            metric.update(**args_to_pass)

    def compute(self) -> Dict[str, Tensor]:
        """
        Computes the Hypervolume for each user.
        HV is calculated as the product of positive distances from the nadir point
        across all defined objectives.
        """
        hv_per_user = None

        for i, metric in enumerate(self.sub_metrics):
            # 1. Compute the sub-metric score (returns a dict {metric_name: tensor_per_user})
            res = metric.compute()

            # Extract the specific score tensor for the current metric
            score = res.get(self.metric_names[i], torch.zeros(self.num_users, device=self.nadir_tensor.device))
            nadir = self.nadir_tensor[i]

            # 2. Calculate the distance from the nadir based on the optimization direction
            if self.higher_better[i]:
                # Maximization: distance = score - nadir (we want to be above the nadir)
                dist = score - nadir
            else:
                # Minimization: distance = nadir - score (we want to be below the nadir)
                dist = nadir - score

            # 3. Clamp distance to 0: if the score is worse than the nadir, the volume contribution is zero
            dist = torch.clamp(dist, min=0.0)

            # 4. Cumulative product: HV is the volume of the hyper-rectangle
            if hv_per_user is None:
                hv_per_user = dist
            else:
                hv_per_user = hv_per_user * dist

        return {self.name: hv_per_user}

    @property
    def name(self) -> str:
        """
        Generates a dynamic name for the metric identifying its components,
        nadir points, and directions.
        """
        m_str = ", ".join(self.metric_names)
        n_str = ", ".join(map(str, self.nadir_tensor.tolist()))
        h_str = ", ".join([str(h).lower() for h in self.higher_better])
        return f"Hypervolume[{m_str}]({n_str})({h_str})"