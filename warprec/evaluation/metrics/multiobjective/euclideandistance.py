from typing import Any, List, Dict
import inspect
import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("EucDistance")
class EucDistance(TopKMetric):
    """
    Computes the Euclidean Distance between the model's performance and an Utopia Point (Ideal Point).

    Formula: sqrt( sum( (score_i - utopia_i)^2 ) )

    Usually, a lower Euclidean Distance indicates a better performance as it is closer
    to the ideal objectives.

    Example string format: EucDistance[Precision, Recall](1.0, 1.0)(true, true)
    """

    def __init__(
            self,
            k: int,
            num_users: int,
            num_items: int,
            metric_names: List[str],
            utopia_points: List[float],
            higher_is_better: List[bool],
            dist_sync_on_step: bool = False,
            **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_users = num_users
        self.metric_names = metric_names
        self.higher_better = higher_is_better

        if not (len(metric_names) == len(utopia_points) == len(higher_is_better)):
            raise ValueError("The lists (names, utopia points, directions) must have the same length.")

        # Register utopia points as a buffer for automatic device management (CPU/GPU)
        self.register_buffer("utopia_tensor", torch.tensor(utopia_points, dtype=torch.float32))

        # Dynamically load the component metrics
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
            required_blocks |= m_inst._REQUIRED_COMPONENTS

        self._REQUIRED_COMPONENTS = required_blocks

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """
        Updates sub-metrics by inspecting their signatures and passing only supported arguments.
        """
        for metric in self.sub_metrics:
            sig = inspect.signature(metric.update).parameters
            available_data = {
                "preds": preds,
                "user_indices": user_indices,
                **kwargs
            }

            args_to_pass = {}
            for param_name in sig.keys():
                if param_name in available_data:
                    args_to_pass[param_name] = available_data[param_name]

            has_kwargs_catchall = any(p.kind == p.VAR_KEYWORD for p in sig.values())
            if has_kwargs_catchall:
                for k, v in available_data.items():
                    if k not in args_to_pass:
                        args_to_pass[k] = v

            metric.update(**args_to_pass)

    def compute(self) -> Dict[str, Tensor]:
        """
        Computes the Euclidean distance for each user towards the Utopia Point.
        """
        sum_squared_diff = torch.zeros(self.num_users, device=self.utopia_tensor.device)

        for i, metric in enumerate(self.sub_metrics):
            # 1. Get metric score per user
            res = metric.compute()
            score = res.get(self.metric_names[i], torch.zeros(self.num_users, device=self.utopia_tensor.device))

            utopia = self.utopia_tensor[i]

            # 2. Calculate squared difference
            # Note: Optimization direction is technically handled by the Utopia Point value itself
            # (e.g., Utopia is 1.0 for Precision, 0.0 for Error), but we keep the logic consistent.
            diff = score - utopia
            sum_squared_diff += torch.pow(diff, 2)

        # 3. Final Euclidean distance: sqrt( sum(diff^2) )
        euc_dist = torch.sqrt(sum_squared_diff)

        return {self.name: euc_dist}

    @property
    def name(self) -> str:
        """Constructs the dynamic name for the metric."""
        m_str = ", ".join(self.metric_names)
        u_str = ", ".join(map(str, self.utopia_tensor.tolist()))
        h_str = ", ".join([str(h).lower() for h in self.higher_better])
        return f"EucDistance[{m_str}]({u_str})({h_str})"