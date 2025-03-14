# pylint: disable=arguments-differ
from abc import abstractmethod, ABC
from typing import Any

import torch
from torchmetrics import Metric
from torch import Tensor
from elliotwo.utils.registry import metric_registry


class BaseMetric(Metric, ABC):
    """The base definition of a metric using Torchmetrics."""

    @abstractmethod
    def compute(self):
        pass

    @property
    def name(self):
        """The name of the metric."""
        return self.__class__.__name__


class TopKMetric(BaseMetric):
    """The definition of a Top-K metric."""

    def __init__(self, k: int, dist_sync_on_step=False, *args: Any, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k


@metric_registry.register("Precision")
class Precision(TopKMetric):
    """The Precision@k counts the number of item retrieved correctly,
        over the maximum number of possible retrieve items.

    Attributes:
        correct (Tensor): The number of hits in the top-k recommendations.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    correct: Tensor
    users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        target = target.clone()
        target[target > 0] = 1
        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)
        self.correct += rel.sum().float()
        self.users += target.shape[0]

    def compute(self):
        return (
            self.correct / (self.users * self.k)
            if self.users > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        self.correct.zero_()
        self.users.zero_()


@metric_registry.register("Recall")
class Recall(TopKMetric):
    """The Recall@k counts the number of item retrieve correctly,
        over the total number of relevant item in the ground truth.

    Attributes:
        correct (Tensor): The number of hits in the top-k recommendations.
        real_correct (Tensor): The number of real hits in user transactions.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    correct: Tensor
    real_correct: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("real_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        target = target.clone()
        target[target > 0] = 1
        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)
        self.correct += rel.sum().float()
        self.real_correct += target.sum().float()

    def compute(self):
        return (
            self.correct / self.real_correct
            if self.real_correct > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        self.correct.zero_()
        self.real_correct.zero_()


@metric_registry.register("HitRate")
class HitRate(TopKMetric):
    """The HitRate@k metric counts the number of users for which
        the model retrieved at least one item.

    This is normalized by the total number of users.

    Attributes:
        hits (Tensor): The number of hits in the top-k recommendations.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    hits: Tensor
    users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        target = target.clone()
        target[target > 0] = 1
        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)
        self.hits += (rel.sum(dim=1) > 0).sum().float()
        self.users += target.shape[0]

    def compute(self):
        return self.hits / self.users if self.users > 0 else torch.tensor(0.0)

    def reset(self):
        self.hits.zero_()
        self.users.zero_()


@metric_registry.register("nDCG")
class nDCG(TopKMetric):
    """The nDCG@k metric is defined as the rapport of the DCG@k and the IDCG@k.

    The DCG@k represent the Discounted Cumulative Gain,
        which measures the gain of the items retrieved.

    The IDCG@k represent the Ideal Discounted Cumulative Gain,
        which measures the maximum gain possible
        obtainable by a perfect model.

    Attributes:
        ndcg (Tensor): The total value of ndcg per user.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    ndcg: Tensor
    users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def dcg(self, rel: Tensor) -> Tensor:
        """The Discounted Cumulative Gain definition.

        Args:
            rel (Tensor): The relevance tensor.

        Returns:
            Tensor: The discounted tensor.
        """
        return (
            rel / torch.log2(torch.arange(2, rel.size(1) + 2, device=rel.device))
        ).sum(dim=1)

    def update(self, preds: Tensor, target: Tensor):
        target = torch.where(target > 0, 2 ** (target + 1) - 1, target)
        top_k = torch.topk(preds, self.k, dim=1, largest=True, sorted=True).indices
        rel = torch.gather(target, 1, top_k).float()
        ideal_rel = torch.topk(target, self.k, dim=1, largest=True, sorted=True).values

        dcg_score = self.dcg(rel)
        idcg_score = self.dcg(ideal_rel).clamp(min=1e-10)

        self.ndcg += (dcg_score / idcg_score).nan_to_num(0).sum()
        self.users += target.shape[0]

    def compute(self):
        return self.ndcg / self.users if self.users > 0 else torch.tensor(0.0)

    def reset(self):
        self.ndcg.zero_()
        self.users.zero_()


@metric_registry.register("UserCoverage")
class UserCoverage(TopKMetric):
    """The UserCoverage@k metric counts the number of users
       that received at least one recommendation.

    Attributes:
        covered_users (Tensor): The number of users with at least one recommendation.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    covered_users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("covered_users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        top_k = torch.topk(preds, self.k, dim=1).indices
        self.covered_users += top_k.shape[0]

    def compute(self):
        """Computes the final metric value."""
        return self.covered_users

    def reset(self):
        """Resets the metric state."""
        self.covered_users.zero_()


@metric_registry.register("ItemCoverage")
class ItemCoverage(TopKMetric):
    """The ItemCoverage@k metric counts the number of unique items
       that were recommended across all users.

    Attributes:
        unique_items (list): The list of unique items per batch.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    unique_items: list

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("unique_items", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        top_k = torch.topk(preds, self.k, dim=1).indices
        self.unique_items.append(top_k.detach().cpu())

    def compute(self):
        """Computes the final metric value."""
        if len(self.unique_items) == 0:
            return torch.tensor(0.0)
        all_items_tensor = torch.cat(self.unique_items, dim=0)
        unique_items: Tensor = torch.unique(all_items_tensor)
        return torch.tensor(unique_items.numel())

    def reset(self):
        """Resets the metric state."""
        self.unique_items = []


@metric_registry.register("Gini")
class Gini(TopKMetric):
    """The Gini index metric measures the inequality in the distribution of recommended items,
    computed on a per-user basis and averaged over users. This implementation accounts
    for items that were never recommended by applying an offset.

    The Gini coefficient is computed as:
        Gini = 1 - (sum_{j=1}^{n_rec} (2*(j + offset) - num_items - 1) * (count_j / free_norm)) / (num_items - 1)
    where:
        - n_rec is the number of items that were recommended at least once,
        - offset = num_items - n_rec (to account for items with zero recommendations),
        - count_j is the recommendation count for the j-th item in ascending order,
        - free_norm is the total number of recommendations made (i.e., sum over users).

    Attributes:
        recommended_items (list): List of tensors containing recommended item indices.
        free_norm (Tensor): Total number of recommendations made (accumulated per user).
        num_items (int): Total number of items in the catalog, inferred from the prediction tensor.

    Args:
        k (int): The cutoff for recommendations.
        dist_sync_on_step (bool): Torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    recommended_items: list
    free_norm: Tensor
    num_items: int

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        # Accumulate recommended indices from each update call.
        self.add_state("recommended_items", default=[], dist_reduce_fx=None)
        # Accumulate the total number of recommendations given (free_norm).
        self.add_state("free_norm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.num_items = None

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        if self.num_items is None:
            self.num_items = preds.shape[1]

        top_k = torch.topk(preds, self.k, dim=1).indices
        batch_size = top_k.shape[0]
        self.free_norm += torch.tensor(batch_size * self.k, dtype=torch.float)
        self.recommended_items.append(top_k.detach().cpu())

    def compute(self):
        """Computes the final metric value."""
        if (
            len(self.recommended_items) == 0
            or self.num_items is None
            or self.free_norm == 0
        ):
            return torch.tensor(0.0)

        all_items = torch.cat(self.recommended_items, dim=0).view(-1)
        unique, counts = torch.unique(all_items, return_counts=True)
        n_rec_items = unique.numel()
        sorted_counts, _ = torch.sort(counts.float())
        # Offset to account for items never recommended.
        offset = self.num_items - n_rec_items
        j = torch.arange(n_rec_items, dtype=sorted_counts.dtype)
        contributions = (2 * (j + offset + 1) - self.num_items - 1) * (
            sorted_counts / self.free_norm
        )
        # Sum contributions and normalize.
        gini = 1 - torch.sum(contributions) / (self.num_items - 1)
        return gini

    def reset(self):
        """Reset the metric state."""
        self.recommended_items = []
        self.free_norm = torch.tensor(0.0)
        self.num_items = None


@metric_registry.register("MAE")
class MAE(BaseMetric):
    """
    Mean Absolute Error (MAE) metric.

    This metric computes the average absolute difference between the predictions and targets.

    Attributes:
        sum_absolute_errors (Tensor): Sum of absolute errors across all batches.
        total_count (Tensor): Total number of elements processed.

    Args:
        dist_sync_on_step (bool): Torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    sum_absolute_errors: Tensor
    total_count: Tensor

    def __init__(self, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "sum_absolute_errors", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        mask = target > 0
        abs_errors = torch.abs(preds[mask] - target[mask])
        self.sum_absolute_errors += abs_errors.sum()
        self.total_count += target.numel()

    def compute(self):
        """Computes the final metric value."""
        return (
            self.sum_absolute_errors / self.total_count
            if self.total_count > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        """Reset the metric state."""
        self.sum_absolute_errors.zero_()
        self.total_count.zero_()


@metric_registry.register("MSE")
class MSE(BaseMetric):
    """
    Mean Squared Error (MSE) metric.

    This metric computes the average squared difference between the predictions and targets.

    Attributes:
        sum_squared_errors (Tensor): Sum of squared errors across all batches.
        total_count (Tensor): Total number of elements processed.

    Args:
        dist_sync_on_step (bool): Torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    sum_squared_errors: Tensor
    total_count: Tensor

    def __init__(self, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "sum_squared_errors", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        mask = target > 0
        squared_errors = (preds[mask] - target[mask]) ** 2
        self.sum_squared_errors += squared_errors.sum()
        self.total_count += target.numel()

    def compute(self):
        """Computes the final metric value."""
        return (
            self.sum_squared_errors / self.total_count
            if self.total_count > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        """Reset the metric state."""
        self.sum_squared_errors.zero_()
        self.total_count.zero_()


@metric_registry.register("RMSE")
class RMSE(BaseMetric):
    """
    Root Mean Squared Error (RMSE) metric.

    This metric computes the square root of the average squared difference between the predictions and targets.

    Attributes:
        sum_squared_errors (Tensor): Sum of squared errors across all batches.
        total_count (Tensor): Total number of elements processed.

    Args:
        dist_sync_on_step (bool): Torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    sum_squared_errors: Tensor
    total_count: Tensor

    def __init__(self, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "sum_squared_errors", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        mask = target > 0
        squared_errors = (preds[mask] - target[mask]) ** 2
        self.sum_squared_errors += squared_errors.sum()
        self.total_count += target.numel()

    def compute(self):
        """Computes the final metric value."""
        mse = (
            self.sum_squared_errors / self.total_count
            if self.total_count > 0
            else torch.tensor(0.0)
        )
        return torch.sqrt(mse)

    def reset(self):
        """Reset the metric state."""
        self.sum_squared_errors.zero_()
        self.total_count.zero_()
