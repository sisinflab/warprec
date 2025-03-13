# pylint: disable=arguments-differ
from abc import abstractmethod, ABC

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

    def __init__(self, k: int, dist_sync_on_step=False):
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
    """

    correct: Tensor
    users: Tensor

    def __init__(self, k: int, dist_sync_on_step: bool = False):
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
    """

    correct: Tensor
    real_correct: Tensor

    def __init__(self, k: int, dist_sync_on_step: bool = False):
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
    """

    hits: Tensor
    users: Tensor

    def __init__(self, k: int, dist_sync_on_step: bool = False):
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
    """

    ndcg: Tensor
    users: Tensor

    def __init__(self, k: int, dist_sync_on_step: bool = False):
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
    """

    covered_users: Tensor

    def __init__(self, k: int, dist_sync_on_step: bool = False):
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
    """

    unique_items: list

    def __init__(self, k: int, dist_sync_on_step: bool = False):
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
