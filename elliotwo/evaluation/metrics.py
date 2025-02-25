from typing import Tuple
from abc import abstractmethod, ABC

import torch
from torch import Tensor
from elliotwo.utils.config import Configuration
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.abstract_recommender import AbstractRecommender
from elliotwo.utils.registry import metric_registry


class AbstractMetric(ABC):
    """The abstract definition of a metric. Every metric should extend this class.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def __init__(self, config: Configuration):
        self.config = config
        self.device = torch.device(self.config.general.device)

    @abstractmethod
    def eval(
        self, model: AbstractRecommender, dataset: Interactions, top_k: int
    ) -> float:
        """The evaluation method of a metric should return a float value.

        Args:
            model (AbstractRecommender): The trained model to evaluate.
            dataset (Interactions): The partition of data to use as ground truth.
            top_k (int): The k most relevant interactions to take into account.

        Returns:
            float: The calculated metric.
        """

    def get_name(self) -> str:
        """This method returns the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return self.__class__.__name__


class TopKMetric(AbstractMetric):
    """TopKMetric is the interface to be used by ranking metrics.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def data_preparation(
        self, model: AbstractRecommender, dataset: Interactions, top_k: int
    ) -> Tuple[Tensor, Tensor]:
        """The data preparation of a TopKMetric starts with the definition of the ranked relevance.

        The different implementation of a TopKMetric should define their \
            proper way of calculating ground truth.

        Args:
            model (AbstractRecommender): The trained model to evaluate.
            dataset (Interactions): The partition of data to use as ground truth.
            top_k (int): The k most relevant interactions to take into account.

        Returns:
            Tuple[Tensor, Tensor]:
                Tensor: The ranked relevance.
                Tensor: The ground truth.
        """
        # Retrieve and apply masking to the prediction data
        predictions = model.forward().to(self.device)

        # Set ground truth and top k values
        gt = self.ground_truth(dataset)

        # Retrieve indices of top k items
        top_k_items = torch.topk(
            predictions, top_k, dim=1, largest=True, sorted=False
        ).indices

        # Retrieve top k predictions
        ranked_predictions = predictions.gather(1, top_k_items)

        # Retrieve top k relevant items
        ranked_relevances = gt.gather(1, top_k_items)

        # We mask out of ranked_relevances all possible invalid indexes
        ranked_relevances[ranked_predictions == -torch.inf] = 0

        return ranked_relevances, gt

    @abstractmethod
    def ground_truth(self, dataset: Interactions) -> Tensor:
        """This method should return the ground truth for the metric calculation.

        Args:
            dataset (Interactions): The partition of data to use as ground truth.

        Returns:
            Tensor: The ground truth.
        """


class BinaryMetric(TopKMetric):
    """A Binary TopKMetric defines a relevant item either as 1 or 0, \
        so wether it was or not present in the user interaction list.
    A metric that is defined as binary will not take into account how \
        important was the item but rather focus on the quantity of
    retrieved items.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def ground_truth(self, dataset: Interactions) -> Tensor:
        """This method extract ground truth and binarize it for metric calculation.

        Args:
            dataset (Interactions): The partition of data to use as ground truth.

        Returns:
            Tensor: The binarized ground truth.
        """
        gt = torch.tensor(
            dataset.get_sparse().todense(),
            dtype=self.config.precision_torch(),
            device=self.device,
        )
        _gt_mask = gt != 0
        gt[_gt_mask] = 1
        return gt


class DiscountedMetric(TopKMetric):
    """A Discounted TopKMetric defines a relevant item with its proper score.

    In this implementation the discounted relevance is calculated as: 2^(score + 1) - 1.

    This ensures to penalize relevant items that the system did not retrieve properly.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def data_preparation(
        self, model: AbstractRecommender, dataset: Interactions, top_k: int
    ) -> Tuple[Tensor, Tensor]:
        ranked_relevances, gt = super().data_preparation(model, dataset, top_k)

        # We precompure the discounted relevance score
        ranked_relevances = torch.where(
            ranked_relevances != 0,
            2 ** (ranked_relevances + 1) - 1,
            ranked_relevances,
        )
        return ranked_relevances, gt

    def ground_truth(self, dataset: Interactions) -> Tensor:
        """This method extracts the ground truth and uses it as it is, without binarization.

        Args:
            dataset (Interactions): The partition of data to use as ground truth.

        Returns:
            Tensor: The ground truth.
        """
        return torch.tensor(
            dataset.get_sparse().todense(),
            dtype=self.config.precision_torch(),
            device=self.device,
        )


@metric_registry.register("nDCG")
class NDCG(DiscountedMetric):
    """Implementation of nDCG@k metric.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def eval(
        self, model: AbstractRecommender, dataset: Interactions, top_k: int
    ) -> float:
        """The nDCG@k metric is defined as the rapport of the DCG@k and the IDCG@k.
        The DCG@k represent the Discounted Cumulative Gain, \
            wich measures the gain of the items retrieved.
        The IDCG@k represent the Ideal Discounted Cumulative Gain, \
            wich measures the maximum gain possible
        obtainable by a perfect model.

        Args:
            model (AbstractRecommender): The trained model to evaluate.
            dataset (Interactions): The partition of data to use as ground truth.
            top_k (int): The k most relevant interactions to take into account.

        Returns:
            float: The calculated nDCG@k metric.
        """
        ranked_relevances, gt = super().data_preparation(model, dataset, top_k)

        # Precompute the denominator for DCG and IDCG (log2 scaling)
        discount = torch.log2(
            torch.arange(
                2,
                top_k + 2,
                device=self.device,
                dtype=self.config.precision_torch(),
            )
        )

        # Compute DCG@k for all valid users (using discounted relevance)
        dcg = (ranked_relevances[:, :top_k] / discount).sum(dim=1)

        # Compute IDCG@k for all valid users (using the ideal ranking)
        ideal_relevances = torch.topk(
            gt, top_k, dim=1, largest=True, sorted=True
        ).values
        ideal_discounted_relevance = torch.where(
            ideal_relevances != 0, 2 ** (ideal_relevances + 1) - 1, ideal_relevances
        )
        idcg = (ideal_discounted_relevance / discount).sum(dim=1)

        # Using torch.where we skip all possible zero divisions
        ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))

        return ndcg.mean().item()


@metric_registry.register("HitRate")
class HitRate(BinaryMetric):
    """Implementation of the HitRate@k metric.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def eval(
        self, model: AbstractRecommender, dataset: Interactions, top_k: int
    ) -> float:
        """The HitRate@k metric counts the number of users for wich \
            the model retrieved at least one item.
        This is normalized by the total number of users.

        Args:
            model (AbstractRecommender): The trained model to evaluate.
            dataset (Interactions): The partition of data to use as ground truth.
            top_k (int): The k most relevant interactions to take into account.

        Returns:
            float: The calculated HitRate@k metric.
        """
        ranked_relevances, _ = super().data_preparation(model, dataset, top_k)
        return (ranked_relevances.sum(dim=1) > 0).float().mean().item()


@metric_registry.register("Precision")
class Precision(BinaryMetric):
    """Implementation of the Precision metric. More information can \
        be found here: https://en.wikipedia.org/wiki/Precision_and_recall.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def eval(
        self, model: AbstractRecommender, dataset: Interactions, top_k: int
    ) -> float:
        """The Precision@k counts the number of item retrieved correctly, \
            over the maximum number of possible retrieve items.

        Args:
            model (AbstractRecommender): The trained model to evaluate.
            dataset (Interactions): The partition of data to use as ground truth.
            top_k (int): The k most relevant interactions to take into account.

        Returns:
            float: The calculated Precision@k metric.
        """
        ranked_relevances, _ = super().data_preparation(model, dataset, top_k)
        return ranked_relevances.mean().item()


@metric_registry.register("Recall")
class Recall(BinaryMetric):
    """Implementation of the Recall metric. More information can \
        be found here: https://en.wikipedia.org/wiki/Precision_and_recall.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def eval(
        self, model: AbstractRecommender, dataset: Interactions, top_k: int
    ) -> float:
        """The Recall@k counts the number of item retrieve correctly, \
            over the total number of relevant item in the ground truth.

        Args:
            model (AbstractRecommender): The trained model to evaluate.
            dataset (Interactions): The partition of data to use as ground truth.
            top_k (int): The k most relevant interactions to take into account.

        Returns:
            float: The calculated Recall@k metric.
        """
        ranked_relevances, gt = super().data_preparation(model, dataset, top_k)
        return (ranked_relevances.sum() / gt.sum()).item()
