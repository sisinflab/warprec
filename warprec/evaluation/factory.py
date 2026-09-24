from typing import TYPE_CHECKING, Optional

from warprec.data import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.evaluation.propensity import build_propensity
from warprec.utils.config import EvaluationConfig

if TYPE_CHECKING:
    from warprec.recommenders.reranking.base import Reranker


def build_evaluator(
    evaluation: EvaluationConfig,
    dataset: Dataset,
    reranker: Optional["Reranker"] = None,
) -> Evaluator:
    """Build the evaluator every pipeline uses for its main dataset.

    Args:
        evaluation (EvaluationConfig): The evaluation section of the configuration.
        dataset (Dataset): The dataset the models are evaluated on.
        reranker (Optional[Reranker]): The re-ranker applied to every list, so that
            what a run reports is what it would write out.

    Returns:
        Evaluator: The evaluator bound to that dataset.
    """
    train_sparse = dataset.train_set.get_sparse()

    return Evaluator(
        list(evaluation.metrics),
        list(evaluation.top_k),
        train_set=train_sparse,
        additional_data=dataset.get_stash(),
        complex_metrics=evaluation.complex_metrics,
        feature_lookup=dataset.get_feature_matrix(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
        mask_seen=evaluation.mask_seen,
        candidates=evaluation.candidates,
        reranker=reranker,
        propensity=build_propensity(
            train_sparse.getnnz(axis=0),
            estimator=evaluation.propensity.estimator,
            power=evaluation.propensity.power,
            clip=evaluation.propensity.clip,
        ),
    )
