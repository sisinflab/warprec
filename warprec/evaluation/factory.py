from warprec.data import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.utils.config import EvaluationConfig


def build_evaluator(evaluation: EvaluationConfig, dataset: Dataset) -> Evaluator:
    """Build the evaluator every pipeline uses for its main dataset.

    Args:
        evaluation (EvaluationConfig): The evaluation section of the configuration.
        dataset (Dataset): The dataset the models are evaluated on.

    Returns:
        Evaluator: The evaluator bound to that dataset.
    """
    return Evaluator(
        list(evaluation.metrics),
        list(evaluation.top_k),
        train_set=dataset.train_set.get_sparse(),
        additional_data=dataset.get_stash(),
        complex_metrics=evaluation.complex_metrics,
        feature_lookup=dataset.get_features_lookup(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
        mask_seen=evaluation.mask_seen,
    )
