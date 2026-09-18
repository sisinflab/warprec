"""Every registered model must build, train a step, predict and be evaluated.

This suite is deliberately shallow and wide: it makes no claim about the quality
of a model, only that the whole path from configuration to a score matrix holds
together for all of them. Most of the defects this framework has shipped were of
exactly that shape, and none of them needed a deep test to catch.
"""

from typing import Any, Dict

import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import (
    ContextRecommenderUtils,
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.utils.registry import model_registry, params_registry

# Values that keep every model small and fast while staying schema-valid. Names
# are matched first, then the annotation, so a new hyperparameter only needs an
# entry here when its name carries meaning the type cannot express.
BY_NAME: Dict[str, Any] = {
    "epochs": 1,
    "batch_size": 32,
    "batch_size_kd": 32,
    "teacher_epochs": 1,
    "embedding_size": 8,
    "mf_embedding_size": 8,
    "mlp_embedding_size": 8,
    "hidden_size": 8,
    "inner_size": 8,
    "latent_dim": 8,
    "intermediate_dim": 8,
    "attention_size": 8,
    "n_dims": 8,
    "factors": 8,
    "n_factors": 4,
    "k_fac": 2,
    "k_interests": 2,
    "n_layers": 1,
    "num_layers": 1,
    "n_teacher_layers": 1,
    "n_student_layers": 1,
    "n_heads": 2,
    "n_h": 2,
    "n_v": 2,
    "n_iterations": 1,
    "n_ode_steps": 1,
    "max_seq_len": 5,
    "order_len": 2,
    "k": 5,
    "ii_k": 5,
    "neg_samples": 1,
    "mn_ratio": 1,
    "split_to": 1,
    "expert_num": 2,
    "low_rank": 4,
    "cross_layer_num": 1,
    "layer_cl": 1,
    "it": 1,
    "mlp_hidden_size": [8],
    "encoder_hidden_dims": [8],
    "user_mlp_hidden": 8,
    "item_mlp_hidden": 8,
    "weight_size": [8],
    "cin_layer_size": [8],
    "cnn_channels": [2],
    "cnn_kernels": [2],
    "cnn_strides": [1],
    "similarity": "cosine",
    "sim_type": "cos",
    "user_profile": "binary",
    "item_profile": "binary",
    "model_structure": "stacked",
    "hid_activation": "relu",
    "out_activation": "relu",
    "loss_type": "BCE",
    "mode": "parallel",
    "aug_type": "ED",
    "ssl_type": "us",
    "dnn_type": "trm",
    "confidence_type": "linear",
    "target_density": 0.5,
    "pop_ratio": 0.8,
    "mask_prob": 0.2,
    "corruption": 0.1,
    "anneal_cap": 0.2,
    "total_anneal_steps": 10,
    "anneal_step": 10,
    "learning_rate": 0.01,
}
DEFAULT_BY_TYPE = {"int": 2, "float": 0.1, "bool": True, "str": "cosine", "list": [8]}
SKIP_FIELDS = {"meta", "optimization", "early_stopping"}


def _value_for(field_name: str, annotation: Any) -> Any:
    """Pick a small, schema-valid value for one hyperparameter.

    Args:
        field_name (str): The name of the hyperparameter.
        annotation (Any): Its type annotation.

    Returns:
        Any: The value to use in the smoke test.
    """
    if field_name in BY_NAME:
        return BY_NAME[field_name]
    text = str(annotation)
    if "List[int]" in text or "List[List" in text:
        return DEFAULT_BY_TYPE["list"]
    if "bool" in text:
        return DEFAULT_BY_TYPE["bool"]
    if "float" in text:
        return DEFAULT_BY_TYPE["float"]
    if "int" in text:
        return DEFAULT_BY_TYPE["int"]
    return DEFAULT_BY_TYPE["str"]


def build_params(model_name: str) -> Dict[str, Any]:
    """Build a complete, schema-valid parameter set for a model.

    Args:
        model_name (str): The registered name of the model.

    Returns:
        Dict[str, Any]: The hyperparameters to instantiate it with.
    """
    schema = params_registry.get_class(model_name)
    return {
        name: _value_for(name, field.annotation)
        for name, field in schema.model_fields.items()
        if name not in SKIP_FIELDS
    }


MODELS = sorted(
    name
    for name in model_registry.list_registered()
    # ProxyRecommender replays a file produced elsewhere; it has no hyperparameters
    # and is covered by the evaluation tests instead.
    if name != "PROXYRECOMMENDER"
)


@pytest.mark.parametrize("model_name", MODELS)
def test_model_builds_trains_and_predicts(model_name: str, dataset: Dataset):
    """A model must construct, take a training step and score every item."""
    params = build_params(model_name)

    # The schema is the contract the configuration file is validated against, so
    # a parameter set that fails here would fail for a user too.
    params_registry.get(model_name, **params)

    model = model_registry.get(
        model_name,
        params=params,
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
    )

    if isinstance(model, IterativeRecommender):
        loader = model.get_dataloader(
            interactions=dataset.train_set, sessions=dataset.train_session
        )
        batch = next(iter(loader))

        # Some models set per-epoch state (KL annealing, for instance) in the
        # Lightning hook that precedes the first step, so the loop is emulated
        # rather than short-circuited.
        model.on_train_epoch_start()

        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss).all(), f"{model_name}: non-finite training loss"

    model.eval()
    users = torch.arange(min(4, dataset.info()["n_users"]))
    predict_kwargs: Dict[str, Any] = {"user_indices": users}
    if isinstance(model, SequentialRecommenderUtils):
        history, lengths, _ = dataset.train_set.get_history()
        predict_kwargs["user_seq"] = history[users][:, -model.max_seq_len :]
        predict_kwargs["seq_len"] = lengths[users].clamp(max=model.max_seq_len)
    if isinstance(model, ContextRecommenderUtils) and model.context_dims:
        _, _, _, contexts = dataset.train_transactions.get_arrays()
        predict_kwargs["contexts"] = torch.from_numpy(contexts[: len(users)])

    with torch.inference_mode():
        scores = model.predict(**predict_kwargs)

    assert scores.shape[0] == len(users), f"{model_name}: wrong number of rows"
    assert scores.shape[1] == dataset.info()["n_items"], f"{model_name}: wrong width"
    assert torch.isfinite(scores).any(), f"{model_name}: no finite score produced"
