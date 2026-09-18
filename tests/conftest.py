"""Shared fixtures for the test suite.

Every fixture here is generated in memory. The datasets live outside the
repository, so a test that reads one would pass locally and fail in CI.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from warprec.data.dataset import Dataset
from warprec.utils.registry import params_registry

N_USERS = 40
N_ITEMS = 25
N_INTERACTIONS = 600
CONTEXT_LABELS: List[str] = ["daytime", "weather"]


@pytest.fixture(scope="session")
def interactions_frame() -> pd.DataFrame:
    """A small, dense interaction frame with ratings, timestamps and contexts.

    Returns:
        pd.DataFrame: The generated interactions.
    """
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "user_id": rng.integers(0, N_USERS, N_INTERACTIONS),
            "item_id": rng.integers(0, N_ITEMS, N_INTERACTIONS),
            "rating": rng.integers(1, 6, N_INTERACTIONS).astype(float),
            "timestamp": rng.integers(1_000_000, 2_000_000, N_INTERACTIONS),
            "daytime": rng.choice(["morning", "evening"], N_INTERACTIONS),
            "weather": rng.choice(["sunny", "rainy"], N_INTERACTIONS),
        }
    ).drop_duplicates(subset=["user_id", "item_id"])

    # Every user needs at least two interactions so that a split leaves a
    # training history behind, and every item needs to be reachable.
    filler = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(N_USERS), 2),
            "item_id": np.tile(np.arange(N_ITEMS), 4)[: N_USERS * 2],
            "rating": 4.0,
            "timestamp": 1_500_000,
            "daytime": "morning",
            "weather": "sunny",
        }
    )
    frame = pd.concat([frame, filler]).drop_duplicates(subset=["user_id", "item_id"])
    return frame.reset_index(drop=True)


@pytest.fixture(scope="session")
def side_frame() -> pd.DataFrame:
    """Item attributes in the wide, one-hot layout the content models expect.

    Returns:
        pd.DataFrame: The generated item features.
    """
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "item_id": np.arange(N_ITEMS),
            "action": rng.integers(0, 2, N_ITEMS),
            "comedy": rng.integers(0, 2, N_ITEMS),
            "drama": rng.integers(0, 2, N_ITEMS),
        }
    )


@pytest.fixture(scope="session")
def dataset(interactions_frame: pd.DataFrame, side_frame: pd.DataFrame) -> Dataset:
    """A Dataset carrying everything the model families need at once.

    Contexts and side information are both present so that a single fixture
    serves the collaborative, content, context-aware, sequential and hybrid
    families without special-casing.

    Args:
        interactions_frame (pd.DataFrame): The generated interactions.
        side_frame (pd.DataFrame): The generated item features.

    Returns:
        Dataset: The dataset under test.
    """
    train = interactions_frame.groupby("user_id", group_keys=False).apply(
        lambda g: g.iloc[:-1]
    )
    evaluation = interactions_frame.groupby("user_id", group_keys=False).apply(
        lambda g: g.iloc[-1:]
    )
    return Dataset(
        train_data=train,
        eval_data=evaluation,
        side_data=side_frame,
        rating_type="explicit",
        rating_label="rating",
        timestamp_label="timestamp",
        context_labels=CONTEXT_LABELS,
        batch_size=64,
    )


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
