"""Shared fixtures for the test suite.

Every fixture here is generated in memory. The datasets live outside the
repository, so a test that reads one would pass locally and fail in CI.
"""

from typing import Any, Dict, List, Tuple

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
def knowledge_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A small knowledge graph over the catalogue, and its alignment.

    Every item is given an entity so that the knowledge-aware models have
    something to read, and the graph reaches a second hop beyond the items so
    that a propagating model has somewhere to propagate to.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The triples and the alignment.
    """
    rng = np.random.default_rng(13)

    # Entities 0..N_ITEMS-1 stand for the items; the rest are attributes only
    # the graph knows about.
    attributes = np.arange(N_ITEMS, N_ITEMS + 12)
    heads, relations, tails = [], [], []
    for item in range(N_ITEMS):
        for tail in rng.choice(attributes, 2, replace=False):
            heads.append(item)
            relations.append(int(rng.integers(0, 3)))
            tails.append(int(tail))

    # A few facts between attributes, so the graph is not only one hop deep.
    for _ in range(8):
        pair = rng.choice(attributes, 2, replace=False)
        heads.append(int(pair[0]))
        relations.append(3)
        tails.append(int(pair[1]))

    triples = pd.DataFrame({"head": heads, "relation": relations, "tail": tails})
    links = pd.DataFrame(
        {"item_id": np.arange(N_ITEMS), "entity_id": np.arange(N_ITEMS)}
    )
    return triples, links


@pytest.fixture(scope="session")
def multimodal_frames() -> Dict[str, Dict[str, Any]]:
    """Two modalities of differing width over most of the catalogue.

    One item is deliberately left out of each so that the padding path is
    exercised by every model that reads features, and the two widths differ
    because a model that assumed one width would pass an equal-width fixture.

    Returns:
        Dict[str, Dict[str, Any]]: The payload the dataset reads modalities from.
    """
    rng = np.random.default_rng(21)
    return {
        "visual": {
            "features": rng.normal(size=(N_ITEMS - 1, 12)).astype("float32"),
            "items": np.arange(N_ITEMS - 1),
            "normalize": "none",
        },
        "textual": {
            "features": rng.normal(size=(N_ITEMS - 2, 7)).astype("float32"),
            "items": np.arange(N_ITEMS - 2),
            "normalize": "l2",
        },
    }


@pytest.fixture(scope="session")
def dataset(
    interactions_frame: pd.DataFrame,
    side_frame: pd.DataFrame,
    knowledge_frames: Tuple[pd.DataFrame, pd.DataFrame],
    multimodal_frames: Dict[str, Dict[str, Any]],
) -> Dataset:
    """A Dataset carrying everything the model families need at once.

    Contexts and side information are both present so that a single fixture
    serves the collaborative, content, context-aware, sequential and hybrid
    families without special-casing.

    Args:
        interactions_frame (pd.DataFrame): The generated interactions.
        side_frame (pd.DataFrame): The generated item features.
        knowledge_frames (Tuple[pd.DataFrame, pd.DataFrame]): The triples and
            the item alignment.
        multimodal_frames (Dict[str, Dict[str, Any]]): The item features, one
            entry per modality.

    Returns:
        Dataset: The dataset under test.
    """
    triples, links = knowledge_frames
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
        knowledge_data=triples,
        knowledge_links=links,
        multimodal_data=multimodal_frames,
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
    "feature_size": 8,
    "knn_k": 3,
    "n_ui_layers": 1,
    # Left unset so the models read every configured modality, which is the
    # path a user gets by default.
    "modalities": None,
    "modality_weights": None,
    "neighbour_size": 3,
    "n_iter": 1,
    "n_hop": 2,
    "n_memory": 4,
    "n_hops": 1,
    "aggregator": "sum",
    "independence": "cosine",
    "lambda_coeff": 0.5,
    "temperature": 0.2,
    "cl_weight": 0.1,
    "kg_weight": 0.01,
    "ind_weight": 0.01,
    "steps": 4,
    "time_size": 8,
    "noise_min": 0.001,
    "noise_max": 0.01,
    # Start the reverse walk from the real history, and keep it deterministic:
    # resampling noise at inference makes two identical runs disagree, which is
    # the trap MultiVAE's posterior sampling was in 1.9.0.
    "sampling_steps": 0,
    "sampling_noise": False,
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
