"""Behavioural tests for building datasets from a configuration.

These go through initialize_datasets, the call every pipeline makes, because the
defects they guard sat in how the configuration is turned into reader calls and
datasets rather than in the reader or the Dataset on their own.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from warprec.common.initialize import initialize_datasets
from warprec.data.reader import ReaderFactory
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config.config import load_train_configuration

N_USERS, N_ITEMS = 12, 20


def transactions(seed: int) -> pd.DataFrame:
    """Every user interacts with eight distinct items.

    Args:
        seed (int): Seeds the item draw.

    Returns:
        pd.DataFrame: The generated transactions.
    """
    rng = np.random.default_rng(seed)
    rows = [
        (user, int(item))
        for user in range(N_USERS)
        for item in rng.choice(N_ITEMS, 8, replace=False)
    ]
    frame = pd.DataFrame(rows, columns=["user_id", "item_id"])
    frame["rating"] = 1.0
    frame["timestamp"] = np.arange(len(frame))
    return frame


def build(tmp_path: Path, reader: Dict[str, Any], **sections: Any) -> Tuple[Any, ...]:
    """Validate a training configuration and build its datasets.

    Args:
        tmp_path (Path): Where the configuration and outputs are written.
        reader (Dict[str, Any]): The reader section.
        **sections (Any): Any other top-level section.

    Returns:
        Tuple[Any, ...]: The main dataset, the validation dataset and the folds.
    """
    config = {
        "reader": {
            "data_type": "transaction",
            "reading_method": "local",
            "rating_type": "implicit",
            **reader,
        },
        "writer": {
            "dataset_name": "init",
            "writing_method": "local",
            "local_experiment_path": str(tmp_path / "out"),
        },
        "models": {"Pop": {}},
        "evaluation": {"top_k": [5], "metrics": ["nDCG"]},
        **sections,
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    loaded = load_train_configuration(str(path))
    return initialize_datasets(
        ReaderFactory.get_reader(loaded), WarpRecCallback(), loaded
    )


def test_cluster_files_are_read_and_mapped(tmp_path: Path):
    """Each user and item keeps the cluster its file assigns it."""
    data = tmp_path / "data.tsv"
    transactions(1).to_csv(data, sep="\t", index=False)
    user_clusters = pd.DataFrame(
        {"user_id": range(N_USERS), "cluster": [u % 2 + 1 for u in range(N_USERS)]}
    )
    item_clusters = pd.DataFrame(
        {"item_id": range(N_ITEMS), "cluster": [i % 3 + 1 for i in range(N_ITEMS)]}
    )
    user_clusters.to_csv(tmp_path / "users.tsv", sep="\t", index=False)
    item_clusters.to_csv(tmp_path / "items.tsv", sep="\t", index=False)

    main, _, _ = build(
        tmp_path,
        {
            "loading_strategy": "dataset",
            "local_path": str(data),
            "clustering": {
                "user_local_path": str(tmp_path / "users.tsv"),
                "item_local_path": str(tmp_path / "items.tsv"),
            },
        },
        splitter={"test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2}},
    )

    user_map, item_map = main.get_mappings()
    user_cluster = main.get_user_cluster()
    item_cluster = main.get_item_cluster()
    assert len(set(user_cluster.tolist())) == 2
    assert len(set(item_cluster[: len(item_map)].tolist())) == 3
    for raw, index in user_map.items():
        assert user_cluster[index] == user_cluster[user_map[raw % 2]]
    for raw, index in item_map.items():
        assert item_cluster[index] == item_cluster[item_map[raw % 3]]


def test_pre_split_folds_are_loaded_as_train_and_validation_pairs(tmp_path: Path):
    """Each fold directory becomes its own dataset, aligned on its own train set."""
    split = tmp_path / "split"
    folds = 3
    frame = transactions(2)
    test = frame.groupby("user_id").tail(1)
    train = frame.drop(test.index)
    split.mkdir()
    train.to_csv(split / "train.tsv", sep="\t", index=False)
    test.to_csv(split / "test.tsv", sep="\t", index=False)
    fold_of = train.groupby("user_id").cumcount() % folds
    for fold in range(folds):
        directory = split / str(fold + 1)
        directory.mkdir()
        train[fold_of != fold].to_csv(directory / "train.tsv", sep="\t", index=False)
        train[fold_of == fold].to_csv(
            directory / "validation.tsv", sep="\t", index=False
        )

    main, validation, fold_datasets = build(
        tmp_path,
        {"loading_strategy": "split", "split": {"local_path": str(split)}},
    )

    assert validation is None
    assert len(fold_datasets) == folds
    assert main.train_set.get_sparse().nnz == len(train)
    for fold, dataset in enumerate(fold_datasets):
        assert dataset.train_set.get_sparse().nnz == int((fold_of != fold).sum())
        assert dataset.eval_set.get_sparse().nnz > 0
