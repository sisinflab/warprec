import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from warprec.common import dataset_preparation, initialize_datasets
from warprec.data import Dataset
from warprec.data.reader import ReaderFactory
from warprec.data.timing import (
    BUILDING_COLUMN,
    STEP_COLUMNS,
    STRUCTURE_COLUMNS,
    DataPreparationTimes,
)
from warprec.pipelines.common import model_timing_row
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config.config import load_train_configuration

N_USERS, N_ITEMS = 30, 40


def write_inputs(tmp_path: Path) -> Dict[str, str]:
    """Write an interaction file and a side information file.

    Args:
        tmp_path (Path): Where the files are written.

    Returns:
        Dict[str, str]: The path of each file, by name.
    """
    rng = np.random.default_rng(0)
    rows = [
        (user, int(item), float(rng.integers(1, 6)))
        for user in range(N_USERS)
        for item in rng.choice(N_ITEMS, 12, replace=False)
    ]
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
    frame["timestamp"] = np.arange(len(frame))
    side = pd.DataFrame(
        {
            "item_id": range(N_ITEMS),
            "genre": rng.integers(0, 4, N_ITEMS),
            "decade": rng.integers(0, 3, N_ITEMS),
        }
    )
    paths = {"data": tmp_path / "data.tsv", "side": tmp_path / "side.tsv"}
    frame.to_csv(paths["data"], sep="\t", index=False)
    side.to_csv(paths["side"], sep="\t", index=False)
    return {name: str(path) for name, path in paths.items()}


def load_config(tmp_path: Path, **sections: Any) -> Any:
    """Validate a training configuration reading the generated files.

    Args:
        tmp_path (Path): Where the inputs, configuration and outputs are written.
        **sections (Any): Top-level sections added to the configuration.

    Returns:
        Any: The loaded training configuration.
    """
    paths = write_inputs(tmp_path)
    config = {
        "reader": {
            "loading_strategy": "dataset",
            "data_type": "transaction",
            "reading_method": "local",
            "rating_type": "explicit",
            "local_path": paths["data"],
            "side": {
                "local_path": paths["side"],
                "column_names": ["item_id", "genre", "decade"],
            },
        },
        "writer": {
            "dataset_name": "timing",
            "writing_method": "local",
            "local_experiment_path": str(tmp_path / "out"),
        },
        "splitter": {
            "test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2},
            "validation_splitting": {"strategy": "temporal_holdout", "ratio": 0.2},
        },
        "models": {"Pop": {}},
        "evaluation": {"top_k": [5], "metrics": ["nDCG"]},
        **sections,
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return load_train_configuration(str(path))


def test_each_step_and_structure_is_timed_within_the_preparation(tmp_path: Path):
    """Reading, filtering, splitting and every structure built are measured, and
    together take no longer than the preparation itself."""
    config = load_config(tmp_path, filtering={"MinRating": {"min_rating": 2.0}})
    reader = ReaderFactory.get_reader(config)
    timings = DataPreparationTimes()

    start = time.perf_counter()
    main, validation, _ = initialize_datasets(
        reader, WarpRecCallback(), config, timings=timings
    )
    dataset_preparation(main, [], config, timings=timings)
    elapsed = time.perf_counter() - start

    columns = timings.columns()
    built = ["dataset", "interactions", "sessions", "side", "evaluation"]
    for key in [*STEP_COLUMNS, *built]:
        column = {**STEP_COLUMNS, **STRUCTURE_COLUMNS}[key]
        assert columns[column] > 0.0, column
    for key in ["context", "knowledge", "multimodal"]:
        assert math.isnan(columns[STRUCTURE_COLUMNS[key]])

    structures = sum(columns[STRUCTURE_COLUMNS[key]] for key in built)
    assert math.isclose(columns[BUILDING_COLUMN], structures)
    steps = sum(columns[column] for column in STEP_COLUMNS.values())
    assert steps + columns[BUILDING_COLUMN] <= elapsed

    # Both the main and the validation dataset count towards their structures
    for key in ["dataset", "interactions", "sessions", "side"]:
        assert math.isclose(
            timings.seconds[key],
            main.get_build_times()[key] + validation.get_build_times()[key],
        )


def test_a_step_the_run_does_not_take_has_no_time(tmp_path: Path):
    """Without filtering configured the filtering column is NaN."""
    config = load_config(tmp_path)
    timings = DataPreparationTimes()
    initialize_datasets(
        ReaderFactory.get_reader(config), WarpRecCallback(), config, timings=timings
    )

    columns = timings.columns()
    assert math.isnan(columns[STEP_COLUMNS["filtering"]])
    assert math.isnan(columns[STRUCTURE_COLUMNS["evaluation"]])
    assert columns[STEP_COLUMNS["reading"]] > 0.0


def test_a_dataset_times_its_structures_and_they_sum_to_its_construction(
    interactions_frame: pd.DataFrame,
):
    """The structures add up to the dataset's whole construction."""
    start = time.perf_counter()
    dataset = Dataset(interactions_frame, interactions_frame.head(50))
    elapsed = time.perf_counter() - start

    build_times = dataset.get_build_times()
    assert set(build_times) == {"dataset", "interactions", "sessions"}
    assert all(value >= 0.0 for value in build_times.values())
    assert sum(build_times.values()) <= elapsed


def test_the_time_report_row_carries_the_data_columns():
    """The data columns come right after the preparation, in seconds."""
    timings = DataPreparationTimes()
    timings.seconds.update(
        {"reading": 1.0, "splitting": 0.5, "interactions": 2.0, "dataset": 0.25}
    )
    row = model_timing_row(
        model_name="BPR",
        data_preparation_time=10.0,
        exploration_time=10.0,
        ray_report={},
        evaluation_time=2.0,
        inference_time=0.5,
        data_times=timings,
    )

    columns = list(row)
    assert (
        columns[columns.index("Data Preparation Time") + 1] == "Data Reading Time (s)"
    )
    assert row["Data Reading Time (s)"] == 1.0
    assert math.isnan(row["Filtering Time (s)"])
    assert row["Splitting Time (s)"] == 0.5
    assert row["Dataset Building Time (s)"] == 2.25
    assert row["Interactions Building Time (s)"] == 2.0
    assert row["Dataset Core Building Time (s)"] == 0.25
    assert math.isnan(row["Knowledge Graph Building Time (s)"])
    assert row["Total Time"] == 22.0
