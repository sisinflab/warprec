"""Where WarpRec raises each of its errors, triggered the way a user would.

Each error is also the built-in it replaced, which the tests check too, so a
handler written for an earlier release still catches it.
"""

import uuid
from pathlib import Path

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from warprec.data.filtering import UserMin, apply_filtering
from warprec.data.reader.local_reader import LocalReader
from warprec.data.stash import MODEL_ARGUMENTS, check_stash
from warprec.errors import (
    CheckpointError,
    ConfigurationError,
    CustomModuleError,
    DataError,
    DataNotFoundError,
    EvaluationError,
    ExtensionError,
    ModelError,
)
from warprec.evaluation.metrics.multiobjective.hypervolume import Hypervolume
from warprec.evaluation.propensity import build_propensity
from warprec.recommenders.collaborative_filtering_recommender.autoencoder.ease import (
    EASE,
)
from warprec.utils.helpers import load_custom_modules, model_param_from_dict


def test_a_filter_that_empties_the_dataset_is_a_data_error(
    interactions_frame: pd.DataFrame,
):
    with pytest.raises(DataError, match="Dataset is empty after applying filter"):
        apply_filtering(
            nw.from_native(interactions_frame), [UserMin(min_interactions=10**6)]
        )


def test_an_invalid_filter_parameter_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="min_interactions must be"):
        UserMin(min_interactions=0)


def test_a_missing_feature_file_is_a_data_not_found_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as caught:
        LocalReader().read_array(str(tmp_path / "features.npy"))
    assert isinstance(caught.value, DataNotFoundError)


def test_a_stash_entry_named_like_a_model_argument_is_an_extension_error():
    reserved = sorted(MODEL_ARGUMENTS)[0]
    with pytest.raises(ExtensionError, match="stash entry") as caught:
        check_stash({reserved: 1}, MODEL_ARGUMENTS, "model")
    assert isinstance(caught.value, ValueError)


def test_an_unknown_propensity_estimator_is_an_evaluation_error():
    with pytest.raises(EvaluationError, match="Propensity estimator 'naive'"):
        build_propensity(np.ones(4), estimator="naive")


def test_mismatched_multi_objective_lists_are_a_configuration_error():
    with pytest.raises(ConfigurationError, match="must have the same length"):
        Hypervolume(
            k=5,
            num_users=2,
            num_items=4,
            metric_names=["nDCG", "Recall"],
            nadir_points=[0.0],
            higher_is_better=[True, True],
        )


def test_a_custom_module_that_fails_to_import_is_a_custom_module_error(
    tmp_path: Path,
):
    broken = tmp_path / f"broken_{uuid.uuid4().hex}.py"
    broken.write_text("raise ImportError('missing dependency')\n")
    with pytest.raises(CustomModuleError, match="Failed to import") as caught:
        load_custom_modules(str(broken))
    assert isinstance(caught.value, ImportError)
    assert isinstance(caught.value, ExtensionError)


def test_a_checkpoint_of_another_model_is_a_checkpoint_error():
    with pytest.raises(CheckpointError, match="Loading a BPR checkpoint") as caught:
        EASE.from_checkpoint({"name": "BPR"})
    assert isinstance(caught.value, ModelError)
    assert isinstance(caught.value, ValueError)


def test_estimating_a_model_without_its_interactions_is_a_model_error():
    with pytest.raises(ModelError, match="requires interactions"):
        EASE.estimate_space(params={"l2": 1.0}, info={"n_users": 2, "n_items": 3})


def test_invalid_parameters_of_an_unvalidated_model_are_a_configuration_error():
    with pytest.raises(ConfigurationError) as caught:
        model_param_from_dict("MyModel", {"optimization": {"properties": {"seed": -1}}})
    assert [key for key, _ in caught.value.errors] == [
        "models.MyModel.optimization.properties.seed"
    ]
