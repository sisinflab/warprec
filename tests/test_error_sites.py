"""Where WarpRec raises each of its errors, triggered the way a user would.

Each error is also the built-in it replaced, which the tests check too, so a
handler written for an earlier release still catches it.
"""

from pathlib import Path

import narwhals as nw
import pandas as pd
import pytest

from warprec.data.filtering import UserMin, apply_filtering
from warprec.data.reader.local_reader import LocalReader
from warprec.data.stash import MODEL_ARGUMENTS, check_stash
from warprec.errors import (
    ConfigurationError,
    DataError,
    DataNotFoundError,
    ExtensionError,
)


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
