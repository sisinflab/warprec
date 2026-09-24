"""Behavioural tests for the context array and the key built from it.

These two functions sit between the reader and the context-aware models, and
both are places where a mistake is invisible rather than loud: a context array
of the wrong shape trains fine and scores nonsense, and a key that disagrees
with the one the seen-item index was built from silently fails every lookup.
"""

import numpy as np
import pytest

from warprec.data.entities.context import build_context_array, context_key


def test_a_dataset_without_multi_valued_fields_is_left_alone():
    """The common case must not gain a dimension it has no use for."""
    values = np.array([[1, 2], [3, 4], [5, 6]])

    array = build_context_array(values, ["token", "token"], max_len=1)

    assert array.shape == (3, 2)
    assert array.dtype == np.float32
    assert np.array_equal(array, values.astype(np.float32))


def test_a_multi_valued_field_gains_a_dimension_of_its_own():
    """Its values have to sit side by side rather than being folded together."""
    values = np.array([["7", "1 2 3"], ["8", "4 5"]], dtype=object)

    array = build_context_array(values, ["token", "seq"], max_len=3)

    assert array.shape == (2, 2, 3)
    # The single-valued field keeps its value in the first slot and nothing else.
    assert np.array_equal(
        array[:, 0, :], np.array([[7, 0, 0], [8, 0, 0]], dtype="float32")
    )
    # The multi-valued one spreads across the slots, padded where it runs out.
    assert np.array_equal(array[0, 1, :], np.array([1, 2, 3], dtype="float32"))
    assert np.array_equal(array[1, 1, :], np.array([4, 5, 0], dtype="float32"))


def test_a_field_longer_than_the_width_is_cut_to_it():
    """The width is the widest field seen, so anything past it cannot be stored."""
    values = np.array([["1 2 3 4 5"]], dtype=object)

    array = build_context_array(values, ["seq"], max_len=2)

    assert array.shape == (1, 1, 2)
    assert np.array_equal(array[0, 0, :], np.array([1, 2], dtype="float32"))


def test_an_empty_multi_valued_cell_is_all_padding():
    """A row that names no value must not borrow the previous row's."""
    values = np.array([["1 2"], [""]], dtype=object)

    array = build_context_array(values, ["seq"], max_len=2)

    assert np.array_equal(array[1, 0, :], np.zeros(2, dtype="float32"))


def test_the_key_is_hashable_whatever_shape_the_row_has():
    """The seen-item index is a dictionary, so the key has to be usable as one."""
    flat = context_key(np.array([1.0, 2.0]))
    nested = context_key(np.array([[1.0, 0.0], [2.0, 3.0]]))

    assert isinstance(flat, tuple) and isinstance(nested, tuple)
    assert flat == (1.0, 2.0)
    # A multi-valued row is flattened rather than kept ragged.
    assert nested == (1.0, 0.0, 2.0, 3.0)
    assert {flat: "a", nested: "b"}[nested] == "b"


def test_the_same_context_always_gives_the_same_key():
    """The index and the evaluator both go through here; they must agree."""
    row = np.array([[3.0, 1.0], [0.0, 0.0]])

    assert context_key(row) == context_key(row.copy())
    assert context_key(row) != context_key(row[::-1])


@pytest.mark.parametrize("kind", ["token", "float"])
def test_a_single_valued_field_beside_a_multi_valued_one_keeps_its_value(kind: str):
    """Widening the array for one field must not disturb the others."""
    values = np.array([["2.5", "1 2"]], dtype=object)

    array = build_context_array(values, [kind, "seq"], max_len=2)

    assert array[0, 0, 0] == pytest.approx(2.5)
    assert array[0, 0, 1] == 0.0
