from typing import List

import numpy as np


def build_context_array(
    values: np.ndarray, types: List[str], max_len: int
) -> np.ndarray:
    """Turn the encoded context columns into one array the models can consume.

    Without a multi-valued field the array stays two-dimensional, exactly as it was,
    so datasets that have none are untouched. With one it gains a third dimension
    holding each field's values, padded with the index reserved for padding.

    Args:
        values (np.ndarray): The raw context columns, one row per interaction.
        types (List[str]): The type of each field, 'token', 'float' or 'seq'.
        max_len (int): The widest multi-valued field, or 1 when there is none.

    Returns:
        np.ndarray: The context array, [rows, fields] or [rows, fields, max_len].
    """
    if max_len <= 1:
        return values.astype(np.float32)

    rows, fields = values.shape
    array = np.zeros((rows, fields, max_len), dtype=np.float32)
    for field, kind in enumerate(types):
        column = values[:, field]
        if kind != "seq":
            array[:, field, 0] = column.astype(np.float32)
            continue
        for row, cell in enumerate(column):
            indices = [int(v) for v in str(cell).split() if v][:max_len]
            array[row, field, : len(indices)] = indices
    return array


def context_key(row: np.ndarray) -> tuple:
    """Turn one context row into the hashable key the seen-item index is built on.

    A multi-valued field makes the row two-dimensional, so it is flattened. Both
    the index and the evaluator go through here, which is what keeps the key the
    evaluator looks up identical to the one the index was built with.

    Args:
        row (np.ndarray): One row of the context array.

    Returns:
        tuple: The key identifying the context.
    """
    return tuple(np.asarray(row).ravel().tolist())
