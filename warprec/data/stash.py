from typing import Any, Dict, FrozenSet, Iterable

# The keyword arguments WarpRec passes to every model it builds, restores or
# estimates, next to the stash.
MODEL_ARGUMENTS: FrozenSet[str] = frozenset(
    {
        "name",
        "params",
        "info",
        "interactions",
        "transactions",
        "knowledge",
        "multimodal",
        "sessions",
        "seed",
        "block_size",
        "chunk_size",
        "checkpoint",
        "strict",
    }
)

# The keyword arguments the evaluator passes to every metric, next to the stash.
METRIC_ARGUMENTS: FrozenSet[str] = frozenset(
    {
        "k",
        "num_users",
        "num_items",
        "item_interactions",
        "item_indices",
        "train_matrix",
        "feature_lookup",
        "user_cluster",
        "item_cluster",
        "propensity",
    }
)


def check_stash(
    stash: Dict[str, Any], reserved: Iterable[str], receiver: str
) -> Dict[str, Any]:
    """Refuse a stash entry named like an argument WarpRec passes itself.

    The stash is passed as keyword arguments next to WarpRec's own, so an entry
    with one of their names would replace it, or make the call fail with a
    message that does not mention the stash.

    Args:
        stash (Dict[str, Any]): The stash.
        reserved (Iterable[str]): The arguments WarpRec passes to the receiver.
        receiver (str): What the stash is passed to, e.g. 'model' or 'metric'.

    Returns:
        Dict[str, Any]: The stash, unchanged.

    Raises:
        ValueError: If a stash entry has the name of a reserved argument.
    """
    reserved = set(reserved)
    taken = sorted(key for key in stash if key in reserved)
    if taken:
        raise ValueError(
            f"The stash entry {', '.join(repr(key) for key in taken)} has the "
            f"name of an argument WarpRec passes to every {receiver} itself, so "
            "one would replace the other. Rename the entry; the names WarpRec "
            f"uses are {sorted(reserved)}."
        )
    return stash
