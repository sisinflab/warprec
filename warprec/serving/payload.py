from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix

from warprec.recommenders.base_recommender import (
    ContextRecommenderUtils,
    Recommender,
    SequentialRecommenderUtils,
)

if TYPE_CHECKING:
    from warprec.data.dataset import Dataset

# Users are packed in slices so that a large dataset never materialises one
# padded matrix of every history at once.
HISTORY_CHUNK = 8192


def build_serving_payload(
    model: Recommender,
    dataset: "Dataset",
    training: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """What a serving process needs from the training data, packed for a checkpoint.

    A serving process loads the checkpoint and nothing else. It needs the items
    each user saw in training, to keep them out of recommendations and to rank
    by popularity when a user is unknown, and - for a sequential model - each
    user's most recent items, to score a user who sends only their id.

    It also records what a client may want to know about the model: what it
    was trained on and how well it scored, and for a context-aware model how
    often each context value occurred, so that a server can describe the
    context it accepts.

    Args:
        model (Recommender): The model being saved.
        dataset (Dataset): The dataset it was trained on.
        training (Optional[Dict[str, Any]]): What the run knows about the
            training: 'dataset' (its name), 'evaluation' (the strategy) and
            'metrics' (the test results, by cutoff and metric).

    Returns:
        Dict[str, Any]: The binary training matrix under 'seen'; for a
            sequential model, the packed histories under 'histories'; for a
            context-aware model, the index of every known context value under
            'context_maps' and how often each occurred under 'context_stats';
            and the training facts under 'training'.
    """
    train = dataset.train_set.get_sparse().tocsr()
    # A stored zero is not an interaction, so it is dropped before the matrix
    # is reduced to the bare fact that a user saw an item.
    train = train.copy()
    train.eliminate_zeros()
    seen = csr_matrix(
        (np.ones(train.nnz, dtype=np.int8), train.indices, train.indptr),
        shape=train.shape,
    )

    histories: Optional[Dict[str, np.ndarray]] = None
    if isinstance(model, SequentialRecommenderUtils):
        histories = _pack_histories(dataset, model.max_seq_len)
    context_maps: Optional[Dict[str, Dict[Any, int]]] = None
    context_stats: Optional[Dict[str, Dict[str, Any]]] = None
    if isinstance(model, ContextRecommenderUtils):
        context_maps = {
            label: dict(mapping)
            for label, mapping in dataset.get_context_maps().items()
        }
        context_stats = _context_stats(dataset)

    training = training or {}
    return {
        "seen": seen,
        "histories": histories,
        "context_maps": context_maps,
        "context_stats": context_stats,
        "training": {
            "dataset": training.get("dataset"),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_users": seen.shape[0],
            "n_items": seen.shape[1],
            "n_interactions": int(seen.nnz),
            "evaluation": training.get("evaluation"),
            "metrics": _scalar_metrics(training.get("metrics") or {}),
        },
    }


def _scalar_metrics(results: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    """The test results as plain numbers keyed 'metric@k'.

    Args:
        results (Dict[int, Dict[str, Any]]): The results by cutoff and metric.

    Returns:
        Dict[str, float]: Every scalar result. Results that hold one value per
            user or per item are left out.
    """
    metrics: Dict[str, float] = {}
    for k, by_metric in results.items():
        for name, value in by_metric.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.item()
            if isinstance(value, (int, float, np.number)):
                metrics[f"{name}@{k}"] = float(value)
    return metrics


def _context_stats(dataset: "Dataset") -> Dict[str, Dict[str, Any]]:
    """How often each context value occurred in training, field by field.

    Args:
        dataset (Dataset): The dataset the model was trained on.

    Returns:
        Dict[str, Dict[str, Any]]: For a categorical or multi-valued field, its
            type and the count of each value; for a numeric field, its type and
            the minimum, maximum and mean.
    """
    transactions = dataset.train_transactions
    contexts = transactions.get_arrays()[3]
    maps = dataset.get_context_maps()
    stats: Dict[str, Dict[str, Any]] = {}
    for field, label in enumerate(transactions.context_labels):
        kind = transactions.context_types.get(label, "token")
        column = contexts[:, field]
        if kind == "float":
            values = column if column.ndim == 1 else column[:, 0]
            stats[label] = {
                "type": kind,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
            continue
        # Index 0 is padding or an unknown value, never a value of its own.
        indices, counts = np.unique(column.astype(np.int64), return_counts=True)
        by_index = {index: value for value, index in maps.get(label, {}).items()}
        stats[label] = {
            "type": kind,
            "counts": {
                by_index[int(index)]: int(count)
                for index, count in zip(indices, counts)
                if int(index) in by_index
            },
        }
    return stats


def _pack_histories(dataset: "Dataset", max_seq_len: int) -> Dict[str, np.ndarray]:
    """Each user's last items, as one flat array and the offset of every user.

    Args:
        dataset (Dataset): The dataset the model was trained on.
        max_seq_len (int): The longest sequence the model reads.

    Returns:
        Dict[str, np.ndarray]: 'items', the concatenated histories, and
            'offsets', where user u's items are items[offsets[u]:offsets[u + 1]].
    """
    n_users = dataset.info()["n_users"]
    chunks: List[np.ndarray] = []
    lengths: List[int] = []
    for start in range(0, n_users, HISTORY_CHUNK):
        users = list(range(start, min(start + HISTORY_CHUNK, n_users)))
        sequences, lens = dataset.train_session.get_user_history_sequences(
            users, max_seq_len
        )
        for row, length in zip(sequences, lens.tolist()):
            chunks.append(row[:length].numpy().astype(np.int32))
            lengths.append(length)

    offsets = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    items = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
    return {"items": items, "offsets": offsets}
