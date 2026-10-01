from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
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


def build_serving_payload(model: Recommender, dataset: "Dataset") -> Dict[str, Any]:
    """What a serving process needs from the training data, packed for a checkpoint.

    A serving process loads the checkpoint and nothing else. It needs the items
    each user saw in training, to keep them out of recommendations and to rank
    by popularity when a user is unknown, and - for a sequential model - each
    user's most recent items, to score a user who sends only their id.

    Args:
        model (Recommender): The model being saved.
        dataset (Dataset): The dataset it was trained on.

    Returns:
        Dict[str, Any]: The binary training matrix under 'seen'; for a
            sequential model, the packed histories under 'histories'; and, for
            a context-aware model, the index of every known context value under
            'context_maps'.
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
    if isinstance(model, ContextRecommenderUtils):
        context_maps = {
            label: dict(mapping)
            for label, mapping in dataset.get_context_maps().items()
        }
    return {"seen": seen, "histories": histories, "context_maps": context_maps}


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
