from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix

from warprec.errors import ConfigurationError
from warprec.recommenders.reranking.base import Reranker
from warprec.utils.registry import reranker_registry

if TYPE_CHECKING:
    from warprec.data import Dataset
    from warprec.utils.config import RerankConfig


def build_reranker(rerank: "RerankConfig", dataset: "Dataset") -> Optional[Reranker]:
    """Build the re-ranker a run applies to every list it produces.

    Args:
        rerank (RerankConfig): The re-ranking section of the configuration.
        dataset (Dataset): The dataset the models are evaluated on.

    Returns:
        Optional[Reranker]: The re-ranker, or None when none was configured.

    Raises:
        ConfigurationError: If a re-ranker was asked for but the dataset carries no side
            information for it to read.
    """
    if not rerank.enabled():
        return None

    side = dataset.train_set.get_side_sparse()
    if side is None:
        raise ConfigurationError(
            f"The '{rerank.name}' re-ranker orders items by what they are, so the "
            "dataset needs side information. Configure 'reader.side'."
        )

    return build_reranker_from(rerank, side, dataset.train_set.get_sparse())


def build_reranker_from(
    rerank: "RerankConfig",
    item_features: Union[csr_matrix, np.ndarray],
    user_history: Optional[csr_matrix] = None,
) -> Optional[Reranker]:
    """Build a re-ranker from the data it reads rather than from a dataset.

    A served model has no dataset, only what its checkpoint carries, so this is
    the form both share: given the same features and history, a server builds
    the re-ranker a run built.

    Args:
        rerank (RerankConfig): The re-ranking section of the configuration.
        item_features (Union[csr_matrix, np.ndarray]): The {item x feature}
            content matrix, row i belonging to item index i.
        user_history (Optional[csr_matrix]): The {user x item} training matrix,
            for the re-rankers that read a user's own history.

    Returns:
        Optional[Reranker]: The re-ranker, or None when none was configured.
    """
    if not rerank.enabled():
        return None

    dense = (
        item_features.toarray()
        if isinstance(item_features, csr_matrix)
        else item_features
    )
    return reranker_registry.get(
        rerank.name,
        item_features=torch.as_tensor(dense, dtype=torch.float),
        pool=rerank.pool,
        user_history=user_history,
        **rerank.params,
    )
