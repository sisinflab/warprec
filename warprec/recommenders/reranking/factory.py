from typing import TYPE_CHECKING, Optional

import torch

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
        ValueError: If a re-ranker was asked for but the dataset carries no side
            information for it to read.
    """
    if not rerank.enabled():
        return None

    side = dataset.train_set.get_side_sparse()
    if side is None:
        raise ValueError(
            f"The '{rerank.name}' re-ranker orders items by what they are, so the "
            "dataset needs side information. Configure 'reader.side'."
        )

    return reranker_registry.get(
        rerank.name,
        item_features=torch.as_tensor(side.toarray(), dtype=torch.float),
        pool=rerank.pool,
        user_history=dataset.train_set.get_sparse(),
        **rerank.params,
    )
