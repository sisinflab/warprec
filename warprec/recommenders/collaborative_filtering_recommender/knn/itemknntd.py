# pylint: disable = R0801, E1102
from typing import Any

import numpy as np
import scipy.sparse as sp

from warprec.data.entities import Interactions
from warprec.recommenders.collaborative_filtering_recommender.knn import ItemKNN
from warprec.utils.registry import model_registry


@model_registry.register(name="ItemKNN-TD")
class ItemKNNTD(ItemKNN):
    """Implementation of ItemKNN with Temporal Decay (ItemKNN-TD).
    from Time Weight Collaborative Filtering (CIKM 2005).

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
        beta (float): The decay rate parameter. A higher beta means older
            interactions decay faster.

    Raises:
        ValueError: If the timestamp column is not found in the dataset, as
            ItemKNNTD requires timestamps to compute temporal decay.
    """

    k: int
    similarity: str
    beta: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, interactions, *args, seed=seed, **kwargs)

        # Retrieve flat user-item interactions
        users, items, values, timestamps = interactions.get_flat()

        # Extract the data and check for timestamp column
        df = interactions.get_df()
        if interactions.timestamp_label not in df.columns:
            raise ValueError(
                f"Timestamp column '{interactions.timestamp_label}' not found in the dataset. "
                "ItemKNNTD requires timestamps to compute temporal decay."
            )

        # Compute the temporal decay weights
        referring_timestamp = np.max(timestamps)

        diff_seconds = referring_timestamp - timestamps
        diff_days = diff_seconds / (24 * 3600)

        # Formula: decay = exp(-(beta * ndays))
        decay_weights = np.exp(-(self.beta * diff_days))
        decayed_values = values * decay_weights

        # Save the trained matrix as sparse
        self.train_matrix = sp.csr_matrix(
            (decayed_values, (users, items)), shape=(self.n_users, self.n_items)
        )
