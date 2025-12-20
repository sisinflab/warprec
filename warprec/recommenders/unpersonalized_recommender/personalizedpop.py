# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
import pandas as pd
import numpy as np
from torch import Tensor
from scipy.sparse import csr_matrix, coo_matrix
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry
from warprec.data.entities import Interactions

@model_registry.register(name="PersonalizedPop")
class PersonalizedPop(Recommender):

    max_seq_len: int

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        self.pop_matrix = self._build_personalized_pop_matrix(interactions)

    def _build_personalized_pop_matrix(self, interactions: Interactions) -> csr_matrix:
        """Builds an interaction sparse matrix where each interaction (u, i) is
            weighted by the number of total interaction.
        """
        df = interactions.get_df().copy()
        
        # Retrieve lables
        user_label = interactions._user_label
        item_label = interactions._item_label
        
        # Map the DataFrame
        df[user_label] = df[user_label].map(interactions._umap)
        df[item_label] = df[item_label].map(interactions._imap)

        # Clear any NaN values
        df = df.dropna(subset=[user_label, item_label])
        df[user_label] = df[user_label].astype(int)
        df[item_label] = df[item_label].astype(int)

        # Sort interactions
        timestamp_label = "timestamp"
        #timestamp_label = interactions._timestamp_label
        if timestamp_label is not None:
            df = df.sort_values(by=[user_label, timestamp_label])
        else:
            # In case no timestamp label available, assume DataFrame already ordered
            df = df.sort_values(by=[user_label], kind="mergesort")

        # Count transactions per user
        df["transaction_count"] = df.groupby(user_label).cumcount(ascending=False)
        
        # Filter only last 'max_seq_len' transactions
        df_recent = df[df["transaction_count"] < self.max_seq_len]

        # Count interactions
        counts = df_recent.groupby([user_label, item_label]).size().reset_index(name="count")

        # Create the sparse matrix using the weighted interactions values
        row = counts[user_label].values
        col = counts[item_label].values
        data = counts["count"].values

        sparse_pop = coo_matrix(
            (data, (row, col)), 
            shape=(self.n_users, self.n_items)
        ).tocsr()

        return sparse_pop

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        batch_scores_sparse = self.pop_matrix[user_indices.cpu().numpy()]

        predictions = torch.from_numpy(batch_scores_sparse.todense()).to(dtype=torch.float32)

        if item_indices is not None:
            return predictions.gather(1, item_indices.clamp(max=self.n_items - 1))
        
        return predictions