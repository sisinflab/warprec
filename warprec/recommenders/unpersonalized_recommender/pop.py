# pylint: disable = R0801, E1102
from typing import Any, Optional
from collections import defaultdict, Counter

import torch
import mmh3
from torch import Tensor
from warprec.recommenders.base_recommender import Recommender
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
NUM_SPECIAL_ITEMS = 3

@model_registry.register(name="Pop")
class Pop(Recommender):
    """Definition of Popularity unpersonalized model.

    This model will recommend items based on their popularity,
    ensuring that previously seen items are not recommended again.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        normalized_popularity (Tensor): The lookup tensor for normalized
            popularity.
    """

    normalized_popularity: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        X = interactions.get_sparse()

        # Count the number of items to define the popularity
        popularity = torch.tensor(X.sum(axis=0).A1, dtype=torch.float32)
        # Count the total number of interactions
        item_count = torch.tensor(X.sum(), dtype=torch.float32)

        # Normalize popularity by the total number of interactions
        # Add epsilon to avoid division by zero if there are no interactions
        norm_pop = popularity / (item_count + 1e-6)
        self.register_buffer("normalized_popularity", norm_pop)

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using a normalized popularity value.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        if item_indices is None:
            # Case 'full': prediction on all items
            batch_size = user_indices.size(0)

            # Expand the popularity scores for each user in the batch
            return self.normalized_popularity.expand(
                batch_size, -1
            ).clone()  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return self.normalized_popularity[
            item_indices.clamp(max=self.n_items - 1)
        ]  # [batch_size, pad_seq]



@model_registry.register(name="PersonalizedPop")
class PersonalizedPop(Recommender):
    """Definition of Personalized Popularity model.

    This model will recommend items based on their popularity per user,
    computed from the user's last N interactions sorted by timestamp.

    Args:
        params (dict): The dictionary with the model params. Should contain
            'sequence_length' to specify how many recent interactions to consider.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        user_popularity (Tensor): The lookup tensor for user-specific 
            popularity scores [n_users, n_items].
    """

    user_popularity: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # Get sequence length from params, default to 10
        self.sequence_length = params.get("sequence_length", 10)

        # Build the model using temporal information
        self._rebuild_model(interactions)

    def _rebuild_model(self, interactions: Interactions):
        """Rebuild the popularity model based on user's last N interactions.
        
        Args:
            interactions (Interactions): The training interactions with temporal info.
        """
        # Use sparse matrix approach which already has mapped indices
        X = interactions.get_sparse()  # This gives us user_idx x item_idx matrix
        
        # Get the DataFrame to access timestamps
        df = interactions.get_df()
        
        # Get column names
        user_col = df.columns[0]
        item_col = df.columns[1]
        timestamp_col = df.columns[3] if len(df.columns) > 3 else None
        rating_col = df.columns[2] if len(df.columns) > 2 else None
        
        # Convert to dict format for easier access
        df_dict = df.to_dict(as_series=False)
        
        # Build a mapping of (user_idx, item_idx) -> timestamp
        # We need to map original IDs to indices
        user_map = interactions._umap  # user_id -> user_idx
        item_map = interactions._imap  # item_id -> item_idx
        
        # Build user actions with timestamps using mapped indices
        user_actions = defaultdict(list)
        
        for i in range(len(df_dict[user_col])):
            user_id = df_dict[user_col][i]
            item_id = df_dict[item_col][i]
            rating = df_dict[rating_col][i] if rating_col else 1.0
            timestamp = df_dict[timestamp_col][i] if timestamp_col else i
            
            # Map to internal indices
            user_idx = user_map[user_id]
            item_idx = item_map[item_id]
            
            user_actions[user_idx].append((timestamp, item_idx, rating))
        
        # Sort actions by timestamp for each user (with tie-breaking like reference code)
        for user_idx in user_actions:
            # Sort by timestamp, then by hash of (item_id, user_id) for consistent tie-breaking
            user_actions[user_idx].sort(key=lambda x: (x[0], mmh3.hash(f"{x[1]}_{user_idx}")))
        
        # Build popularity scores using Counter (simple frequency count)
        items_counter = defaultdict(Counter)
        
        for user_idx in range(self.n_users):
            if user_idx in user_actions:
                # Get last N positive interactions
                last_positive_actions = [
                    item_idx
                    for _, item_idx, rating in user_actions[user_idx] 
                    if rating > 0
                ][-self.sequence_length:]
                
                # Count occurrences (simple frequency, no weighting)
                for item_idx in last_positive_actions:
                    items_counter[user_idx][item_idx] += 1
        
        # Convert to tensor format
        user_pop = torch.zeros((self.n_users, self.n_items), dtype=torch.float32)
        
        for user_idx in range(self.n_users):
            for item_idx, count in items_counter[user_idx].items():
                user_pop[user_idx, item_idx] = float(count)
        
        self.register_buffer("user_popularity", user_pop)

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using user-specific normalized popularity value.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Clamp user indices to valid range
        user_indices_clamped = user_indices.clamp(max=self.n_users - 1)
        
        if item_indices is None:
            # Case 'full': prediction on all items
            # Return popularity for each user in the batch
            scores = self.user_popularity[user_indices_clamped].clone()  # [batch_size, n_items]
            # Set unseen items (score = 0) to -inf so they are never recommended
            scores[scores == 0] = -torch.inf
            return scores

        # Case 'sampled': prediction on a sampled set of items
        batch_size = user_indices.size(0)
        item_indices_clamped = item_indices.clamp(max=self.n_items - 1)
        
        # Gather user-specific popularity for the sampled items
        user_pop_batch = self.user_popularity[user_indices_clamped]  # [batch_size, n_items]
        
        # Select only the items specified in item_indices
        scores = torch.gather(
            user_pop_batch.unsqueeze(1).expand(-1, item_indices_clamped.size(1), -1),
            dim=2,
            index=item_indices_clamped.unsqueeze(2)
        ).squeeze(2)
        
        return scores  # [batch_size, n_sampled_items]