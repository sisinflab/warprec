"""Inference service providing type-dispatched recommendation logic.

Centralizes the predict flow for sequential, collaborative, and contextual
recommender models so that both the REST API and MCP server share a single
implementation.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.logger import logger

from .model_manager import ModelManager


class InferenceService:
    """Dispatches recommendation requests to the appropriate model type.

    Args:
        model_manager (ModelManager): A loaded ``ModelManager`` instance.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._manager = model_manager

    # -- public API ----------------------------------------------------------

    def recommend(
        self,
        model_key: str,
        top_k: int = 10,
        item_sequence: Optional[List[int]] = None,
        user_index: Optional[int] = None,
        context: Optional[List[int]] = None,
    ) -> List[int]:
        """Unified entry point that dispatches to the correct recommendation method.

        Args:
            model_key (str): Model-dataset identifier (e.g., ``"SASRec_movielens"``).
            top_k (int): Number of recommendations to return.
            item_sequence (Optional[List[int]]): External item IDs for sequential models.
            user_index (Optional[int]): User identifier for collaborative or contextual models.
            context (Optional[List[int]]): Context feature values for contextual models.

        Returns:
            List[int]: Ordered list of recommended external item identifiers.

        Raises:
            ValueError: If the model type is unknown or required parameters are missing.
        """
        endpoint_type = self._manager.get_endpoint_type(model_key)

        if endpoint_type == "sequential":
            if item_sequence is None:
                raise ValueError(
                    "Parameter 'item_sequence' is required for sequential models."
                )
            return self.recommend_sequential(model_key, item_sequence, top_k)

        if endpoint_type == "collaborative":
            if user_index is None:
                raise ValueError(
                    "Parameter 'user_index' is required for collaborative models."
                )
            return self.recommend_collaborative(model_key, user_index, top_k)

        if endpoint_type == "contextual":
            if user_index is None or context is None:
                raise ValueError(
                    "Parameters 'user_index' and 'context' are required for contextual models."
                )
            return self.recommend_contextual(model_key, user_index, context, top_k)

        raise ValueError(
            f"Unknown endpoint type '{endpoint_type}' for key '{model_key}'."
        )

    def recommend_with_names(
        self,
        model_key: str,
        top_k: int = 10,
        item_names: Optional[List[str]] = None,
        user_index: Optional[int] = None,
        context: Optional[List[int]] = None,
    ) -> List[str]:
        """Wrapper that handles string-to-ID conversion for input and output.

        This allows the MCP tool to work with movie titles while keeping the
        core recommendation logic (used by REST API) based on integer IDs.

        Args:
            model_key (str): Model-dataset identifier (e.g., ``"SASRec_movielens"``).
            top_k (int): Number of recommendations to return.
            item_names (Optional[List[str]]): Item names for sequential models.
            user_index (Optional[int]): User identifier for collaborative or contextual models.
            context (Optional[List[int]]): Context feature values for contextual models.

        Returns:
            List[str]: Ordered list of recommended item names.

        Raises:
            ValueError: If the dataset is not found for the model key.
        """
        dataset_name = self._manager.get_dataset_for_model(model_key)
        if not dataset_name:
            raise ValueError(f"No dataset found for model key '{model_key}'")

        # Map input names to external IDs
        external_ids = None
        if item_names:
            external_ids = []
            for name in item_names:
                ext_id = self._manager.name_to_external_id(dataset_name, name)
                if ext_id is not None:
                    external_ids.append(ext_id)
                else:
                    logger.attention(
                        f"Item '{name}' not found in dataset '{dataset_name}'."
                    )

        # Call the existing recommend method (returns List[int] of external IDs)
        recommended_ids = self.recommend(
            model_key=model_key,
            top_k=top_k,
            item_sequence=external_ids,
            user_index=user_index,
            context=context,
        )

        # Map result external IDs back to names
        return [
            self._manager.external_id_to_name(dataset_name, ext_id)
            for ext_id in recommended_ids
        ]

    def recommend_sequential(
        self,
        model_key: str,
        item_sequence: List[int],
        top_k: int = 10,
    ) -> List[int]:
        """Generate sequential recommendations from an item interaction history.

        Maps external item IDs to internal model indices, pads or truncates the
        sequence to match the model's expected length, runs prediction, and maps
        the top-k internal indices back to external item IDs.

        Args:
            model_key (str): Model-dataset identifier.
            item_sequence (List[int]): Ordered list of external item IDs.
            top_k (int): Number of recommendations to return.

        Returns:
            List[int]: Ordered list of recommended external item IDs.
        """
        model = self._manager.get_model(model_key)
        item_mapping = model.info.get("item_mapping", {})

        # Map external item IDs to internal indices, skipping unknown items
        internal_indices: list[int] = []
        for ext_id in item_sequence:
            if ext_id in item_mapping:
                internal_indices.append(item_mapping[ext_id])
            else:
                logger.attention(
                    f"External item ID '{ext_id}' not found in item mapping. Skipping."
                )

        sequence = torch.tensor(internal_indices, device=model.device).unsqueeze(0)
        padded_sequence = _match_sequence_length(sequence, model)

        predictions = model.predict(
            user_indices=None,
            item_indices=None,
            user_seq=padded_sequence,
            seq_len=torch.tensor([len(item_sequence)], device=model.device),
        )

        top_k_indices = torch.topk(predictions, k=top_k).indices.squeeze().tolist()

        # Build reverse mapping: internal index -> external item ID
        inv_mapping = {v: k for k, v in item_mapping.items()}
        return [inv_mapping[idx] for idx in top_k_indices if idx in inv_mapping]

    def recommend_collaborative(
        self,
        model_key: str,
        user_index: int,
        top_k: int = 10,
    ) -> List[int]:
        """Generate collaborative filtering recommendations for a user.

        Args:
            model_key (str): Model-dataset identifier.
            user_index (int): External user identifier.
            top_k (int): Number of recommendations to return.

        Returns:
            List[int]: Ordered list of recommended external item identifiers.
        """
        model = self._manager.get_model(model_key)

        user_tensor = torch.tensor([user_index], device=model.device)
        predictions = model.predict(user_indices=user_tensor, item_indices=None)

        top_k_indices = torch.topk(predictions, k=top_k).indices.squeeze().tolist()

        # Map internal indices back to external item identifiers
        inv_item_mapping = {v: k for k, v in model.info["item_mapping"].items()}
        return [
            inv_item_mapping[idx] for idx in top_k_indices if idx in inv_item_mapping
        ]

    def recommend_contextual(
        self,
        model_key: str,
        user_index: int,
        context: List[int],
        top_k: int = 10,
    ) -> List[int]:
        """Generate context-aware recommendations for a user.

        This method provides the structural framework for contextual inference.
        A working implementation requires an available context-aware model
        checkpoint (e.g., FM trained on a dataset with context features).

        Args:
            model_key (str): Model-dataset identifier.
            user_index (int): External user identifier.
            context (List[int]): List of context feature values.
            top_k (int): Number of recommendations to return.

        Returns:
            List[int]: Ordered list of recommended external item identifiers.

        Raises:
            NotImplementedError: If the contextual inference flow has not been
                finalized for the loaded model.
        """
        raise NotImplementedError(
            f"Contextual inference for '{model_key}' is not yet implemented. "
            "A context-aware model checkpoint and finalized inference flow are required."
        )


# -- private helpers ---------------------------------------------------------


def _match_sequence_length(sequence: Tensor, model: Recommender) -> Tensor:
    """Pad or truncate a sequence tensor to match the model's max sequence length.

    Short sequences are right-padded with the model's item-count index (used as
    a padding token). Long sequences are truncated from the left, keeping the
    most recent interactions.

    Args:
        sequence (Tensor): Input tensor of shape ``(1, seq_len)``.
        model (Recommender): Sequential recommender model instance.

    Returns:
        Tensor: Adjusted tensor of shape ``(1, model.max_seq_len)``.

    Raises:
        RuntimeError: If the model is not an instance of a sequential model.
    """
    if not isinstance(model, SequentialRecommenderUtils):
        raise RuntimeError("The model is not an instance of a Sequential model")

    pad_len = model.max_seq_len - sequence.size(1)

    if pad_len > 0:
        return F.pad(sequence, (0, pad_len), value=model.n_items)
    if pad_len < 0:
        return sequence[:, -model.max_seq_len :]
    return sequence
