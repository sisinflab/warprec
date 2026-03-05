"""Pydantic models for context-aware recommendation requests and responses."""

from typing import List
from .base import BaseModelCustom


class ContextualDataRequest(BaseModelCustom):
    """Request body for context-aware recommendation.

    Attributes:
        top_k (int): Number of recommendations to return.
        user_id (int): External user identifier.
        context (List[int]): List of context feature values describing the recommendation scenario.
    """

    top_k: int = 10
    user_id: int
    context: List[int]


class ContextualDataResponse(BaseModelCustom):
    """Response body for context-aware recommendation.

    Attributes:
        recommendations (List[int]): Ordered list of recommended item identifiers.
    """

    recommendations: List[int]
