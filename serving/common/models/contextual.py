"""Pydantic models for context-aware recommendation requests and responses."""

from .base import BaseModelCustom


class ContextualDataRequest(BaseModelCustom):
    """Request body for context-aware recommendation.

    Attributes:
        top_k: Number of recommendations to return.
        user_id: External user identifier.
        context: List of context feature values describing the recommendation scenario.
    """

    top_k: int = 10
    user_id: int
    context: list[int]


class ContextualDataResponse(BaseModelCustom):
    """Response body for context-aware recommendation.

    Attributes:
        recommendations: Ordered list of recommended item identifiers.
    """

    recommendations: list[int]
