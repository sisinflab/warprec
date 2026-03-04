"""Pydantic models for collaborative recommendation requests and responses."""

from .base import BaseModelCustom


class CollaborativeDataRequest(BaseModelCustom):
    """Request body for collaborative filtering recommendation.

    Attributes:
        top_k: Number of recommendations to return.
        user_index: External user identifier used to look up the user in the model.
    """

    top_k: int = 10
    user_index: int


class CollaborativeDataResponse(BaseModelCustom):
    """Response body for collaborative filtering recommendation.

    Attributes:
        recommendations: Ordered list of recommended item identifiers.
    """

    recommendations: list[int]
