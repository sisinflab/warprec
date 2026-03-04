"""Pydantic models for sequential recommendation requests and responses."""

from .base import BaseModelCustom


class SequentialDataRequest(BaseModelCustom):
    """Request body for sequential recommendation.

    Attributes:
        top_k: Number of recommendations to return.
        sequence: Ordered list of external item IDs representing the user's interaction history.
    """

    top_k: int = 10
    sequence: list[int]


class SequentialDataResponse(BaseModelCustom):
    """Response body for sequential recommendation.

    Attributes:
        recommendations: Ordered list of recommended external item IDs.
    """

    recommendations: list[int]
