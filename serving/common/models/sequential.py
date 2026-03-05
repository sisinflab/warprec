"""Pydantic models for sequential recommendation requests and responses."""

from typing import List
from .base import BaseModelCustom


class SequentialDataRequest(BaseModelCustom):
    """Request body for sequential recommendation.

    Attributes:
        top_k (int): Number of recommendations to return.
        sequence (List[int]): Ordered list of external item IDs representing the user's interaction history.
    """

    top_k: int = 10
    sequence: List[int]


class SequentialDataResponse(BaseModelCustom):
    """Response body for sequential recommendation.

    Attributes:
        recommendations (List[int]): Ordered list of recommended external item IDs.
    """

    recommendations: List[int]
