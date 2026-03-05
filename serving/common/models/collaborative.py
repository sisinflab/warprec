"""Pydantic models for collaborative recommendation requests and responses."""

from typing import List
from .base import BaseModelCustom


class CollaborativeDataRequest(BaseModelCustom):
    """Request body for collaborative filtering recommendation.

    Attributes:
        top_k (int): Number of recommendations to return.
        user_index (int): External user identifier used to look up the user in the model.
    """

    top_k: int = 10
    user_index: int


class CollaborativeDataResponse(BaseModelCustom):
    """Response body for collaborative filtering recommendation.

    Attributes:
        recommendations (List[int]): Ordered list of recommended item identifiers.
    """

    recommendations: List[int]
