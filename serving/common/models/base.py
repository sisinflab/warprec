"""Base Pydantic models shared across all serving endpoints."""

from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseModelCustom(BaseModel):
    """Base model with custom configuration that excludes description from JSON schema."""

    model_config = ConfigDict(
        json_schema_extra=lambda schema, _: schema.pop("description", None)
    )


class ApiResponse(BaseModelCustom):
    """Standard API response wrapper.

    Attributes:
        message: Human-readable status message.
        timestamp: ISO 8601 timestamp of the response.
        data: Optional payload dictionary.
    """

    message: str = Field(..., description="Message of response")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of response generation",
    )
    data: Optional[Dict] = Field(None, description="Additional data if any")
