from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import Field

from .base_model_custom import BaseModelCustom


class ApiResponse(BaseModelCustom):
    """API response data model
    
    Attributes:
        message (str): Message of response
        timestamp (datetime): Timestamp of response generation in ISO 8601 format
        data (Optional[Dict]): Additional data if any
    """
    message: str = Field(..., description="Message of response")
    timestamp: datetime = Field(datetime.now(timezone.utc).isoformat(), description="Timestamp of response generation")
    data: Optional[Dict] = Field(None, description="Additional data if any", example={})