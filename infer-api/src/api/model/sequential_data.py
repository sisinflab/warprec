from typing import List

from .base_model import BaseModelCustom


class SequentialDataRequest(BaseModelCustom):
    """Model for sequential data request.
    
    Attributes:
        sequence (List[str]): A list of numerical values representing the sequential data.
    """
    top_k: int = 10
    sequence: List[str]

class SequentialDataResponse(BaseModelCustom):
    """Model for sequential data response.
    
    Attributes:
        recommendations (List[str]): A list of recommended item IDs.    
    """
    recommendations: List[str]