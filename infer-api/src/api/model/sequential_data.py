from typing import List

from .base_model import BaseModelCustom


class SequentialDataRequest(BaseModelCustom):
    """Model for sequential data request.
    
    Attributes:
        sequence (List[int]): A list of numerical values representing the sequential data.
    """
    sequence: List[int]

class SequentialDataResponse(BaseModelCustom):
    """Model for sequential data response.
    
    Attributes:
        recommendations (List[int]): A list of recommended item IDs.    
    """
    recommendations: List[int]