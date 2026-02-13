from typing import List

from .base_model import BaseModelCustom


class ContextualDataRequest(BaseModelCustom):
    """Model for contextual data request.
    
    Attributes:
        user_id (int): The ID of the user for whom recommendations are requested.
    """
    top_k: int = 10
    user_id: int
    context: List[int]
    
    

class ContextualDataResponse(BaseModelCustom):
    """Model for contextual data response.
    
    Attributes:
        recommendations (List[int]): List of recommended item IDs.
    """
    recommendations: List[int]