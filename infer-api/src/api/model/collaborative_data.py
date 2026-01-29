from typing import List

from .base_model import BaseModelCustom


class CollaborativeDataRequest(BaseModelCustom):
    """Model for collaborative data request.
    
    Attributes:
        user_index (int): The index of the user for whom recommendations are requested.
    """
    top_k: int = 10
    user_index: int
    

class CollaborativeDataResponse(BaseModelCustom):
    """Model for collaborative data response.
    
    Attributes:
        recommendations (List[int]): List of recommended item IDs.
    """
    recommendations: List[int]