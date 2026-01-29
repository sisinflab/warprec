import torch
import torch.nn.functional as F

from fastapi import APIRouter, Depends

from warprec.recommenders.sequential_recommender import SASRec

from ....security import get_api_key
from ....model import SequentialDataRequest, SequentialDataResponse
from .....utils import match_sequence_length


router = APIRouter(
    prefix="/sequential",
    tags=["Sequential Models"],
    # dependencies=[Depends(get_api_key)],
)

checkpoints_directory = "checkpoints"
model = "SASRec"

# Load the models checkpoints
# Model for Movielens sequential recommendations
movielens_model = torch.load(f"{checkpoints_directory}/{model}_movielens.pth", weights_only=False, map_location='cpu')
movielens_model = SASRec.from_checkpoint(checkpoint=movielens_model)
movielens_model = movielens_model.to("cuda:1")

# Store user and item mappings
movielens_model_user_mapping: dict = movielens_model.info['user_mapping']
movielens_model_item_mapping: dict = movielens_model.info['item_mapping']

# Invert the mappings
inv_movielens_model_user_mapping = {v: k for k, v in movielens_model_user_mapping.items()}
inv_movielens_model_item_mapping = {v: k for k, v in movielens_model_item_mapping.items()}

#! ...


@router.post(
    "/movielens",
    status_code=200,
    response_model=SequentialDataResponse,
    description="Get Movielens sequential recommendations",
    responses={
        # 403: {"description": "Not authenticated"},
    }
)
def get_movielens_recommendations(data: SequentialDataRequest):
    """Get Movielens sequential recommendations

    Args:
        data (SequentialDataRequest): Sequential data request model

    Returns:
        SequentialDataResponse: Sequential data response model
    """
    # Create the tensor from the input data
    sequence = torch.tensor(data.sequence, device=movielens_model.device).unsqueeze(0)
    
    # Adjust the sequence length
    padded_sequence = match_sequence_length(
        sequence=sequence,
        model=movielens_model,
    )
        
    # Get predictions from the model
    predictions = movielens_model.predict(
        user_indices=None,
        item_indices=None,
        user_seq=padded_sequence,
        seq_len=torch.tensor([len(data.sequence)], device=movielens_model.device),
    )
    
    # Get top-k recommendations
    top_k_indices = torch.topk(predictions, k=data.top_k).indices.squeeze().tolist()
    
    #! Map back to original item indices
    top_k_indices = [inv_movielens_model_item_mapping[idx] for idx in top_k_indices]
    
    return SequentialDataResponse(recommendations=top_k_indices)