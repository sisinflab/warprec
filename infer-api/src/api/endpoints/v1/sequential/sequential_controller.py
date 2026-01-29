import torch

from fastapi import APIRouter, Depends

from warprec.recommenders.sequential_recommender import SASRec

from ....security import get_api_key
from ....model import SequentialDataRequest, SequentialDataResponse


router = APIRouter(
    prefix="/sequential",
    tags=["Sequential Models"],
    # dependencies=[Depends(get_api_key)],
)

checkpoints_directory = "checkpoints/sequential"
model = "SASRec"

# Load the models checkpoints
#TODO Model for Movielens sequential recommendations
# movielens_model = torch.load(f"{checkpoints_directory}/{model}_movielens.pth")
# movielens_model = SASRec.from_checkpoint(checkpoint=movielens_model)

#! ...


@router.get(
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
    
    # Get predictions from the model
    predictions = movielens_model.predict(
        user_indices=None,
        item_indices=None,
        user_seq=sequence,
        seq_len=torch.tensor([len(data.sequence)], device=movielens_model.device),
    )
    
    # Get top-k recommendations
    top_k_indices = torch.topk(predictions, k=data.top_k).indices.squeeze().tolist()
    
    return SequentialDataResponse(recommendations=top_k_indices)
    