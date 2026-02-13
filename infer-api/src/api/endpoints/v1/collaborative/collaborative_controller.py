import torch

from fastapi import APIRouter, Depends

from warprec.recommenders.collaborative_filtering_recommender.latent_factor import BPR

from ....security import get_api_key
from ....model import CollaborativeDataRequest, CollaborativeDataResponse


router = APIRouter(
    prefix="/collaborative",
    tags=["Collaborative Models"],
    # dependencies=[Depends(get_api_key)],
)

checkpoints_directory = "checkpoints"
model = "BPR"

# Load the models checkpoints
# Model for Movielens collaborative recommendations
movielens_model = torch.load(f"{checkpoints_directory}/{model}_movielens.pth", weights_only=False, map_location='cpu')
movielens_model = BPR.from_checkpoint(checkpoint=movielens_model)
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
    response_model=CollaborativeDataResponse,
    description="Get Movielens collaborative recommendations",
    responses={
        # 403: {"description": "Not authenticated"},
    }
)
def get_movielens_recommendations(data: CollaborativeDataRequest):
    """Get Movielens collaborative recommendations

    Args:
        data (CollaborativeDataRequest): Collaborative data request model

    Returns:
        CollaborativeDataResponse: Collaborative data response model
    """
    # Create the tensor from the input data
    user_index = torch.tensor([data.user_index], device=movielens_model.device)
    
    # Get predictions from the model
    predictions = movielens_model.predict(
        user_indices=user_index,
        item_indices=None,
    )
    
    # Get top-k recommendations
    top_k_indices = torch.topk(predictions, k=data.top_k).indices.squeeze().tolist()
    
    #! Map back to original item indices
    top_k_indices = [inv_movielens_model_item_mapping[idx] for idx in top_k_indices]

    return CollaborativeDataResponse(recommendations=top_k_indices)
    