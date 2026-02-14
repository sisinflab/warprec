import torch

from fastapi import APIRouter, Depends

from warprec.recommenders.context_aware_recommender import FM

from ....security import get_api_key


router = APIRouter(
    prefix="/contextual",
    tags=["Contextual Models"],
    include_in_schema=False,
    # dependencies=[Depends(get_api_key)],
)

checkpoints_directory = "checkpoints"
model = "FM"

# Load the models checkpoints
#! Model for Movielens contextual recommendations
# movielens_model = torch.load(f"{checkpoints_directory}/{model}_movielens.pth")

#! ...


@router.get(
    "/movielens",
    status_code=200,
    response_model="", #!ContextualDataResponse
    description="Get Movielens contextual recommendations",
    responses={
        # 403: {"description": "Not authenticated"},
    }
)
def get_movielens_recommendations(data): #!ContextualDataRequest
    """Get Movielens contextual recommendations

    Args:
        data ():  data request model

    Returns:
        "": 
    """

    return "" #!ContextualDataResponse(recommendations=)
    