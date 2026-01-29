import torch

from fastapi import APIRouter, Depends

from ....security import get_api_key


router = APIRouter(
    prefix="/collaborative",
    tags=["Collaborative Models"],
    # dependencies=[Depends(get_api_key)],
)

checkpoints_directory = "checkpoints/collaborative"
model = ""

# Load the models checkpoints
#! Model for Movielens collaborative recommendations
# movielens_model = torch.load(f"{checkpoints_directory}/{model}_movielens.pth")

#! ...


@router.get(
    "/movielens",
    status_code=200,
    response_model="", #!CollaborativeDataResponse
    description="Get Movielens collaborative recommendations",
    responses={
        # 403: {"description": "Not authenticated"},
    }
)
def get_movielens_recommendations(data): #!CollaborativeDataRequest
    """Get Movielens collaborative recommendations

    Args:
        data ():  data request model

    Returns:
        "": 
    """

    return "" #!ContextualDataResponse(recommendations=)
    