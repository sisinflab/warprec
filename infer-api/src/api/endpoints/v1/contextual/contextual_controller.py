import torch

from fastapi import APIRouter, Depends

from ....security import get_api_key


router = APIRouter(
    prefix="/contextual",
    tags=["Contextual Models"],
    # dependencies=[Depends(get_api_key)],
)

checkpoints_directory = "checkpoints/contextual"
model = ""

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
    