import torch
import torch.nn.functional as F

from typing import Dict

from fastapi import APIRouter, Depends
from pandas import DataFrame

from warprec.recommenders import Recommender
from warprec.utils.logger import logger

from ....security import get_api_key
from ....model import SequentialDataRequest, SequentialDataResponse
from .....utils import check_models_existance, init_controller, match_sequence_length, get_external_ids_from_item, get_items_from_external_ids

# Define the router for sequential models
router = APIRouter(
    prefix="/sequential",
    tags=["Sequential Models"],
    # dependencies=[Depends(get_api_key)],
)

# Define model parameters
checkpoints_directory = "checkpoints"
models = [
    "SASRec",
    #! ... Add any additional dataset parameters if needed
]

# Define dataset parameters
datasets_directory = "datasets"
datasets = [
    "movielens",
    #! ... Add any additional dataset parameters if needed
]

# Initialize the controller by loading models and datasets
models, datasets = init_controller(
    models=models,
    datasets=datasets,
)

print("[INFO] Sequential controller initialized with models and datasets.")

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
    #! Note: Now the model is hardcoded, but in the future it should be selected based on the request data
    model: Recommender = models["SASRec_movielens"]
    dataset: Dict[str, int] = datasets["movielens"]
    
    # Get the sequence from the request
    sequence = data.sequence
    
    #! Considering the sequence, map the id from the original dataset
    # external_ids = get_external_ids_from_item(sequence, dataset)
    
    #! Map the sequence to the model's internal indices
    # internal_indices = []
    # item_mapping = model.info['item_mapping']
    # for ext_id in external_ids:
    #     if ext_id in item_mapping:
    #         internal_indices.append(item_mapping[ext_id])
    #     else:
    #         logger.attention(f"External ID {ext_id} not found in item mapping. Skipping.")
    
    # For each item in the sequence, map it to the internal index using the dataset mapping
    internal_indices = []
    for item in sequence:
        if item in dataset:
            internal_indices.append(dataset[item])
        else:
            logger.attention(f"Item {item} not found in dataset mapping. Skipping.")
    
    # Create the tensor from the input data
    sequence = torch.tensor(internal_indices, device=model.device).unsqueeze(0)
    
    # Adjust the sequence length
    padded_sequence = match_sequence_length(
        sequence=sequence,
        model=model,
    )
        
    # Get predictions from the model
    predictions = model.predict(
        user_indices=None,
        item_indices=None,
        user_seq=padded_sequence,
        seq_len=torch.tensor([len(data.sequence)], device=model.device),
    )
    
    # Get top-k recommendations
    top_k_indices = torch.topk(predictions, k=data.top_k).indices.squeeze().tolist()

    # Map internal indices back to original item strings
    inv_dataset_mapping = {v: k for k, v in dataset.items()}
    top_k_items = [inv_dataset_mapping[idx] for idx in top_k_indices if idx in inv_dataset_mapping]
    
    return SequentialDataResponse(recommendations=top_k_items)