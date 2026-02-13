from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

import torch
import sys
from pathlib import Path

# Add warprec to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
from warprec.recommenders import Recommender
from warprec.utils.logger import logger

from src import init_controller, match_sequence_length


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

mcp = FastMCP("Warprec MCP Server")

@mcp.tool(description="Recommend items based on a sequence of previously watched films. "
                    "The model used is SASRec trained on the MovieLens dataset."
                    "Use this tool to get sequential recommendations based on a list of liked movies."
                    "Format the liked movies as a list of strings, e.g., ['movie1', 'movie2', 'movie3']."
                    "Do not use years or additional metadata, just the movie titles."
                    "Also, if specified, add the number of top recommendations to return as an integer parameter 'top_k', defaulting to 10.")
def recommend_movielens_sequential(
    item_sequence: list[str],
    top_k: int = 10,
):
    """

    Args:
        item_sequence (list[str]): list of items

    Returns:
        list[int]: top k recommendation
    """
    #! Note: Now the model is hardcoded, but in the future it should be selected based on the request data
    model: Recommender = models["SASRec_movielens"]
    dataset: Dict[str, int] = datasets["movielens"]
    
    # Get the sequence from the request
    sequence = item_sequence
    
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
        seq_len=torch.tensor([len(item_sequence)], device=model.device),
    )
    
    # Get top-k recommendations
    top_k_indices = torch.topk(predictions, k=top_k).indices.squeeze().tolist()

    # Map internal indices back to original item strings
    inv_dataset_mapping = {v: k for k, v in dataset.items()}
    top_k_items = [inv_dataset_mapping[idx] for idx in top_k_indices if idx in inv_dataset_mapping]
    
    return top_k_items
    

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8081
    )