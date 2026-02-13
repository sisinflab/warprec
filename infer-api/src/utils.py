import os
from typing import List, Dict, Tuple
import re

import torch
import pandas as pd
from pandas import DataFrame

from torch import Tensor
import torch.nn.functional as F

from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry
from warprec.utils.logger import logger


def check_models_existance(models: List[str], datasets: List[str], checkpoints_directory: str = "checkpoints") -> List[str]:
    """Check if model checkpoints exist in the specified directory.

    Args:
        models (List[str]): List of model names to check.
        datasets (List[str]): List of dataset names to check.
        checkpoints_directory (str, optional): Path to the checkpoints directory. Defaults to "checkpoints".
        
    Returns:
        (List[str]): Updated list of model names with existing checkpoints.
    """
    for model in models:
        for dataset in datasets:
            
            # Construct the expected model checkpoint path
            model_path = os.path.join(checkpoints_directory, f"{model}_{dataset}.pth")
            
            # If the checkpoint does not exist, remove the model from the list
            if not os.path.exists(model_path):
                models.pop(models.index(model))
                logger.msg(f"Checkpoint of model {model} for dataset {dataset} not found at {model_path}. Model will be removed from the list.")
    
    return models
                

def init_controller(models: List[str], datasets: List[str], checkpoints_directory: str = "checkpoints", datasets_directory: str = "datasets", device: str = "cpu") -> Tuple[Dict[str, Recommender], Dict[str, DataFrame]]:
    """Initialize the controller by loading models and datasets.

    Args:
        models (List[str]): List of model names to load.
        datasets (List[str]): List of dataset names to load.
        checkpoints_directory (str, optional): Path to the checkpoints directory. Defaults to "checkpoints".
        datasets_directory (str, optional): Path to the datasets directory. Defaults to "datasets".
        device (str, optional): Device to load the models onto. Defaults to "cpu".
        
    Returns:
        Tuple[Dict[str, Recommender], Dict[str, DataFrame]]: Loaded models and datasets.
    """
    # Check for the existence of model checkpoints and datasets
    models_list = check_models_existance(models=models, datasets=datasets)
    datasets_list = check_datasets_existance(datasets_directory=datasets_directory, datasets=datasets)
    
    # Load datasets
    datasets_dict = load_datasets(datasets_directory=datasets_directory, datasets=datasets_list)
    
    # Map the datasets with key the item in the column 1 and value its external id in column 0
    item_to_ext_mapping = {name: dict(zip(df.iloc[:, 1], df.iloc[:, 0])) for name, df in datasets_dict.items()}
    
    # For each key in datasets_dict remove the " (year)" from the item strings if present
    for name, mapping in item_to_ext_mapping.items():
        new_mapping = {}
        for item, ext_id in mapping.items():
            # Remove " (year)" using regex
            new_item = re.sub(r" \(\d{4}\)$", "", item)
            new_mapping[new_item] = ext_id
        item_to_ext_mapping[name] = new_mapping
    
    models_dict = load_models(models=models_list, datasets=datasets_dict.keys(), checkpoints_directory=checkpoints_directory, device=device)
    
    # From each model in models_dict, remap the value of item_to_ext_mapping to the internal indices used by the model
    for model_key, model in models_dict.items():
        # Extract dataset name from model key
        dataset_name = model_key.split("_")[-1]
        
        if dataset_name in item_to_ext_mapping:
            item_mapping = {}
            for item, ext_id in item_to_ext_mapping[dataset_name].items():
                if ext_id in model.info['item_mapping']:
                    item_mapping[item] = model.info['item_mapping'][ext_id]
            datasets_dict[dataset_name] = item_mapping
        else:
            logger.attention(f"Dataset {dataset_name} not found in datasets. Skipping item mapping for model {model_key}.")
    
    return models_dict, datasets_dict
    
    
def check_datasets_existance(datasets_directory: str, datasets: List[str]) -> List[str]:
    """Check if dataset files exist in the specified directory.
    
    Args:
        datasets_directory (str): Path to the datasets directory.
        datasets (List[str]): List of dataset names to check.
        
    Returns:
        List[str]: Updated list of dataset names with existing files.
    """
    for dataset in datasets: 

        # Construct the expected dataset path
        dataset_path = f"{datasets_directory}/{dataset}.dat"
        
        # If the dataset file does not exist, remove it from the list
        if not os.path.exists(dataset_path):
            datasets.pop(datasets.index(dataset))
            logger.msg(f"Dataset file for {dataset} not found at {dataset_path}. Dataset will be removed from the list.")
            
    return datasets


def load_datasets(datasets_directory: str, datasets: List[str]) -> Dict[str, DataFrame]:
    """Load datasets from the specified directory.
    
    Args:
        datasets_directory (str): Path to the datasets directory.
        datasets (List[str]): List of dataset names to load.
        
    Returns:
        Dict[str, DataFrame]: Loaded datasets as Pandas DataFrames.
    """
    datasets_dict = {}
    
    # Load each dataset
    for dataset in datasets:
        # Construct the dataset path
        dataset_path = f"{datasets_directory}/{dataset}.dat"
        
        # Load the dataset into a Pandas DataFrame
        df = pd.read_csv(dataset_path, sep="::", encoding='latin-1', engine='python', header=None)
        
        # Store the loaded DataFrame in the dictionary
        datasets_dict[dataset] = df
        
    return datasets_dict


def load_models(models: List[str], datasets: List[str], checkpoints_directory: str, device: str) -> Dict[str, Recommender]:
    """Load models from the specified checkpoints directory.
    
    Args:
        models (List[str]): List of model names to load.
        datasets (List[str]): List of dataset names to load.
        checkpoints_directory (str): Path to the checkpoints directory.
        device (str): Device to load the models onto.
        
    Returns:
        Dict[str, Recommender]: Loaded models.
    """
    models_dict = {}
    
    for model in models:
        for dataset in datasets:
            # Load the model checkpoint
            checkpoint = torch.load(f"{checkpoints_directory}/{model}_{dataset}.pth", weights_only=False, map_location="cpu")
            
            # Initialize the model from the checkpoint
            model = model_registry.get_class(checkpoint['name']).from_checkpoint(checkpoint=checkpoint)
            
            # Move the model to the specified device
            model = model.to(device)
            
            # Store the model in the dictionary
            models_dict[f"{model.name}_{dataset}"] = model
            
    return models_dict
            

def match_sequence_length(sequence: Tensor, model: Recommender) -> Tensor:
    """Pad or truncate the input sequence to match the model's max sequence length.

    Args:
        sequence (Tensor): The input sequence tensor of shape (1, seq_len).
        model (Recommender): The sequential recommender model instance.
    Returns:
        Tensor: The adjusted sequence tensor of shape (1, model.max_seq_len).
    """
    # Adjust the sequence length
    pad_len = model.max_seq_len - sequence.size(1)
    
    # If padding is needed
    if pad_len > 0:
        # Pad the sequence with the model's padding index
        padded_sequence = F.pad(sequence, (0, pad_len), value=model.n_items)
    elif pad_len < 0:
        # Truncate the sequence to the maximum length
        padded_sequence = sequence[:, -model.max_seq_len:]
    else:
        padded_sequence = sequence
        
    return padded_sequence


def detect_separator(dataset_path: str, num_lines=5) -> str:
    """Detect the separator used in a file by reading the first few lines.

    Args:
        dataset_path (str): Path to the dataset file.
        num_lines (int, optional): Number of lines to read for detection. Defaults to 5.
        
    Returns:
        str: Detected separator character. Defaults to comma (",") if detection fails.
    """
    # List of candidate separators to check
    candidates = ["::", "\t", ",", ";", "|", " "]
    
    try:
        # Read the first few lines of the file
        with open(dataset_path, 'r', ) as f:
            lines = [f.readline().strip() for _ in range(num_lines)]
            
            # Remove empty lines
            lines = [line for line in lines if line]
            
        # If no lines were read, return default separator
        if not lines:
            return ","
        
        # Initialize variables to track the best separator
        best_sep = None
        max_consistency = -1
        
        for sep in candidates:
            # Count occurrences of the separator in each line
            counts = [line.count(sep) for line in lines]
            
            # If the separator never appears, skip it
            if sum(counts) == 0:
                continue
                
            # Check for consistency: we want the same number of columns in all lines
            unique_counts = set(counts)
            is_consistent = (len(unique_counts) == 1)
            
            # Selection logic:
            # 1. Prefer separators that yield the same number of columns across all lines (consistent)
            # 2. If both are consistent, prefer the longer one (e.g., "::" over ":")
            # 3. If equal length, prefer the one that appears more frequently (more columns)
            
            # If is consistent, return the separator
            if is_consistent:
                return sep
            
        # If no separator is perfectly consistent, return default
        return ","
    
    except Exception as e:
        logger.attention(f"Unable to detect separator ({e}). Using default ','.")
        return ","
    
    
def get_external_ids_from_item(items: List[str], dataset: DataFrame) -> List[int]:
    """Get the external ID from the dataset given an item using regex.

    Args:
        items (List[str]): List representing the item attributes.
        dataset (DataFrame): Pandas DataFrame representing the dataset.

    Returns:
        List[int]: List of external IDs matching the item.
    """    
    # Build the regex pattern from the item attributes
    pattern_parts = []
    for attr in items:
        if attr == "":
            pattern_parts.append(".*")  # Match any value for empty attributes
        else:
            pattern_parts.append(re.escape(attr))  # Escape special characters in attribute
    
    pattern = "^" + "::".join(pattern_parts) + "$"
    
    # Filter the dataset using the regex pattern on the combined string of all columns
    mask = dataset.apply(lambda row: re.match(pattern, "::".join(row.astype(str))) is not None, axis=1)
    
    # Get the external IDs (assuming the first column is the external ID)
    external_ids = dataset[mask].iloc[:, 0].tolist()
    
    return external_ids
    
    
def get_items_from_external_ids(external_ids: List[int], dataset: DataFrame) -> List[str]:
    """Get the item attributes from the dataset given a list of external IDs.

    Args:
        external_ids (List[int]): List of external IDs.
        dataset (DataFrame): Pandas DataFrame representing the dataset.

    Returns:
        List[str]: List of items represented as strings of concatenated attributes.
    """    
    items = []
    
    for ext_id in external_ids:
        # Find the row in the dataset with the matching external ID
        row = dataset[dataset.iloc[:, 0] == ext_id]
        
        if not row.empty:
            # Concatenate all attributes into a single string separated by "::"
            item_str = "::".join(row.iloc[0].astype(str).tolist())
            items.append(item_str)
        else:
            logger.attention(f"External ID {ext_id} not found in dataset.")
    
    return items
    
    
    