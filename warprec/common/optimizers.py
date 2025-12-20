import torch
from torch import nn
from torch.optim import Optimizer

from warprec.recommenders.base_recommender import IterativeRecommender


def standard_optimizer(model: IterativeRecommender) -> Optimizer:
    """Pre-construct the standard optimizer used within WarpRec.

    The standard approach uses Adam optimizer and separates parameters into two groups:
    1. Decay Group (Adam handles L2):
       - Dense layers weights (Linear, Conv).
       - Structural embeddings (e.g., Positional Embeddings).
    2. No-Decay Group:
       - Sparse Entity Embeddings (User/Item) -> Handled manually by EmbLoss.
       - Biases -> Standard DL practice (no decay).
       - LayerNorm weights -> Standard Transformer practice (no decay).

    Args:
        model (IterativeRecommender): The model on which the optimization
            will be performed.

    Returns:
        Optimizer: The PyTorch optimizer adapted to model parameter.
    """
    # Identify parameters that belong to nn.Embedding modules
    embedding_param_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for param in module.parameters():
                embedding_param_ids.add(id(param))

    # Separate parameters into groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # We disable Adam's weight decay for:
        # A. Biases (standard practice)
        # B. LayerNorm parameters (standard Transformer practice)
        # C. Sparse Embeddings (User/Item), because we use EmbLoss for them.
        #    EXCEPTION: Positional Embeddings should have weight decay applied by Adam.

        is_bias = "bias" in name
        is_layernorm = "layernorm" in name or "norm" in name
        is_embedding = id(param) in embedding_param_ids
        is_positional = "position" in name  # Heuristic to catch position_embedding

        if is_bias or is_layernorm or (is_embedding and not is_positional):
            no_decay_params.append(param)
        else:
            # Linear weights, Conv weights, and Positional Embeddings go here
            decay_params.append(param)

    # Finalize the Optimizer with correct groups
    decay = getattr(model, "weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.Adam(optimizer_grouped_parameters, lr=model.learning_rate)
