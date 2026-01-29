from torch import Tensor
import torch.nn.functional as F

from warprec.recommenders.base_recommender import SequentialRecommenderUtils

def match_sequence_length(sequence: Tensor, model: SequentialRecommenderUtils) -> Tensor:
    """Pad or truncate the input sequence to match the model's max sequence length.

    Args:
        sequence (Tensor): The input sequence tensor of shape (1, seq_len).
        model (SequentialRecommenderUtils): The sequential recommender model instance.

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