import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn_cloze_mask(batch, padding_token_id):
    """Custom collate function to pad variable-length masked samples for cloze data."""
    masked_seqs, pos_items, neg_items, masked_indices = zip(*batch)

    # Pad the main sequence (already padded to max_seq_len, just stack)
    padded_masked_seqs = torch.stack(masked_seqs, 0)

    # Pad the variable-length tensors
    padded_pos_items = pad_sequence(
        pos_items, batch_first=True, padding_value=padding_token_id
    )
    padded_masked_indices = pad_sequence(
        masked_indices, batch_first=True, padding_value=0
    )  # Pad indices with 0

    # Pad negative items (which have an extra dimension)
    max_neg_len = max(t.size(0) for t in neg_items)
    neg_samples_dim = neg_items[0].size(1)
    padded_neg_items = torch.full(
        (len(batch), max_neg_len, neg_samples_dim), padding_token_id, dtype=torch.long
    )
    for i, t in enumerate(neg_items):
        padded_neg_items[i, : t.size(0), :] = t

    return padded_masked_seqs, padded_pos_items, padded_neg_items, padded_masked_indices
