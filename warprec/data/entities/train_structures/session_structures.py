from typing import Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    """Personalized dataset for session based data.

    Used by sequential models to capture temporal information from
    user interaction history.

    Args:
        sequences (Tensor): The sequence of interactions.
        sequence_lengths (Tensor): The length of the sequence (before padding).
        positive_items (Tensor): The positive items.
        negative_items (Optional[Tensor]): The negative items.
        users (Optional[Tensor]): The user IDs associated with the sequences.
    """

    def __init__(
        self,
        sequences: Tensor,
        sequence_lengths: Tensor,
        positive_items: Tensor,
        negative_items: Optional[Tensor] = None,
        users: Optional[Tensor] = None,
    ):
        self.sequences = sequences
        self.sequence_lengths = sequence_lengths
        self.positive_items = positive_items
        self.negative_items = negative_items
        self.users = users

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.users is not None:
            if self.negative_items is not None:
                return (
                    self.users[idx],
                    self.sequences[idx],
                    self.sequence_lengths[idx],
                    self.positive_items[idx],
                    self.negative_items[idx],
                )
            return (
                self.users[idx],
                self.sequences[idx],
                self.sequence_lengths[idx],
                self.positive_items[idx],
            )
        if self.negative_items is not None:
            return (
                self.sequences[idx],
                self.sequence_lengths[idx],
                self.positive_items[idx],
                self.negative_items[idx],
            )
        return (
            self.sequences[idx],
            self.sequence_lengths[idx],
            self.positive_items[idx],
        )


class UserHistoryDataset(Dataset):
    """Personalized dataset for user history-based data.

    Used by sequential models that process the entire user interaction history
    as a single sequence.

    Args:
        positive_sequences (Tensor): The sequences of positive interactions.
        negative_samples (Tensor): The sequences of negative interactions.
    """

    def __init__(self, positive_sequences: Tensor, negative_samples: Tensor):
        self.positive_sequences = positive_sequences
        self.negative_samples = negative_samples

    def __len__(self) -> int:
        return len(self.positive_sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.positive_sequences[idx], self.negative_samples[idx]
