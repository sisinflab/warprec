import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinePositionEncoding(nn.Module):
    def __init__(self, seq_length: int, hidden_size: int, max_wavelength: int = 10000):
        super().__init__()
        pe = torch.zeros(seq_length, hidden_size)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-math.log(max_wavelength) / hidden_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor):
        return F.embedding(positions, self.pe)


class ExpPositionEncoding(nn.Module):
    def __init__(self, seq_len: int, emb_size: int, init: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.pow = nn.Parameter(torch.empty(emb_size))
        nn.init.uniform_(self.pow, -init, init)

    def forward(self, positions: torch.Tensor):
        weights = torch.exp(self.pow)
        positions_norm = (positions.float() + 1.0) / (self.seq_len + 1.0)
        return torch.pow(positions_norm.unsqueeze(-1), weights)


def get_pos_embedding(seq_len: int, emb_size: int, kind: str):
    if kind in {"default", "learnable"}:
        return nn.Embedding(seq_len, emb_size)
    if kind == "exp":
        return ExpPositionEncoding(seq_len, emb_size)
    if kind == "sin":
        return SinePositionEncoding(seq_len, emb_size)
    raise ValueError(f"Unknown pos_embedding kind: {kind}")
