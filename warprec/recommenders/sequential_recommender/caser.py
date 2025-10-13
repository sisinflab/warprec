# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235, R0914
from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss
from warprec.data.dataset import Interactions, Sessions
from warprec.utils.registry import model_registry


@model_registry.register(name="Caser")
class Caser(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of Caser algorithm from
    "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding"
    in WSDM 2018.

    For further details, check the `paper <https://arxiv.org/abs/1809.07426>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If essential values like 'items' or 'users'
                    are not passed through the info dict.

    Attributes:
        embedding_size (int): The dimension of the item and user embeddings.
        n_h (int): The number of horizontal filters.
        n_v (int): The number of vertical filters.
        dropout_prob (float): The probability of dropout for the fully connected layer.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    # Model hyperparameters
    embedding_size: int
    n_h: int
    n_v: int
    dropout_prob: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    neg_samples: int
    max_seq_len: int

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, *args, **kwargs)

        # Get information from dataset info
        self.n_items = info.get("items", None)
        self.n_users = info.get("users", None)
        if not self.n_items or not self.n_users:
            raise ValueError(
                "All 'items' and 'users' must be provided to correctly initialize the model."
            )

        # Layers
        self.user_embedding = nn.Embedding(
            self.n_users + 1, self.embedding_size, padding_idx=0
        )
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0
        )

        # Vertical conv layer
        self.conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_len, 1)
        )

        # Horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_len)]
        self.conv_h = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.n_h,
                    kernel_size=(i, self.embedding_size),
                )
                for i in lengths
            ]
        )

        # Fully-connected layers
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)

        # The second FC layer takes the concatenated output of the first FC layer and the user embedding
        self.fc2 = nn.Linear(
            self.embedding_size + self.embedding_size, self.embedding_size
        )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()

        # Initialize weights
        self.apply(self._init_weights)

        # Loss function
        self.loss: nn.Module
        if self.neg_samples > 0:
            self.loss = BPRLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights."""
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
        elif isinstance(module, nn.Conv2d):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_dataloader(self, interactions: Interactions, sessions: Sessions, **kwargs):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            num_negatives=self.neg_samples,
            batch_size=self.batch_size,
            user_id=True,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        if self.neg_samples > 0:
            user, item_seq, _, pos_item, neg_item = [x.to(self._device) for x in batch]
        else:
            user, item_seq, _, pos_item = [x.to(self._device) for x in batch]
            neg_item = None

        seq_output = self.forward(user, item_seq)

        loss: Tensor
        if self.neg_samples > 0:
            pos_items_emb = self.item_embedding(
                pos_item
            )  # [batch_size, embedding_size]
            neg_items_emb = self.item_embedding(
                neg_item
            )  # [batch_size, neg_samples, embedding_size]

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [batch_size]
            neg_score = torch.sum(
                seq_output.unsqueeze(1) * neg_items_emb, dim=-1
            )  # [batch_size, neg_samples]
            loss = self.loss(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss(logits, pos_item)

        return loss

    def forward(self, user: Tensor, item_seq: Tensor) -> Tensor:
        """Forward pass of the Caser model.

        Args:
            user (Tensor): The user ID for each sequence [batch_size,].
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].

        Returns:
            Tensor: The final sequence output embedding [batch_size, embedding_size].
        """
        # --- Embedding Look-up ---
        # Unsqueeze to get a 4-D input for convolution layers:
        # (batch_size, 1, max_seq_len, embedding_size)
        item_seq_emb = self.item_embedding(item_seq).unsqueeze(1)
        user_emb = self.user_embedding(user)  # [batch_size, embedding_size]

        # --- Convolutional Layers ---
        out_v = None
        # Vertical convolution
        if self.n_v > 0:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # Reshape for FC layer

        # Horizontal convolution
        out_hs = []
        out_h = None
        if self.n_h > 0:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # Concatenate outputs of all filters

        # Concatenate vertical and horizontal outputs
        conv_out = torch.cat([out_v, out_h], 1)

        # --- Fully-connected Layers ---
        # Apply dropout
        conv_out = self.dropout(conv_out)

        # First FC layer
        z = self.ac_fc(self.fc1(conv_out))

        # Concatenate with user embedding
        x = torch.cat([z, user_emb], 1)

        # Second FC layer
        seq_output = self.fc2(x)
        seq_output = self.ac_fc(seq_output)

        return seq_output

    @torch.no_grad()
    def predict_full(
        self,
        user_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned session embeddings (full sort prediction).

        Args:
            user_indices (Tensor): The batch of user indices.
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_indices = user_indices.to(self._device)
        user_seq = user_seq.to(self._device)

        seq_output = self.forward(user_indices, user_seq)

        all_item_embeddings = self.item_embedding.weight[1:]
        predictions = torch.matmul(seq_output, all_item_embeddings.transpose(0, 1))
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction of given items using the learned session embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices to predict for.
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.
        """
        # Move inputs to the correct device
        user_indices = user_indices.to(self._device)
        user_seq = user_seq.to(self._device)
        item_indices = item_indices.to(self._device)

        # Calculate the sequential output embedding for each user in the batch
        seq_output = self.forward(
            user_indices, user_seq
        )  # [batch_size, embedding_size]

        # Get embeddings for candidate items. We clamp the indices to avoid
        # out-of-bounds errors with the padding value (-1).
        candidate_item_embeddings = self.item_embedding(
            item_indices.clamp(min=0)
        )  # [batch_size, pad_seq, embedding_size]

        # Compute scores using a batch matrix multiplication or einsum.
        predictions = torch.einsum(
            "bi,bji->bj", seq_output, candidate_item_embeddings
        )  # [batch_size, pad_seq]
        return predictions.to(self._device)
