# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235, R0914
from typing import Any, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="Caser")
class Caser(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of Caser algorithm from
    "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding"
    in WSDM 2018.

    For further details, check the `paper <https://arxiv.org/abs/1809.07426>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The dimension of the item and user embeddings.
        n_h (int): The number of horizontal filters.
        n_v (int): The number of vertical filters.
        dropout_prob (float): The probability of dropout for the fully connected layer.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER_WITH_USER_ID

    # Model hyperparameters
    embedding_size: int
    n_h: int
    n_v: int
    dropout_prob: float
    reg_weight: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    neg_samples: int
    max_seq_len: int

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # Layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
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
        self.main_loss: nn.Module
        if self.neg_samples > 0:
            self.main_loss = BPRLoss()
        else:
            self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            include_user_id=True,
            low_memory=low_memory,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        if self.neg_samples > 0:
            user, item_seq, _, pos_item, neg_item = batch
        else:
            user, item_seq, _, pos_item = batch
            neg_item = None

        seq_output = self.forward(user, item_seq)
        pos_items_emb = self.item_embedding(pos_item)  # [batch_size, embedding_size]

        # Calculate main loss and L2 regularization
        if self.neg_samples > 0:
            neg_items_emb = self.item_embedding(
                neg_item
            )  # [batch_size, neg_samples, embedding_size]

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [batch_size]
            neg_score = torch.sum(
                seq_output.unsqueeze(1) * neg_items_emb, dim=-1
            )  # [batch_size, neg_samples]
            main_loss = self.main_loss(pos_score, neg_score)

            # L2 regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                self.user_embedding(user),
                pos_items_emb,
                neg_items_emb,
            )
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.main_loss(logits, pos_item)

            # L2 regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                self.user_embedding(user),
                pos_items_emb,
            )

        return main_loss + reg_loss

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
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        user_seq: Optional[Tensor] = None,
        seq_len: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned session embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            user_seq (Optional[Tensor]): Padded sequences of item IDs for users to predict for.
            seq_len (Optional[Tensor]): Actual lengths of these sequences, before padding.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Get sequence output embeddings
        seq_output = self.forward(
            user_indices, user_seq
        )  # [batch_size, embedding_size]

        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = self.item_embedding.weight[
                :-1, :
            ]  # [n_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = self.item_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        predictions = torch.einsum(
            einsum_string, seq_output, item_embeddings
        )  # [batch_size, n_items] or [batch_size, pad_seq]
        return predictions
