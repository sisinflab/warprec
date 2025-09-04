# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_uniform_, xavier_normal_

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss
from warprec.data.dataset import Interactions, Sessions
from warprec.utils.registry import model_registry


@model_registry.register(name="GRU4Rec")
class GRU4Rec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of GRU4Rec algorithm from
    "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items value was not passed through the info dict.

    Attributes:
        embedding_size (int): The dimension of the item embeddings.
        hidden_size (int): The number of features in the hidden state of the GRU.
        num_layers (int): The number of recurrent layers.
        dropout_prob (float): The probability of dropout for the embeddings.
        weight_decay (float): The value of weight decay used in optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    # Model hyperparameters
    embedding_size: int
    hidden_size: int
    num_layers: int
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
        self._name = "GRU4Rec"

        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

        self.item_embedding = nn.Embedding(
            items + 1,
            self.embedding_size,
            padding_idx=0,  # Taking into account the extra "item" for padding
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,  # Input tensors are (batch, seq_len, features)
        )

        # Dense layer to project GRU output back to embedding_size
        # Used for prediction
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # Initialize weights
        self.apply(self._init_weights)

        # Loss function will be based on number of
        # negative samples
        self.loss: nn.Module
        if self.neg_samples > 0:
            self.loss = BPRLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        # Move to device
        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.GRU):
            # Uniform Xavier init for GRU net
            # Weight_ih represents input-hidden layers
            # Weight_hh represents hidden-hidden layers
            # No bias used in this network
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    xavier_uniform_(param.data)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_dataloader(self, interactions: Interactions, sessions: Sessions, **kwargs):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            num_negatives=self.neg_samples,
            batch_size=self.batch_size,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = [
                x.to(self._device) for x in batch
            ]
        else:
            item_seq, item_seq_len, pos_item = [x.to(self._device) for x in batch]
            neg_item = None

        seq_output = self.forward(item_seq, item_seq_len)

        loss: Tensor
        if self.neg_samples > 0:
            pos_items_emb = self.item_embedding(
                pos_item
            )  # [batch_size, embedding_size]
            neg_items_emb = self.item_embedding(
                neg_item
            )  # [batch_size, embedding_size]

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [batch_size]
            neg_score = torch.sum(
                seq_output.unsqueeze(1) * neg_items_emb, dim=-1
            )  # [batch_size]
            loss = self.loss(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight  # [num_items, embedding_size]
            logits = torch.matmul(
                seq_output, test_item_emb.transpose(0, 1)
            )  # [batch_size, num_items]
            loss = self.loss(logits, pos_item)

        return loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the GRU4Rec model.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences [batch_size,].

        Returns:
            Tensor: The embedding of the predicted item (last session state)
                    [batch_size, embedding_size].
        """
        item_seq_emb = self.item_embedding(
            item_seq
        )  # [batch_size, max_seq_len, embedding_size]
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        # GRU layers
        # NOTE: Only the output sequence is used in the forward pass
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)  # [batch_size, max_seq_len, embedding_size]

        # Use the utility method to gather the last index of
        # the predicted sequence (the next item)
        seq_output = self._gather_indexes(
            gru_output, item_seq_len - 1
        )  # [batch_size, embedding_size]
        return seq_output

    @torch.no_grad()
    def predict_full(
        self,
        train_batch: Tensor,
        user_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned session embeddings (full sort prediction).

        Args:
            train_batch (Tensor): The train batch of user interactions.
            user_indices (Tensor): The batch of user indices.
            user_seq (Tensor): The user sequence of item interactions.
            seq_len (Tensor): The user sequence length.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)

        # Get the session output embedding for each user
        seq_output = self.forward(user_seq, seq_len)

        # Get embeddings for all items
        all_item_embeddings = self.item_embedding.weight[1:]

        # Calculate scores for all items
        predictions = torch.matmul(seq_output, all_item_embeddings.transpose(0, 1))

        # Masking interaction already seen in train
        predictions[train_batch != 0] = -torch.inf
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(
        self,
        train_batch: Tensor,
        user_indices: Tensor,
        item_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction of given items using the learned embeddings.

        Args:
            train_batch (Tensor): The train batch of user interactions.
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
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)
        item_indices = item_indices.to(self._device)

        # Calculate the sequential output embedding for each user in the batch
        seq_output = self.forward(user_seq, seq_len)  # [batch_size, embedding_size]

        # Get embeddings for candidate items. We clamp the indices to avoid
        # out-of-bounds errors with the padding value (-1)
        candidate_item_embeddings = self.item_embedding(
            item_indices.clamp(min=0)
        )  # [batch_size, pad_seq, embedding_size]

        # Compute scores using a batch matrix multiplication or einsum
        predictions = torch.einsum(
            "bi,bji->bj", seq_output, candidate_item_embeddings
        )  # [batch_size, pad_seq]

        # Mask padded indices
        predictions[item_indices == -1] = -torch.inf
        return predictions.to(self._device)
