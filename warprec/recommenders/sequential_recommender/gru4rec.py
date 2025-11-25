# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Any, Optional

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_uniform_, xavier_normal_

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
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
        DATALOADER_TYPE: The type of dataloader used.
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

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

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

        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

        self.item_embedding = nn.Embedding(
            self.items + 1,
            self.embedding_size,
            padding_idx=self.items,
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
            low_memory=low_memory,
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
        seq_output = self.forward(user_seq, seq_len)  # [batch_size, embedding_size]

        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = self.item_embedding.weight[
                :-1, :
            ]  # [num_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = self.item_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        predictions = torch.einsum(
            einsum_string, seq_output, item_embeddings
        )  # [batch_size, num_items] or [batch_size, pad_seq]
        return predictions
