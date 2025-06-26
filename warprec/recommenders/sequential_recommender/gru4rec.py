# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Optional, Callable, Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_uniform_, xavier_normal_
from scipy.sparse import csr_matrix

from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss
from warprec.data.dataset import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="GRU4Rec")
class GRU4Rec(Recommender, SequentialRecommenderUtils):
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
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len(int): The maximum length of sequences.
    """

    # Model hyperparameters
    embedding_size: int
    hidden_size: int
    num_layers: int
    dropout_prob: float
    weight_decay: float
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

        # Get information from dataset info
        self.n_items = info.get("items", None)
        if not self.n_items:
            raise ValueError(
                "Both 'items' must be provided to correctly initialize the model."
            )

        self.item_embedding = nn.Embedding(
            self.n_items + 1,
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

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Loss function will be based on number of
        # negative samples
        self.loss: nn.Module
        if self.neg_samples > 0:
            self.loss = BPRLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

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

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method for GRU4Rec.

        The training will be conducted on sequential data:
        (item_sequence, sequence_length, positive_next_item, negative_next_item (optional)).

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        dataloader = interactions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            num_negatives=self.neg_samples,
        )

        self.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                if self.neg_samples > 0:
                    item_seq, item_seq_len, pos_item, neg_item = [
                        x.to(self._device) for x in batch
                    ]
                else:
                    item_seq, item_seq_len, pos_item = [
                        x.to(self._device) for x in batch
                    ]
                    neg_item = None  # We set them to None for consistency

                # Forward pass: get the session embedding
                self.optimizer.zero_grad()
                seq_output = self.forward(
                    item_seq, item_seq_len
                )  # [batch_size, embedding_size]

                # Loss computation
                total_loss: Tensor
                if self.neg_samples > 0:
                    pos_items_emb = self.item_embedding(
                        pos_item
                    )  # [batch_size, embedding_size]
                    neg_items_emb = self.item_embedding(
                        neg_item
                    )  # [batch_size, embedding_size]

                    pos_score = torch.sum(
                        seq_output * pos_items_emb, dim=-1
                    )  # [batch_size]
                    neg_score = torch.sum(
                        seq_output * neg_items_emb, dim=-1
                    )  # [batch_size]
                    total_loss = self.loss(pos_score, neg_score)
                else:
                    # Compute logits against all item embeddings
                    test_item_emb = (
                        self.item_embedding.weight
                    )  # [n_items, embedding_size]
                    logits = torch.matmul(
                        seq_output, test_item_emb.transpose(0, 1)
                    )  # [batch_size, n_items]
                    # CrossEntropyLoss expects logits and target IDs
                    total_loss = self.loss(logits, pos_item)

                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            if report_fn is not None:
                report_fn(self, loss=epoch_loss)

    @torch.no_grad()
    def predict(
        self,
        interaction_matrix: csr_matrix,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned session embeddings (full sort prediction).

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)

        # Get the session output embedding for each user
        seq_output = self.forward(user_seq, seq_len)  # [num_users, embedding_size]

        # Get embeddings for all items
        all_item_embeddings = self.item_embedding.weight  # [n_items, embedding_size]

        # Calculate scores for all items
        # Scores = dot product of session embedding with all item embeddings
        predictions = torch.matmul(
            seq_output, all_item_embeddings.transpose(0, 1)
        )  # [num_users, n_items]

        # Mask seen items
        coo = interaction_matrix.tocoo()
        user_indices_in_batch = torch.from_numpy(coo.row).to(self._device)
        item_indices = torch.from_numpy(coo.col).to(self._device)
        predictions[user_indices_in_batch, item_indices] = -torch.inf

        return predictions
