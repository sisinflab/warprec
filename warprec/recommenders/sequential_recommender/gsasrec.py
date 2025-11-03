# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235, R0902
from typing import Callable, Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.data.dataset import Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="gSASRec")
class gSASRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of gSASRec (generalized SASRec).

    This model adapts the SASRec architecture to predict the next item at every
    step of the sequence, using a Group-wise Binary Cross-Entropy (GBCE) loss function.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If essential values like 'items' are not passed
                    through the info dict.

    Attributes:
        DATALOADER_TYPE (DataLoaderType): The type of dataloader used.
        embedding_size (int): The dimension of the item embeddings (hidden_size).
        n_layers (int): The number of transformer encoder layers.
        n_heads (int): The number of attention heads in the transformer.
        inner_size (int): The dimensionality of the feed-forward layer in the transformer.
        dropout_prob (float): The probability of dropout for embeddings and other layers.
        attn_dropout_prob (float): The probability of dropout for the attention weights.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        gbce_t (float): The temperature parameter for the Group-wise Binary Cross-Entropy loss.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
        reuse_item_embeddings (bool): Whether to reuse item embeddings for output or not.
    """

    # Dataloader definition
    DATALOADER_TYPE: DataLoaderType = DataLoaderType.USER_HISTORY_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    gbce_t: float
    neg_samples: int
    max_seq_len: int
    reuse_item_embeddings: bool

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
        if not self.n_items or not self.max_seq_len:
            raise ValueError(
                "Both 'items' must be provided to correctly initialize the model."
            )

        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)

        if not self.reuse_item_embeddings:
            self.output_embedding = nn.Embedding(
                self.n_items + 1, self.embedding_size, padding_idx=0
            )

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.loss = self._gbce_loss_function()

        self.to(self._device)

    def _generate_square_subsequent_mask(self, seq_len: int) -> Tensor:
        """Generate a square mask for the sequence.

        Args:
            seq_len (int): Length of the sequence.

        Returns:
            Tensor: A square mask of shape [seq_len, seq_len] with True for positions
                    that should not be attended to.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self._device), diagonal=1)
        return mask.bool()

    def _get_output_embeddings(self) -> nn.Embedding:
        """Return embeddings based on the flag value reuse_item_embeddings.

        Returns:
            nn.Embedding: The item embedding if reuse_item_embeddings is True,
                else the output embedding.
        """
        if self.reuse_item_embeddings:
            return self.item_embedding
        return self.output_embedding

    def _init_weights(self, module: Module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_dataloader(self, interactions, sessions: Sessions, **kwargs):
        return sessions.get_user_history_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
        )

    def get_output_embeddings(self) -> nn.Embedding:
        """Return embeddings based on the flag value reuse_item_embeddings.

        Returns:
            nn.Embedding: The item embedding if reuse_item_embeddings is True,
                else the output embedding.
        """
        if self.reuse_item_embeddings:
            return self.item_embedding
        return self.output_embedding

    def forward(self, item_seq: Tensor) -> Tensor:
        """Forward pass of gSASRec. Returns the output of the Transformer
        for each token in the input sequence.

        Args:
            item_seq (Tensor): Sequence of items [batch_size, seq_len].

        Returns:
            Tensor: Output of the Transformer encoder [batch_size, seq_len, embedding_size].
        """
        seq_len = item_seq.size(1)
        padding_mask = item_seq == 0
        causal_mask = self._generate_square_subsequent_mask(seq_len)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=self._device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        seq_emb = self.layernorm(item_emb + pos_emb)
        seq_emb = self.emb_dropout(seq_emb)

        transformer_output = self.transformer_encoder(
            src=seq_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return transformer_output

    def _gbce_loss_function(self) -> Callable:
        """Return the General Binary Cross-Entropy (GBCE) loss.

        Returns:
            Callable: The GBCE loss.
        """

        def gbce_loss_fn(
            sequence_hidden_states: Tensor,
            labels: Tensor,
            negatives: Tensor,
            model_input: Tensor,
        ):
            pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
            pos_neg_embeddings = self.get_output_embeddings()(pos_neg_concat)

            logits = torch.einsum(
                "bse, bsne -> bsn", sequence_hidden_states, pos_neg_embeddings
            )

            gt = torch.zeros_like(logits, device=self._device)
            gt[:, :, 0] = 1.0

            alpha = self.neg_samples / (self.n_items - 1)
            t = self.gbce_t
            beta = alpha * ((1 - 1 / alpha) * t + 1 / alpha)

            positive_logits = logits[:, :, 0:1].to(torch.float64)
            negative_logits = logits[:, :, 1:].to(torch.float64)
            eps = 1e-10

            positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1 - eps)
            positive_probs_pow = torch.clamp(
                positive_probs.pow(-beta),
                min=1.0 + eps,
                max=torch.finfo(torch.float64).max,
            )
            to_log = torch.clamp(
                torch.div(1.0, (positive_probs_pow - 1)),
                eps,
                torch.finfo(torch.float64).max,
            )
            positive_logits_transformed = to_log.log()

            final_logits = torch.cat(
                [positive_logits_transformed, negative_logits], -1
            ).to(torch.float32)

            mask = (model_input != 0).float()
            loss_per_element = nn.functional.binary_cross_entropy_with_logits(
                final_logits, gt, reduction="none"
            )

            loss_per_element = loss_per_element.mean(-1) * mask
            total_loss = loss_per_element.sum() / mask.sum().clamp(min=1)
            return total_loss

        return gbce_loss_fn

    def train_step(self, batch: Any, *args, **kwargs):
        positives, negatives = [x.to(self._device) for x in batch]

        if positives.shape[0] == 0 or positives.shape[1] < 2:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        model_input = positives[:, :-1]
        labels = positives[:, 1:]

        if model_input.shape[1] == 0:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        sequence_hidden_states = self.forward(model_input)
        total_loss = self.loss(sequence_hidden_states, labels, negatives, model_input)

        return total_loss

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
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)

        transformer_output = self.forward(user_seq)
        seq_output = self._gather_indexes(transformer_output, seq_len - 1)

        all_item_embeddings = self._get_output_embeddings().weight[1:]
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
        """Prediction of given items using the learned embeddings.

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
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)
        item_indices = item_indices.to(self._device)

        # The forward pass of gSASRec returns the output for all items in the sequence
        transformer_output = self.forward(
            user_seq
        )  # [batch_size, max_seq_len, embedding_size]

        # Get the output embedding corresponding to the last item in each sequence,
        # as this is the one used for the next-item prediction.
        seq_output = self._gather_indexes(
            transformer_output, seq_len - 1
        )  # [batch_size, embedding_size]

        # Get embeddings for candidate items. We clamp the indices to avoid
        # out-of-bounds errors with the padding value (-1)
        candidate_item_embeddings = self._get_output_embeddings()(
            item_indices.clamp(min=0)
        )  # [batch_size, pad_seq, embedding_size]

        # Compute scores using the dot product between the user's final session
        # embedding and the embeddings of all candidate items.
        predictions = torch.einsum(
            "bi,bji->bj", seq_output, candidate_item_embeddings
        )  # [batch_size, pad_seq]
        return predictions.to(self._device)
