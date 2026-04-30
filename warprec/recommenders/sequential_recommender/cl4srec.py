# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Any, Optional, Tuple

import torch
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="CL4SRec")
class CL4SRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of CL4SRec model from
    "Contrastive learning for sequential recommendation" in SIGIR 2021.

    This implementation follows the original paper:
    1. A SASRec-style unidirectional Transformer encoder.
    2. Two random augmentations sampled from crop/mask/reorder.
    3. A multi-task objective with sampled-softmax next-item prediction
       and InfoNCE contrastive learning.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The dimension of the item embeddings (hidden_size).
        n_layers (int): The number of transformer encoder layers.
        n_heads (int): The number of attention heads in the transformer.
        inner_size (int): The dimensionality of the feed-forward layer.
        dropout_prob (float): The probability of dropout for embeddings.
        attn_dropout_prob (float): The probability of dropout for attention weights.
        ssl_lambda (float): The weight for the unsupervised CL loss.
        tau (float): The temperature parameter for contrastive loss.
        sim_type (str): The similarity metric for contrastive loss ("dot" or "cos").
        crop_eta (float): The probability of cropping items in the augmentation.
        mask_gamma (float): The probability of masking items in the augmentation.
        reorder_beta (float): The probability of reordering items in the augmentation.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    ssl_lambda: float
    tau: float
    sim_type: str
    crop_eta: float
    mask_gamma: float
    reorder_beta: float
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

        self.padding_token_id = self.n_items
        self.mask_token_id = self.n_items + 1

        self.item_embedding = nn.Embedding(
            self.n_items + 2,
            self.embedding_size,
            padding_idx=self.padding_token_id,
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        causal_mask = self._generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

        self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()
        self.contrastive_loss = nn.CrossEntropyLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs,
    ):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            **kwargs,
        )

    def _item_crop_batch(
        self, item_seq: Tensor, item_seq_len: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Crop a valid continuous subsequence of length floor(eta * |s|)."""
        batch_size, max_seq = item_seq.shape
        num_left = torch.floor(item_seq_len * self.crop_eta).long().clamp(min=1)

        max_start = (item_seq_len - num_left).clamp(min=0)
        random_offsets = torch.rand(batch_size, device=item_seq.device)
        crop_begin = torch.floor(random_offsets * (max_start + 1).float()).long()

        seq_indices = torch.arange(max_seq, device=item_seq.device).unsqueeze(0)
        gather_idx = crop_begin.unsqueeze(1) + seq_indices
        gathered = torch.gather(item_seq, 1, gather_idx.clamp(max=max_seq - 1))

        keep_mask = seq_indices < num_left.unsqueeze(1)
        cropped = torch.where(
            keep_mask,
            gathered,
            torch.full_like(gathered, self.padding_token_id),
        )
        return cropped, num_left

    def _item_mask_batch(
        self, item_seq: Tensor, item_seq_len: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Mask valid items with a dedicated [mask] token."""
        batch_size, max_seq = item_seq.shape
        num_mask = torch.floor(item_seq_len * self.mask_gamma).long().clamp(min=1)

        rand_vals = torch.rand(batch_size, max_seq, device=item_seq.device)
        pos_indices = torch.arange(max_seq, device=item_seq.device).unsqueeze(0)
        valid_mask = pos_indices < item_seq_len.unsqueeze(1)
        rand_vals = torch.where(valid_mask, rand_vals, torch.ones_like(rand_vals))
        _, sorted_indices = torch.sort(rand_vals, dim=1, descending=False)

        mask_positions = torch.zeros(
            (batch_size, max_seq), dtype=torch.bool, device=item_seq.device
        )
        for i in range(batch_size):
            mask_positions[i, sorted_indices[i, : num_mask[i].item()]] = True  # type: ignore[misc]

        masked = torch.where(
            mask_positions,
            torch.full_like(item_seq, self.mask_token_id),
            item_seq,
        )
        return masked, item_seq_len

    def _item_reorder_batch(
        self, item_seq: Tensor, item_seq_len: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Shuffle a valid continuous subsequence of length floor(beta * |s|)."""
        reordered = item_seq.clone()
        num_reorder = torch.floor(item_seq_len * self.reorder_beta).long().clamp(min=1)

        for i in range(item_seq.shape[0]):
            seq_len_i = int(item_seq_len[i].item())
            reorder_count = min(int(num_reorder[i].item()), max(1, seq_len_i - 1))
            if reorder_count <= 1:
                continue

            max_start = seq_len_i - reorder_count
            reorder_start = torch.randint(
                0,
                max_start + 1,
                (1,),
                device=item_seq.device,
            ).item()
            shuffle_perm = torch.randperm(reorder_count, device=item_seq.device)
            original_window = item_seq[i, reorder_start : reorder_start + reorder_count]  # type: ignore[misc]
            reordered[i, reorder_start : reorder_start + reorder_count] = (  # type: ignore[misc]
                original_window[shuffle_perm]
            )

        return reordered, item_seq_len

    def augment(
        self, item_seq: Tensor, item_seq_len: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate two random augmented views for each sequence."""
        batch_size = item_seq_len.shape[0]
        augment_choices1 = torch.randint(0, 3, (batch_size,), device=item_seq.device)
        augment_choices2 = torch.randint(0, 3, (batch_size,), device=item_seq.device)

        aug_seq1 = item_seq.clone()
        aug_len1 = item_seq_len.clone()
        aug_seq2 = item_seq.clone()
        aug_len2 = item_seq_len.clone()

        crop_mask1 = augment_choices1 == 0
        if crop_mask1.any():
            cropped, crop_len = self._item_crop_batch(
                item_seq[crop_mask1], item_seq_len[crop_mask1]
            )
            aug_seq1[crop_mask1] = cropped
            aug_len1[crop_mask1] = crop_len

        crop_mask2 = augment_choices2 == 0
        if crop_mask2.any():
            cropped, crop_len = self._item_crop_batch(
                item_seq[crop_mask2], item_seq_len[crop_mask2]
            )
            aug_seq2[crop_mask2] = cropped
            aug_len2[crop_mask2] = crop_len

        mask_mask1 = augment_choices1 == 1
        if mask_mask1.any():
            masked, mask_len = self._item_mask_batch(
                item_seq[mask_mask1], item_seq_len[mask_mask1]
            )
            aug_seq1[mask_mask1] = masked
            aug_len1[mask_mask1] = mask_len

        mask_mask2 = augment_choices2 == 1
        if mask_mask2.any():
            masked, mask_len = self._item_mask_batch(
                item_seq[mask_mask2], item_seq_len[mask_mask2]
            )
            aug_seq2[mask_mask2] = masked
            aug_len2[mask_mask2] = mask_len

        reorder_mask1 = augment_choices1 == 2
        if reorder_mask1.any():
            reordered, reorder_len = self._item_reorder_batch(
                item_seq[reorder_mask1], item_seq_len[reorder_mask1]
            )
            aug_seq1[reorder_mask1] = reordered
            aug_len1[reorder_mask1] = reorder_len

        reorder_mask2 = augment_choices2 == 2
        if reorder_mask2.any():
            reordered, reorder_len = self._item_reorder_batch(
                item_seq[reorder_mask2], item_seq_len[reorder_mask2]
            )
            aug_seq2[reorder_mask2] = reordered
            aug_len2[reorder_mask2] = reorder_len

        return aug_seq1, aug_len1, aug_seq2, aug_len2

    def _sampled_softmax_loss(
        self, seq_output: Tensor, pos_item: Tensor, neg_item: Optional[Tensor]
    ) -> Tensor:
        """Main next-item objective used by CL4SRec."""
        if neg_item is None:
            logits = torch.matmul(
                seq_output, self.item_embedding.weight[: self.n_items].T
            )
            return self.main_loss(logits, pos_item)

        pos_logits = torch.sum(
            seq_output * self.item_embedding(pos_item),
            dim=-1,
            keepdim=True,
        )
        neg_logits = torch.sum(
            seq_output.unsqueeze(1) * self.item_embedding(neg_item),
            dim=-1,
        )
        sampled_logits = torch.cat([pos_logits, neg_logits], dim=1)
        sampled_labels = torch.zeros(
            sampled_logits.size(0),
            dtype=torch.long,
            device=seq_output.device,
        )
        return self.main_loss(sampled_logits, sampled_labels)

    def training_step(self, batch: Any, batch_idx: int):
        if len(batch) == 4:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        seq_output = self.forward(item_seq, item_seq_len)
        main_loss = self._sampled_softmax_loss(seq_output, pos_item, neg_item)

        reg_terms = [self.item_embedding(item_seq), self.item_embedding(pos_item)]
        if neg_item is not None:
            reg_terms.append(self.item_embedding(neg_item))
        reg_loss = self.reg_weight * self.reg_loss(*reg_terms)
        total_loss = main_loss + reg_loss

        if self.ssl_lambda > 0 and item_seq.size(0) > 1:
            aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = self.augment(
                item_seq, item_seq_len
            )
            seq_output1 = self.forward(aug_item_seq1, aug_len1)
            seq_output2 = self.forward(aug_item_seq2, aug_len2)

            nce_logits, nce_labels = self._info_nce(
                seq_output1,
                seq_output2,
                temp=self.tau,
                batch_size=item_seq.size(0),
                sim=self.sim_type,
            )
            nce_loss = self.contrastive_loss(nce_logits, nce_labels)
            total_loss += self.ssl_lambda * nce_loss

        return total_loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the SASRec-style encoder used by CL4SRec."""
        seq_len = item_seq.size(1)
        padding_mask = item_seq == self.padding_token_id

        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        seq_emb = self.layernorm(item_emb + pos_emb)
        seq_emb = self.emb_dropout(seq_emb)

        transformer_output = self.transformer_encoder(
            src=seq_emb,
            mask=self.causal_mask[:seq_len, :seq_len],  # type: ignore[index]
            src_key_padding_mask=padding_mask,
        )

        return self._gather_indexes(transformer_output, item_seq_len - 1)

    def _prepare_contrastive_representations(
        self, z_i: Tensor, z_j: Tensor, sim: str
    ) -> Tensor:
        if sim == "cos":
            z_i = torch.nn.functional.normalize(z_i, dim=1)
            z_j = torch.nn.functional.normalize(z_j, dim=1)
        elif sim != "dot":
            raise ValueError(f"Unknown similarity metric: {sim}")
        return torch.cat([z_i, z_j], dim=0)

    def _info_nce(
        self, z_i: Tensor, z_j: Tensor, temp: float, batch_size: int, sim: str = "dot"
    ) -> Tuple[Tensor, Tensor]:
        """Compute InfoNCE with 2N views and 2(N-1) negatives per sample."""
        representations = self._prepare_contrastive_representations(z_i, z_j, sim)
        total_views = 2 * batch_size

        similarity_matrix = torch.matmul(representations, representations.T) / temp
        eye_mask = torch.eye(
            total_views, dtype=torch.bool, device=representations.device
        )
        similarity_matrix = similarity_matrix.masked_fill(eye_mask, float("-inf"))

        positive_indices = torch.arange(total_views, device=representations.device)
        positive_indices = (positive_indices + batch_size) % total_views

        positive_samples = similarity_matrix[
            torch.arange(total_views, device=representations.device),
            positive_indices,
        ].unsqueeze(1)

        negative_mask = ~eye_mask
        negative_mask[
            torch.arange(total_views, device=representations.device),
            positive_indices,
        ] = False
        negative_samples = similarity_matrix[negative_mask].reshape(total_views, -1)

        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(
            total_views, dtype=torch.long, device=representations.device
        )
        return logits, labels

    def _decompose(
        self, z_i: Tensor, z_j: Tensor, origin_z: Tensor, batch_size: int
    ) -> Tuple[Tensor, Tensor]:
        """Decompose contrastive behavior into alignment and uniformity metrics."""
        total_views = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.cdist(z, z, p=2)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(total_views, 1)
        alignment = positive_samples.mean()

        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones(
            (batch_size, batch_size), dtype=torch.bool, device=origin_z.device
        )
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

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
        """Prediction using the learned sequence embeddings."""
        seq_output = self.forward(user_seq, seq_len)

        if item_indices is None:
            item_embeddings = self.item_embedding.weight[: self.n_items, :]
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = self.item_embedding(item_indices)
            einsum_string = "be,bse->bs"

        return torch.einsum(einsum_string, seq_output, item_embeddings)
