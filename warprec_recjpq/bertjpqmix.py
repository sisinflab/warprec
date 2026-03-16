# pylint: disable=R0801,E1102,W0221,C0103,W0613,W0235,R0902
"""
BertJPQMix: BERT4Rec + JPQ + PPS/SubID Bonus (Mix)
Based on BERT4RecJPQ but with bonus user support from SASRecJPQMix
"""
from collections import defaultdict
from typing import Any, Optional
import hashlib
import random

import narwhals as nw
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.utils.registry import model_registry

from .rec_jpq_combination import ItemCodeLayer


def _stable_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


class MaskingCollator:
    """Collator identico a BertJPQ."""

    def __init__(
        self,
        user_actions: dict[int, list[tuple[Any, int, float]]],
        sequence_length: int,
        val_users: set[int],
        masking_prob: float,
        pad_id: int,
        mask_id: int,
        ignore_val: int = -100,
        mode: str = "train",
        border_timestamp: Optional[Any] = None,
        val_ndcg_at: Optional[int] = None,
        n_items: Optional[int] = None,
    ) -> None:
        self.val_users = val_users
        self.sequence_length = sequence_length
        self.user_actions = user_actions
        self.masking_prob = masking_prob
        self.pad_id = pad_id
        self.ignore_val = ignore_val
        self.mask_id = mask_id
        self.mode = mode
        self.is_validation = mode == "val"
        self.is_train = mode == "train"
        self.is_test = mode == "test"
        self.border_timestamp = border_timestamp
        self.val_ndcg_at = val_ndcg_at
        self.n_items = n_items

    def __call__(self, batch):
        seqs = []
        labels_all = []
        attns = []
        val_ratings_all = []
        val_not_null_seqs_after_ts_idxs = []

        for batch_idx, user_id in enumerate(batch):
            user_hist = self.user_actions[user_id]

            if self.is_train:
                if user_id not in self.val_users or self.border_timestamp is None:
                    seq = [x[1] for x in user_hist if x[2] > 0]
                else:
                    seq = [
                        x[1]
                        for x in user_hist
                        if x[2] > 0 and x[0] < self.border_timestamp
                    ][:-1]

                if len(seq) > self.sequence_length:
                    borderline = random.randint(self.sequence_length, len(seq))
                    seq = seq[borderline - self.sequence_length : borderline]

                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)

                if len(seq) == 0:
                    labels = torch.full(
                        (self.sequence_length,),
                        self.ignore_val,
                        dtype=torch.long,
                    )
                    attn = torch.zeros(self.sequence_length, dtype=torch.long)
                    seq = torch.full(
                        (self.sequence_length,), self.pad_id, dtype=torch.long
                    )
                    seqs.append(seq)
                    attns.append(attn)
                    labels_all.append(labels)
                    continue

                num_masked_items = max(1, int(len(seq) * self.masking_prob)) - 1
                masked_positions = torch.randperm(len(seq))[:num_masked_items]
                last_position = torch.tensor([len(seq) - 1], dtype=torch.long)
                masked_positions = torch.cat((masked_positions, last_position))

                masked_mask = torch.zeros(len(seq), dtype=torch.long)
                masked_mask[masked_positions] = 1
                labels = seq.clone() * masked_mask + self.ignore_val * (1 - masked_mask)
                seq[masked_positions] = self.mask_id

                if len(seq) < self.sequence_length:
                    ignore_pad = torch.tensor(
                        [self.ignore_val] * (self.sequence_length - len(seq)),
                        dtype=torch.long,
                    )
                    labels = torch.cat([ignore_pad, labels], dim=0)

                labels_all.append(labels)

            elif self.is_validation:
                seq = [
                    x[1]
                    for x in user_hist
                    if x[2] > 0 and x[0] < self.border_timestamp
                ]
                seq.append(self.mask_id)

                if len(seq) > self.sequence_length:
                    seq = seq[-self.sequence_length :]

                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)

                label = seq[-2].clone()
                masked_positions = torch.tensor(
                    [len(seq) - 2], dtype=torch.long, requires_grad=False
                )
                seq[masked_positions] = self.mask_id

                seq_after_ts = torch.tensor(
                    [(x[1], x[2]) for x in user_hist if x[0] >= self.border_timestamp],
                    dtype=torch.long,
                    requires_grad=False,
                )

                if seq_after_ts.shape[0] > 0:
                    val_not_null_seqs_after_ts_idxs.append(batch_idx)

                ratings = torch.zeros(
                    self.n_items, dtype=torch.long, requires_grad=False
                )
                if seq_after_ts.shape[0] != 0:
                    track_ids = seq_after_ts[:, 0]
                    track_ratings = seq_after_ts[:, 1]
                    likes_dislikes_mask = (track_ratings == -2) | (track_ratings == 2)
                    plays_skips_mask = ~likes_dislikes_mask
                    ratings.index_put_(
                        (track_ids[likes_dislikes_mask],),
                        track_ratings[likes_dislikes_mask],
                    )
                    current_ratings = ratings[track_ids[plays_skips_mask]]
                    update_mask = (current_ratings != -2) & (current_ratings != 2)
                    ratings.index_put_(
                        (track_ids[plays_skips_mask][update_mask],),
                        track_ratings[plays_skips_mask][update_mask],
                    )

                labels_all.append(label)
                val_ratings_all.append(ratings)

            elif self.is_test:
                seq = [x[1] for x in user_hist if x[2] > 0]
                seq.append(self.mask_id)

                if len(seq) > self.sequence_length:
                    seq = seq[-self.sequence_length :]

                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)
            else:
                raise ValueError(f"Unsupported mode '{self.mode}'.")

            attn = torch.ones_like(seq, requires_grad=False)

            if len(seq) < self.sequence_length:
                pad = torch.tensor(
                    [self.pad_id] * (self.sequence_length - len(seq)),
                    dtype=torch.long,
                )
                zero_pad = torch.zeros_like(pad, requires_grad=False)
                seq = torch.cat([pad, seq], dim=0)
                attn = torch.cat([zero_pad, attn], dim=0)

            seqs.append(seq)
            attns.append(attn)

        if self.is_test:
            return {"seq": torch.stack(seqs), "attn": torch.stack(attns)}
        if self.is_train:
            return {
                "seq": torch.stack(seqs),
                "attn": torch.stack(attns),
                "labels": torch.stack(labels_all),
            }
        return {
            "seq": torch.stack(seqs),
            "attn": torch.stack(attns),
            "labels": torch.stack(labels_all),
            "ratings": torch.stack(val_ratings_all),
            "not_null_seqs_after_ts_idxs": torch.tensor(
                val_not_null_seqs_after_ts_idxs,
                dtype=torch.long,
                requires_grad=False,
            ),
        }


@model_registry.register(name="BertJPQMix")
class BertJPQMix(IterativeRecommender, SequentialRecommenderUtils):
    """
    BERT4Rec + JPQ + PPS/SubID Bonus (Mix).
    Basato su BERT4RecJPQ con supporto bonus user da SASRecJPQMix.
    
    Training: usa score_masked_tokens (standard JPQ)
    Validation/Inference: usa score_all_items_with_bonus_user (con PPS/SubID)
    """

    DATALOADER_TYPE = None

    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    mask_prob: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    max_seq_len: int
    pq_m: int
    centroid_strategy: str
    
    def __dataclass_transform__(self):
        # Ensure attn_dropout_prob has default value
        if not hasattr(self, 'attn_dropout_prob'):
            self.attn_dropout_prob = self.dropout_prob

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)
        self.seed = seed
        
        # Set attn_dropout_prob if not provided
        if not hasattr(self, 'attn_dropout_prob') or self.attn_dropout_prob is None:
            self.attn_dropout_prob = params.get('attn_dropout_prob', self.dropout_prob)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        self.padding_token_id = self.n_items
        self.mask_token_id = self.n_items + 1

        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_device = self.train_device
        self.max_steps_per_epoch = params.get("max_steps_per_epoch", 128)
        self.val_batch_size = params.get("val_batch_size", self.batch_size)
        self.val_ndcg_at = params.get("val_ndcg_at", 10)
        self.early_stop_epochs = params.get("early_stop_epochs", 500)
        self.positions = torch.arange(1, self.val_ndcg_at + 1, dtype=torch.float)
        self.ndcg_discounts = torch.unsqueeze(
            1 / torch.log2(self.positions + 1), 0
        )

        # ItemCodeLayer con supporto bonus (rec_jpq_combination)
        self.item_codes_layer = ItemCodeLayer(
            embedding_size=self.embedding_size,
            pq_m=self.pq_m,
            num_items=self.n_items,
            sequence_length=self.max_seq_len,
            codes_strategy=self.centroid_strategy,
        )
        
        # Set bonus parameters se specificati
        if "alpha" in params:
            self.item_codes_layer.alpha.copy_(torch.tensor(params["alpha"], dtype=torch.float32))
        if "beta" in params:
            self.item_codes_layer.beta.copy_(torch.tensor(params["beta"], dtype=torch.float32))
        
        self.position_embedding = nn.Embedding(
            self.max_seq_len, self.embedding_size
        )
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)
        self.dropout = nn.Dropout(self.dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        self._centroids_assigned = False
        self.user_actions: dict[int, list[tuple[Any, int, float]]] = defaultdict(list)
        self.val_users_internal: set[int] = set()
        self.val_border_timestamp: Optional[Any] = None

        if hasattr(self, "_init_weights"):
            self.apply(self._init_weights)

    def _append_actions_from_sessions(self, sessions: Sessions) -> None:
        """Identico a BertJPQ."""
        processed_df = sessions._get_processed_data()
        cols = [sessions.user_label, sessions.item_label]
        has_timestamp = sessions.timestamp_label in processed_df.columns
        if has_timestamp:
            cols.append(sessions.timestamp_label)

        rows = processed_df.select(cols).to_numpy()
        for idx, row in enumerate(rows):
            user_id = int(row[0])
            item_id = int(row[1])
            timestamp = row[2] if has_timestamp else idx
            self.user_actions[user_id].append((timestamp, item_id, 1.0))

    def _append_actions_from_interactions(
        self,
        interactions: Interactions,
        user_mapping: dict,
        item_mapping: dict,
        timestamp_label: str,
        rating_label: Optional[str],
    ) -> None:
        """Identico a BertJPQ."""
        inter_df = interactions.get_df()
        cols = [interactions.user_label, interactions.item_label]
        has_rating = rating_label is not None and rating_label in inter_df.columns
        has_timestamp = timestamp_label in inter_df.columns

        if has_rating:
            cols.append(rating_label)
        if has_timestamp:
            cols.append(timestamp_label)

        rows = inter_df.select(cols).to_numpy()
        for idx, row in enumerate(rows):
            raw_user = row[0]
            raw_item = row[1]
            user_id = user_mapping.get(raw_user)
            item_id = item_mapping.get(raw_item)
            if user_id is None or item_id is None:
                continue
            rating = row[2] if has_rating else 1.0
            timestamp = row[3] if has_timestamp and has_rating else (row[2] if has_timestamp else idx)
            self.user_actions[user_id].append((timestamp, item_id, rating))

    def _sort_actions(self) -> None:
        for user_id in self.user_actions:
            self.user_actions[user_id].sort(
                key=lambda x: (x[0], _stable_hash(f"{x[1]}_{user_id}"))
            )

    def _assign_centroids(self, sessions: Sessions) -> None:
        """Assegna i centroidi PQ usando le sequenze di training."""
        if self._centroids_assigned:
            return

        train_users = []
        user_offsets = getattr(sessions, "_user_offsets", None)
        flat_items = getattr(sessions, "_flat_items", None)

        if user_offsets is not None and flat_items is not None:
            for user_idx in range(len(user_offsets) - 1):
                start = user_offsets[user_idx]
                end = user_offsets[user_idx + 1]
                if end <= start:
                    continue
                items = flat_items[start:end]
                train_users.append([(0, int(item_id)) for item_id in items])
        else:
            processed_df = sessions._get_processed_data()
            user_col = sessions.user_label
            item_col = sessions.item_label
            
            for user_id in processed_df.select(user_col).unique().to_numpy().flatten():
                user_items = processed_df.filter(
                    processed_df[user_col] == user_id
                ).select(item_col).to_numpy().flatten()
                train_users.append([(0, int(item_id)) for item_id in user_items])

        if not train_users:
            raise ValueError("Unable to extract user histories for JPQ centroid assignment.")

        print(f"\n[BertJPQMix] Assigning codes for {len(train_users)} users...")
        self.item_codes_layer.assign_codes(train_users)
        print(f"[BertJPQMix] Codes assigned successfully!")
        self._centroids_assigned = True

    def set_validation_context(self, dataset) -> None:
        self.user_actions = defaultdict(list)
        self.val_users_internal = set()
        self.val_border_timestamp = None

        train_session = dataset.train_session
        self._append_actions_from_sessions(train_session)
        self._assign_centroids(train_session)

        if dataset.eval_set is not None:
            self._append_actions_from_interactions(
                interactions=dataset.eval_set,
                user_mapping=train_session._umap,
                item_mapping=train_session._imap,
                timestamp_label=train_session.timestamp_label,
                rating_label=dataset.eval_set.rating_label,
            )

            eval_df = dataset.eval_set.get_df()
            if train_session.timestamp_label in eval_df.columns:
                self.val_border_timestamp = eval_df.select(
                    nw.col(train_session.timestamp_label).min()
                ).item()

            eval_users = eval_df.select(dataset.eval_set.user_label).to_numpy().flatten()
            self.val_users_internal = {
                int(train_session._umap[user])
                for user in eval_users
                if user in train_session._umap
            }

        self._sort_actions()

    def _ensure_train_context(
        self,
        interactions: Interactions,
        sessions: Sessions,
    ) -> None:
        del interactions
        if self.user_actions:
            self._assign_centroids(sessions)
            return

        self.user_actions = defaultdict(list)
        self._append_actions_from_sessions(sessions)
        self._sort_actions()
        self._assign_centroids(sessions)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ) -> DataLoader:
        """Allineato a BertJPQ per stabilità del contesto train/val."""
        self._ensure_train_context(interactions, sessions)

        mode = kwargs.pop("mode", "train")
        shuffle = kwargs.pop("shuffle", mode == "train")
        kwargs.pop("generator", None)

        collator = MaskingCollator(
            user_actions=self.user_actions,
            sequence_length=self.max_seq_len,
            val_users=self.val_users_internal,
            masking_prob=self.mask_prob,
            pad_id=self.padding_token_id,
            mask_id=self.mask_token_id,
            mode=mode,
            border_timestamp=self.val_border_timestamp,
            n_items=self.n_items,
        )

        return DataLoader(
            list(self.user_actions.keys()),
            batch_size=self.batch_size if mode == "train" else self.val_batch_size,
            collate_fn=collator,
            shuffle=shuffle,
            **kwargs,
        )

    def _encode(
        self,
        item_seq: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Identico a BertJPQ."""
        if attention_mask is None:
            padding_mask = item_seq == self.padding_token_id
        else:
            padding_mask = attention_mask == 0

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        # Get item embeddings
        item_emb = self.item_codes_layer(item_seq)
        
        # Check embeddings from JPQ layer
        if torch.isnan(item_emb).any() or torch.isinf(item_emb).any():
            print(f"[DEBUG] NaN/Inf in item_codes_layer embeddings!")
            print(f"[DEBUG] item_emb shape: {item_emb.shape}, has NaN: {torch.isnan(item_emb).any()}, has Inf: {torch.isinf(item_emb).any()}")
            print(f"[DEBUG] item_seq min/max: {item_seq.min()}, {item_seq.max()}")
            # Return zero embeddings to continue
            item_emb = torch.zeros_like(item_emb)
        
        pos_emb = self.position_embedding(position_ids)
        
        # Check position embeddings
        if torch.isnan(pos_emb).any() or torch.isinf(pos_emb).any():
            print(f"[DEBUG] NaN/Inf in position embeddings!")
            pos_emb = torch.zeros_like(pos_emb)
        
        input_emb = item_emb + pos_emb
        input_emb = self.layernorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        # Check after layernorm and dropout
        if torch.isnan(input_emb).any() or torch.isinf(input_emb).any():
            print(f"[DEBUG] NaN/Inf after layernorm/dropout!")
            print(f"[DEBUG] input_emb shape: {input_emb.shape}, has NaN: {torch.isnan(input_emb).any()}, has Inf: {torch.isinf(input_emb).any()}")

        return self.transformer_encoder(
            src=input_emb,
            src_key_padding_mask=padding_mask,
        )

    def forward(self, item_seq: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Identico a BertJPQ."""
        return self._encode(item_seq, attention_mask=attention_mask)

    def train_step(self, batch: Any, *args: Any, **kwargs: Any):
        """Training step - usa standard JPQ (NO bonus user)."""
        del args, kwargs
        seq = batch["seq"]
        labels = batch["labels"]
        attn = batch["attn"]

        # Evita batch rows completamente vuote (solo padding), che possono causare NaN
        # dentro TransformerEncoder quando la maschera è tutta True.
        valid_rows = attn.sum(dim=1) > 0
        if not valid_rows.any():
            return torch.tensor(0.0, device=seq.device, requires_grad=True)
        if not valid_rows.all():
            seq = seq[valid_rows]
            labels = labels[valid_rows]
            attn = attn[valid_rows]

        # Check input
        if torch.isnan(seq.float()).any() or torch.isinf(seq.float()).any():
            print(f"[DEBUG] NaN/Inf in seq input!")
            return torch.tensor(0.0, device=seq.device, requires_grad=True)

        hidden = self._encode(seq, attention_mask=attn)
        
        # Check after encoding
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            print(f"[DEBUG] NaN/Inf after encoding!")
            print(f"[DEBUG] hidden shape: {hidden.shape}, has NaN: {torch.isnan(hidden).any()}, has Inf: {torch.isinf(hidden).any()}")
            return torch.tensor(0.0, device=hidden.device, requires_grad=True)
        
        masked_indices = labels != -100

        if not masked_indices.any():
            return torch.tensor(0.0, device=hidden.device, requires_grad=True)

        hidden_masked = hidden[masked_indices]
        labels_masked = labels[masked_indices]
        
        # Training usa standard JPQ (no bonus)
        logits = self.item_codes_layer.score_masked_tokens(hidden_masked)
        
        # Check logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[DEBUG] NaN/Inf in logits!")
            print(f"[DEBUG] logits shape: {logits.shape}, has NaN: {torch.isnan(logits).any()}, has Inf: {torch.isinf(logits).any()}")
            print(f"[DEBUG] hidden_masked stats - min: {hidden_masked.min()}, max: {hidden_masked.max()}, mean: {hidden_masked.mean()}")
            return torch.tensor(0.0, device=hidden.device, requires_grad=True)
        
        loss = F.cross_entropy(logits, labels_masked)
        
        # Check loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"[DEBUG] NaN/Inf in loss!")
            print(f"[DEBUG] loss: {loss}")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        return loss

    def _prepare_prediction_batch(
        self,
        user_seq: Tensor,
        seq_len: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Identico a BertJPQ."""
        seqs = []
        attns = []

        for seq_row, length in zip(user_seq, seq_len):
            row_len = int(length.item())
            seq = seq_row[:row_len].tolist()
            seq.append(self.mask_token_id)
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len :]

            attn = [1] * len(seq)
            if len(seq) < self.max_seq_len:
                pad_len = self.max_seq_len - len(seq)
                seq = [self.padding_token_id] * pad_len + seq
                attn = [0] * pad_len + attn

            seqs.append(seq)
            attns.append(attn)

        return (
            torch.tensor(seqs, dtype=torch.long, device=user_seq.device),
            torch.tensor(attns, dtype=torch.long, device=user_seq.device),
        )

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
        Prediction con bonus user support.
        Usa score_all_items_with_bonus_user per inference personalizzata.
        """
        del user_indices, args, kwargs
        pred_seq, pred_attn = self._prepare_prediction_batch(user_seq, seq_len)
        hidden = self._encode(pred_seq, attention_mask=pred_attn)[:, -1, :]

        # Usa bonus user in prediction
        if item_indices is None:
            return self.item_codes_layer.score_all_items_with_bonus_user(hidden, pred_seq)

        # Per item specifici usa standard (bonus complicato con subset)
        return self.item_codes_layer.score_emb_items(hidden, item_indices)

    @torch.no_grad()
    def recommend_impl(self, batch: dict[str, Tensor], limit: int, mode: str):
        """Recommendation con bonus user in validation."""
        if mode == "val":
            seq = batch["seq"]
            labels = batch["labels"]
            attn = batch["attn"]
            ratings = batch["ratings"]
            retain_idxs = batch["not_null_seqs_after_ts_idxs"].tolist()

            hidden = self._encode(seq, attention_mask=attn)
            
            # Loss calculation usa standard (position -2)
            logits_for_loss = self.item_codes_layer.score_masked_tokens(hidden[:, -2, :])
            log_probs = torch.nn.functional.log_softmax(logits_for_loss, dim=1)
            gt_logprobs = log_probs[torch.arange(len(seq), device=seq.device), labels]

            if retain_idxs:
                retain_tensor = torch.tensor(retain_idxs, dtype=torch.long, device=seq.device)
                
                # nDCG scoring usa bonus user (position -1)
                logits_for_ndcg = self.item_codes_layer.score_all_items_with_bonus_user(
                    hidden[retain_tensor, -1, :],
                    seq[retain_tensor]  # Full sequence for bonus calculation
                )
                log_probs_for_ndcg = torch.nn.functional.log_softmax(
                    logits_for_ndcg, dim=1
                )
                top_k = torch.topk(log_probs_for_ndcg, limit, dim=1)
                return {
                    "gt_logprobs": gt_logprobs,
                    "items_for_ndcg": top_k.indices.cpu(),
                    "ratings": ratings[retain_tensor].cpu(),
                }

            return {
                "gt_logprobs": gt_logprobs,
                "items_for_ndcg": torch.empty((0, limit), dtype=torch.long),
                "ratings": torch.empty((0, self.n_items), dtype=torch.long),
            }

        if mode == "test":
            seq = batch["seq"].to(self.device)
            attn = batch["attn"].to(self.device)
            hidden = self._encode(seq, attention_mask=attn)[:, -1, :]
            
            # Test usa bonus user
            scores = self.item_codes_layer.score_all_items_with_bonus_user(hidden, seq)
            top_k = torch.topk(scores, limit, dim=1)
            return {
                "items": top_k.indices,
                "scores": top_k.values,
            }

        raise ValueError(f"Unknown mode {mode}")

    @torch.no_grad()
    def custom_validation(
        self,
        dataset,
        validation_top_k: int,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation identica a BertJPQ."""
        del dataset, kwargs
        if not self.val_users_internal or self.val_border_timestamp is None:
            return {f"nDCG@{validation_top_k}": float("-inf"), "val_loss": float("inf")}

        current_device = self.device
        self.eval()
        self.to(self.eval_device)

        val_collator = MaskingCollator(
            user_actions=self.user_actions,
            sequence_length=self.max_seq_len,
            val_users=self.val_users_internal,
            masking_prob=self.mask_prob,
            pad_id=self.padding_token_id,
            mask_id=self.mask_token_id,
            mode="val",
            border_timestamp=self.val_border_timestamp,
            val_ndcg_at=validation_top_k,
            n_items=self.n_items,
        )
        val_loader = DataLoader(
            list(self.val_users_internal),
            batch_size=self.val_batch_size,
            collate_fn=val_collator,
            shuffle=False,
        )

        if validation_top_k == self.val_ndcg_at:
            ndcg_discounts = self.ndcg_discounts
        else:
            positions = torch.arange(1, validation_top_k + 1, dtype=torch.float)
            ndcg_discounts = torch.unsqueeze(1 / torch.log2(positions + 1), 0)

        ndcgs = []
        losses = []
        for batch in val_loader:
            batch = {key: value.to(self.eval_device) for key, value in batch.items()}
            recommendations = self.recommend_impl(batch, validation_top_k, "val")
            items_for_ndcg = recommendations["items_for_ndcg"]
            ratings = recommendations["ratings"]

            if items_for_ndcg.numel() > 0:
                true_scores = torch.gather(ratings, 1, items_for_ndcg)
                true_scores[true_scores < 0] = 0
                dcg = torch.sum(true_scores * ndcg_discounts, 1)

                best_ratings = torch.topk(ratings, validation_top_k, dim=1).values
                best_ratings[best_ratings < 0] = 0
                idcg = torch.sum(best_ratings * ndcg_discounts, 1)
                rec_ndcg = torch.nan_to_num(torch.div(dcg, idcg))
                ndcgs.append(rec_ndcg)

            losses.append(-recommendations["gt_logprobs"].cpu())

        self.to(current_device)

        ndcg_value = torch.cat(ndcgs).mean().item() if ndcgs else float("-inf")
        loss_value = torch.cat(losses).mean().item() if losses else float("inf")
        return {
            f"nDCG@{validation_top_k}": ndcg_value,
            "val_loss": loss_value,
        }
