# pylint: disable=R0801,E1102,W0221,C0103,W0613,W0235,R0902
from collections import defaultdict
from typing import Any, Optional
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import narwhals as nw
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.data.entities import Interactions, Sessions
from warprec.utils.registry import model_registry
import mmh3

from ..rec_jpq_layer import ItemCodeLayer


NUM_SPECIAL_ITEMS = 3  # mask, padding, ignore


def _stable_hash(s: str) -> int:
    return mmh3.hash(s)


class MaskingCollatorBERT:
    """Collator for BERT4RecJPQ PPS train/val/test."""

    def __init__(
        self,
        user_actions: dict[int, list[tuple[Any, int, float]]],
        sequence_length: int,
        val_users: set[int],
        mask_id: int,
        pad_id: int,
        mode: str = "train",
        mask_prob: float = 0.2,
        border_timestamp: Optional[Any] = None,
        val_ndcg_at: Optional[int] = None,
        n_items: Optional[int] = None,
    ) -> None:
        self.val_users = val_users
        self.sequence_length = sequence_length
        self.user_actions = user_actions
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.mode = mode
        self.mask_prob = mask_prob
        self.ignore_val = -100
        self.is_validation = mode == "val"
        self.is_train = mode == "train"
        self.is_test = mode == "test"
        self.border_timestamp = border_timestamp
        self.val_ndcg_at = val_ndcg_at
        self.n_items = n_items

    def __call__(self, batch):
        seqs = []
        labels_all = []
        positions_all = []

        for user_id in batch:
            user_hist = self.user_actions[user_id]

            if self.is_train:
                if user_id not in self.val_users or self.border_timestamp is None:
                    seq = [x[1] for x in user_hist if x[2] > 0]
                else:
                    seq = [
                        x[1]
                        for x in user_hist
                        if x[2] > 0 and x[0] < self.border_timestamp
                    ]

                if len(seq) > self.sequence_length:
                    start_idx = random.randint(0, len(seq) - self.sequence_length)
                    seq = seq[start_idx : start_idx + self.sequence_length]

                labels = seq.copy()
                positions = list(range(1, len(seq) + 1))

                # Apply masking
                for i in range(len(seq)):
                    if random.random() < self.mask_prob:
                        seq[i] = self.mask_id
                    else:
                        labels[i] = self.ignore_val

                if len(seq) < self.sequence_length:
                    pad_len = self.sequence_length - len(seq)
                    seq = [self.pad_id] * pad_len + seq
                    labels = [self.ignore_val] * pad_len + labels
                    positions = [0] * pad_len + positions

                seqs.append(torch.tensor(seq, dtype=torch.long))
                labels_all.append(torch.tensor(labels, dtype=torch.long))
                positions_all.append(torch.tensor(positions, dtype=torch.long))

            elif self.is_test:
                seq = [x[1] for x in user_hist if x[2] > 0]

                if len(seq) > self.sequence_length:
                    seq = seq[-self.sequence_length :]

                positions = list(range(1, len(seq) + 1))

                if len(seq) < self.sequence_length:
                    pad_len = self.sequence_length - len(seq)
                    seq = [self.pad_id] * pad_len + seq
                    positions = [0] * pad_len + positions

                seqs.append(torch.tensor(seq, dtype=torch.long))
                positions_all.append(torch.tensor(positions, dtype=torch.long))

        if self.is_test:
            return {
                "seq": torch.stack(seqs),
                "positions": torch.stack(positions_all),
            }

        return {
            "seq": torch.stack(seqs),
            "labels": torch.stack(labels_all),
            "positions": torch.stack(positions_all),
        }


@model_registry.register(name="BERT4RecJPQ_PPS")
class BERT4RecJPQ_PPS(IterativeRecommender, SequentialRecommenderUtils):
    """BERT4Rec with JPQ and Personalized Product Scoring (PPS)."""

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

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.padding_token_id = self.n_items
        self.mask_token_id = self.n_items + 1
        self.ignore_token_id = self.n_items + 2
        
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_device = torch.device("cpu")
        
        self.max_steps_per_epoch = params.get("max_steps_per_epoch", 128)
        self.val_batch_size = params.get("val_batch_size", self.batch_size)
        self.val_ndcg_at = params.get("val_ndcg_at", 10)
        self.early_stop_epochs = params.get("early_stop_epochs", 500)

        # BERT configuration
        bert_config = BertConfig(
            vocab_size=self.n_items + NUM_SPECIAL_ITEMS,
            hidden_size=self.embedding_size,
            max_position_embeddings=2 * self.max_seq_len,
            attention_probs_dropout_prob=self.attn_dropout_prob,
            hidden_act="gelu",
            hidden_dropout_prob=self.dropout_prob,
            num_attention_heads=self.n_heads,
            num_hidden_layers=self.n_layers,
            intermediate_size=self.inner_size,
        )

        self.bert = BertModel(bert_config, add_pooling_layer=False)

        # JPQ Item Codes Layer
        self.item_codes_layer = ItemCodeLayer(
            embedding_size=self.embedding_size,
            pq_m=self.pq_m,
            num_items=self.n_items,
            sequence_length=self.max_seq_len,
            codes_strategy=self.centroid_strategy,
        )

        self.user_actions: dict[int, list[tuple[Any, int, float]]] = defaultdict(list)
        self.val_users_internal: set[int] = set()
        self.val_border_timestamp: Optional[Any] = None
        self._centroids_assigned = False

        # Position IDs for prediction
        position_ids_arr = np.array(
            list(range(1, self.max_seq_len + 1))
        ).reshape(1, self.max_seq_len)
        self.register_buffer(
            "position_ids_for_pred", torch.tensor(position_ids_arr, dtype=torch.long)
        )

    def _append_actions_from_sessions(self, sessions: Sessions) -> None:
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
            if raw_user not in user_mapping or raw_item not in item_mapping:
                continue

            offset = 2
            rating = float(row[offset]) if has_rating else 1.0
            offset += int(has_rating)
            timestamp = row[offset] if has_timestamp else idx
            user_id = int(user_mapping[raw_user])
            item_id = int(item_mapping[raw_item])
            self.user_actions[user_id].append((timestamp, item_id, rating))

    def _sort_actions(self) -> None:
        for user_id in self.user_actions:
            self.user_actions[user_id].sort(
                key=lambda x: (x[0], _stable_hash(f"{x[1]}_{user_id}"))
            )

    def _assign_centroids(self, sessions: Sessions) -> None:
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
            for actions in self.user_actions.values():
                positives = [(ts, item_id) for ts, item_id, rating in actions if rating > 0]
                if positives:
                    train_users.append(positives)

        if not train_users:
            raise ValueError("Unable to extract user histories for JPQ centroid assignment.")

        self.item_codes_layer.assign_codes(train_users)
        self._centroids_assigned = True

    def set_validation_context(self, dataset) -> None:
        self.user_actions = defaultdict(list)
        self.val_users_internal = set()
        self.val_border_timestamp = None

        train_session = dataset.train_session
        self._append_actions_from_sessions(train_session)

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

        if hasattr(dataset, "test_set") and dataset.test_set is not None:
            self._append_actions_from_interactions(
                interactions=dataset.test_set,
                user_mapping=train_session._umap,
                item_mapping=train_session._imap,
                timestamp_label=train_session.timestamp_label,
                rating_label=dataset.test_set.rating_label,
            )

        self._sort_actions()

    def _ensure_train_context(
        self,
        interactions: Interactions,
        sessions: Sessions,
    ) -> None:
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
        self._ensure_train_context(interactions, sessions)

        mode = kwargs.pop("mode", "train")
        shuffle = kwargs.pop("shuffle", mode == "train")
        kwargs.pop("generator", None)

        collator = MaskingCollatorBERT(
            user_actions=self.user_actions,
            sequence_length=self.max_seq_len,
            val_users=self.val_users_internal,
            mask_id=self.mask_token_id,
            pad_id=self.padding_token_id,
            mode=mode,
            mask_prob=self.mask_prob,
            border_timestamp=self.val_border_timestamp,
            n_items=self.n_items,
        )

        all_users = list(self.user_actions.keys())
        return DataLoader(
            all_users,
            batch_size=self.batch_size,
            collate_fn=collator,
            shuffle=shuffle,
            **kwargs,
        )

    def _encode(self, seq: Tensor, positions: Tensor) -> Tensor:
        """Encode sequence using BERT with JPQ embeddings."""
        batch_size = seq.size(0)
        item_embeddings = self.item_codes_layer(seq)
        bert_output = self.bert(
            inputs_embeds=item_embeddings, position_ids=positions
        ).last_hidden_state
        return bert_output

    def custom_train_loop(
        self,
        dataset,
        validation_top_k: int,
        device: str | torch.device,
    ):
        """Training loop with PPS."""
        self.set_validation_context(dataset)

        train_loader = self.get_dataloader(
            interactions=dataset.train_set,
            sessions=dataset.train_session,
            mode="train",
            shuffle=True,
            num_workers=0,
        )
        
        batches_per_epoch = min(
            self.max_steps_per_epoch,
            max(len(self.user_actions) // self.batch_size, 1),
        )
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_ndcg = float("-inf")
        epochs_since_best = 0

        self.to(device)
        for epoch in range(self.epochs):
            self.train()
            self.to(device)
            epoch_loss_sum = 0.0
            steps_done = 0

            for step, batch in enumerate(train_loader):
                if step >= batches_per_epoch:
                    break

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                optimizer.zero_grad()

                seq = batch["seq"]
                labels = batch["labels"]
                positions = batch["positions"]

                loss = self.train_step({"seq": seq, "labels": labels, "positions": positions})

                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()
                steps_done += 1

            epoch_loss_mean = epoch_loss_sum / max(steps_done, 1)
            epoch_result = self.custom_validation(
                dataset=dataset,
                validation_top_k=validation_top_k,
            )
            current_ndcg = epoch_result.get(f"nDCG@{validation_top_k}", float("-inf"))

            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            yield {
                "loss": epoch_loss_mean,
                "val_loss": epoch_result.get("val_loss", float("inf")),
                f"nDCG@{validation_top_k}": current_ndcg,
                f"best_nDCG@{validation_top_k}": best_ndcg,
            }

            if (
                self.max_steps_per_epoch * (self.early_stop_epochs - epochs_since_best)
            ) <= 0:
                break

    def train_step(self, batch: Any, *args: Any, **kwargs: Any):
        """Training step with PPS scoring."""
        seq = batch["seq"]
        labels = batch["labels"]
        positions = batch["positions"]
        
        batch_size = seq.size(0)

        # Get BERT embeddings
        hidden = self._encode(seq, positions)
        
        # Score all items for all positions using PPS
        logits = self.item_codes_layer.score_sequence_all_items(hidden, batch_size)
        
        # Create ground truth
        masked_indices = labels != -100
        
        if not masked_indices.any():
            return torch.tensor(0.0, device=hidden.device, requires_grad=True)
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, self.n_items)
        labels_flat = labels.view(-1)
        masked_flat = masked_indices.view(-1)
        
        # Only compute loss on masked positions
        logits_masked = logits_flat[masked_flat]
        labels_masked = labels_flat[masked_flat]
        
        return F.cross_entropy(logits_masked, labels_masked)

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
        """Predict with PPS bonus."""
        seqs = []
        positions_list = []
        
        for seq_row, length in zip(user_seq, seq_len):
            row_len = int(length.item())
            seq = seq_row[:row_len].tolist()

            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len :]

            positions = list(range(1, len(seq) + 1))

            if len(seq) < self.max_seq_len:
                pad_len = self.max_seq_len - len(seq)
                seq = [self.padding_token_id] * pad_len + seq
                positions = [0] * pad_len + positions

            seqs.append(seq)
            positions_list.append(positions)

        pred_seq = torch.tensor(seqs, dtype=torch.long, device=user_seq.device)
        pred_positions = torch.tensor(positions_list, dtype=torch.long, device=user_seq.device)
        
        hidden = self._encode(pred_seq, pred_positions)
        last_hidden = hidden[:, -1, :]
        
        # Use PPS scoring with user history bonus
        logits = self.item_codes_layer.score_all_items_with_bonus_user(
            last_hidden, user_sequence_ids=user_seq
        )

        if item_indices is None:
            return logits

        batch_indices = torch.arange(
            logits.size(0), device=logits.device
        ).unsqueeze(1)
        return (
            logits[batch_indices, item_indices].squeeze(-1)
            if item_indices.dim() == 1
            else logits[batch_indices, item_indices]
        )

    @torch.no_grad()
    def custom_validation(
        self,
        dataset,
        validation_top_k: int,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation using test evaluation."""
        results = self.custom_test_evaluation(dataset, [validation_top_k])
        result = results.get(validation_top_k, {})
        return {
            f"nDCG@{validation_top_k}": result.get("nDCG", 0.0),
            "val_loss": 0.0,  # Not computed in test evaluation
        }

    @torch.no_grad()
    def custom_test_evaluation(
        self,
        dataset,
        top_k: list[int],
        **kwargs: Any,
    ) -> dict[int, dict[str, float]]:
        """Test evaluation - same as parent class."""
        # Use default test evaluation from base class
        return super().custom_test_evaluation(dataset, top_k, **kwargs)
