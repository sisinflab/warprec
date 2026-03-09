# pylint: disable=R0801,E1102,W0221,C0103,W0613,W0235,R0902
from collections import defaultdict
import os
from typing import Any, Optional
import random

import numpy as np
import torch
import narwhals as nw
from torch import Tensor
from torch.utils.data import DataLoader
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.data.entities import Interactions, Sessions
from warprec.utils.registry import model_registry
import mmh3
import os


class MaskingCollator:
    """Collator per SASRec train/val/test."""

    def __init__(
        self,
        user_actions: dict[int, list[tuple[Any, int, float]]],
        sequence_length: int,
        val_users: set[int],
        pad_id: int,
        mode: str = "train",
        border_timestamp: Optional[Any] = None,
        val_ndcg_at: Optional[int] = None,
        n_items: Optional[int] = None,
    ) -> None:
        self.val_users = val_users
        self.sequence_length = sequence_length
        self.user_actions = user_actions
        self.pad_id = pad_id
        self.mode = mode
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
                labels = seq.clone()

                if len(seq) < self.sequence_length:
                    pad = torch.tensor(
                        [self.ignore_val] * (self.sequence_length - len(seq)),
                        requires_grad=False,
                    )
                    labels = torch.cat([pad, labels], dim=0)

                labels_all.append(labels[1:])  # Labels are shifted

            elif self.is_validation:
                seq = [
                    x[1]
                    for x in user_hist
                    if x[2] > 0 and x[0] < self.border_timestamp
                ]

                if len(seq) > self.sequence_length:
                    seq = seq[-self.sequence_length :]

                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)
                label = seq[-1].clone()

                seq_after_ts = torch.tensor(
                    [
                        (x[1], x[2])
                        for x in user_hist
                        if x[0] >= self.border_timestamp
                    ],
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

                if len(seq) > self.sequence_length:
                    seq = seq[-self.sequence_length :]

                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)

            if len(seq) < self.sequence_length:
                pad = torch.tensor(
                    [self.pad_id] * (self.sequence_length - len(seq)),
                    requires_grad=False,
                )
                seq = torch.cat([pad, seq], dim=0)

            seqs.append(seq)

        if self.is_test:
            return {"seq": torch.stack(seqs)}
        if self.is_train:
            negatives = torch.randint(
                low=0,
                high=self.pad_id,
                size=(len(batch), self.sequence_length - 1),
                requires_grad=False,
            )
            return {
                "seq": torch.stack(seqs),
                "labels": torch.stack(labels_all),
                "negatives": negatives,
            }
        # Validation
        negatives = torch.randint(
            low=0, high=self.pad_id, size=(len(batch),), requires_grad=False
        )
        return {
            "seq": torch.stack(seqs),
            "labels": torch.stack(labels_all),
            "negatives": negatives,
            "ratings": torch.stack(val_ratings_all),
            "not_null_seqs_after_ts_idxs": torch.tensor(
                val_not_null_seqs_after_ts_idxs,
                dtype=torch.long,
                requires_grad=False,
            ),
        }


@model_registry.register(name="SASRec_gts")
class SASRec_gts(IterativeRecommender, SequentialRecommenderUtils):
    """Port di SASRec_gts il piu' vicino possibile all'implementazione originale."""

    DATALOADER_TYPE = None

    embedding_size: int
    n_layers: int
    n_heads: int
    dropout_prob: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    max_seq_len: int
    max_steps_per_epoch: int
    val_batch_size: int
    val_ndcg_at: int
    early_stop_epochs: int

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
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_device = torch.device("cpu")
        self.max_steps_per_epoch = params.get("max_steps_per_epoch", 128)
        self.val_batch_size = params.get("val_batch_size", self.batch_size)
        self.val_ndcg_at = params.get("val_ndcg_at", 10)
        self.early_stop_epochs = params.get("early_stop_epochs", 500)
        self.positions = torch.arange(1, self.val_ndcg_at + 1, dtype=torch.float)
        self.ndcg_discounts = torch.unsqueeze(
            1 / torch.log2(self.positions + 1), 0
        )

        # Initialize SASRec model
        self.item_embedding = torch.nn.Embedding(self.n_items + 2, self.embedding_size)
        self.position_embedding = torch.nn.Embedding(self.max_seq_len, self.embedding_size)
        self.embeddings_dropout = torch.nn.Dropout(self.dropout_prob)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    self.embedding_size,
                    self.n_heads,
                    self.embedding_size,
                    self.dropout_prob,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.seq_norm = torch.nn.LayerNorm(self.embedding_size)
        self.output_embedding = torch.nn.Embedding(self.n_items + 2, self.embedding_size)

        self.user_actions: dict[int, list[tuple[Any, int, float]]] = defaultdict(list)
        self.val_users_internal: set[int] = set()
        self.val_border_timestamp: Optional[Any] = None

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
                key=lambda x: (x[0], mmh3.hash(f"{x[1]}_{user_id}"))
            )

    def set_validation_context(self, dataset) -> None:
        """Prepara user_actions, val_users e border timestamp come nel codice originale."""
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

        if hasattr(dataset, 'test_set') and dataset.test_set is not None:
            self._append_actions_from_interactions(
                interactions=dataset.test_set,
                user_mapping=train_session._umap,
                item_mapping=train_session._imap,
                timestamp_label=train_session.timestamp_label,
                rating_label=dataset.test_set.rating_label,
            )
            
            test_df = dataset.test_set.get_df()
            if train_session.timestamp_label in test_df.columns:
                self.val_border_timestamp = test_df.select(
                    nw.col(train_session.timestamp_label).min()
                ).item()

        self._sort_actions()

    def _ensure_train_context(
        self,
        interactions: Interactions,
        sessions: Sessions,
    ) -> None:
        if self.user_actions:
            return

        self.user_actions = defaultdict(list)
        self._append_actions_from_sessions(sessions)
        self._sort_actions()

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

        collator = MaskingCollator(
            user_actions=self.user_actions,
            sequence_length=self.max_seq_len,
            val_users=self.val_users_internal,
            pad_id=self.padding_token_id,
            mode=mode,
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

    def sasrec_forward(self, input_seq: Tensor) -> tuple[Tensor, list]:
        """Forward pass del modello SASRec."""
        seq = self.item_embedding(input_seq.long())
        mask = (input_seq != self.padding_token_id).float().unsqueeze(-1)

        bs = seq.size(0)
        positions = (
            torch.arange(seq.shape[1])
            .unsqueeze(0)
            .repeat(bs, 1)
            .to(input_seq.device)
        )
        pos_embeddings = self.position_embedding(positions)[: input_seq.size(0)]
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        attentions = []
        for block in self.transformer_blocks:
            seq, attention = block(seq, mask)
            attentions.append(attention)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    def get_logits(self, model_out: Tensor) -> Tensor:
        """Calcola i logits da output embeddings."""
        scores = torch.einsum("bse, de -> bsd", model_out, self.output_embedding.weight)
        scores[:, :, self.n_items :] = float("-inf")
        return scores.to(torch.float64)

    def custom_train_loop(
        self,
        dataset,
        validation_top_k: int,
        device: str | torch.device,
    ):
        """Replica del loop custom originale con validazione interna ed early stopping."""
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

                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad()

                seq = batch["seq"]
                labels = batch["labels"]
                negatives = batch["negatives"]

                last_hidden_state, _ = self.sasrec_forward(seq)

                # Loss computation
                pos_neg_concat = torch.cat(
                    [labels.unsqueeze(-1), negatives.unsqueeze(-1)], dim=-1
                )
                pos_neg_concat[pos_neg_concat < 0] = self.padding_token_id + 1

                logits = torch.einsum(
                    "bse, bsne -> bsn",
                    last_hidden_state[:, :-1, :],
                    self.output_embedding(pos_neg_concat),
                )

                gt = torch.zeros_like(logits)
                gt[:, :, 0] = 1
                mask = (seq[:, :-1] != self.padding_token_id).float()

                loss_per_element = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.to(torch.float64), gt, reduction="none"
                    ).mean(-1)
                    * mask
                )
                loss = loss_per_element.sum() / mask.sum()

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

    def forward(self, item_seq: Tensor) -> Tensor:
        """Forward pass."""
        hidden, _ = self.sasrec_forward(item_seq)
        return self.get_logits(hidden)

    def train_step(self, batch: Any, *args: Any, **kwargs: Any) -> Tensor:
        """Training step."""
        seq = batch["seq"]
        labels = batch["labels"]
        negatives = batch["negatives"]

        last_hidden_state, _ = self.sasrec_forward(seq)

        pos_neg_concat = torch.cat(
            [labels.unsqueeze(-1), negatives.unsqueeze(-1)], dim=-1
        )
        pos_neg_concat[pos_neg_concat < 0] = self.padding_token_id + 1

        logits = torch.einsum(
            "bse, bsne -> bsn",
            last_hidden_state[:, :-1, :],
            self.output_embedding(pos_neg_concat),
        )

        gt = torch.zeros_like(logits)
        gt[:, :, 0] = 1
        mask = (seq[:, :-1] != self.padding_token_id).float()

        loss_per_element = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits.to(torch.float64), gt, reduction="none"
            ).mean(-1)
            * mask
        )
        return loss_per_element.sum() / mask.sum()

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
        """Predict scores."""
        # Prepare sequences
        seqs = []
        for seq_row, length in zip(user_seq, seq_len):
            row_len = int(length.item())
            seq = seq_row[:row_len].tolist()

            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len :]

            if len(seq) < self.max_seq_len:
                pad_len = self.max_seq_len - len(seq)
                seq = [self.padding_token_id] * pad_len + seq

            seqs.append(seq)

        pred_seq = torch.tensor(seqs, dtype=torch.long, device=user_seq.device)
        hidden, _ = self.sasrec_forward(pred_seq)
        logits = self.get_logits(hidden)
        last_logits = logits[:, -1, :]

        if item_indices is None:
            return last_logits

        batch_indices = torch.arange(
            last_logits.size(0), device=last_logits.device
        ).unsqueeze(1)
        return (
            last_logits[batch_indices, item_indices].squeeze(-1)
            if item_indices.dim() == 1
            else last_logits[batch_indices, item_indices]
        )

    @torch.no_grad()
    def recommend_impl(self, batch: dict[str, Tensor], limit: int, mode: str):
        """Recommend implementation."""
        if mode == "val":
            seq = batch["seq"]
            labels = batch["labels"]
            negatives = batch["negatives"]
            ratings = batch["ratings"]
            retain_idxs = batch["not_null_seqs_after_ts_idxs"].tolist()

            if not retain_idxs:
                return {
                    "losses": torch.tensor([0.0]),
                    "items_for_ndcg": torch.zeros((1, limit), dtype=torch.long),
                    "ratings": torch.zeros((1, ratings.shape[1])),
                }

            ratings = ratings[retain_idxs]
            last_hidden_state, _ = self.sasrec_forward(seq)
            logits = self.get_logits(last_hidden_state)

            # Loss computation
            all_logits_for_loss = logits[:, -2, :]
            pos_neg_concat = torch.cat(
                [labels.unsqueeze(-1), negatives.unsqueeze(-1)], dim=-1
            )
            logits_for_loss = all_logits_for_loss.gather(1, pos_neg_concat)

            gt = torch.zeros_like(logits_for_loss)
            gt[:, 0] = 1

            loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_for_loss, gt, reduction="none"
            ).mean(-1)

            # NDCG recommendations
            logits_for_ndcg = logits[retain_idxs, -1, :]
            top_k = torch.topk(logits_for_ndcg, limit, dim=1)

            return {
                "losses": loss_per_element,
                "items_for_ndcg": top_k.indices.cpu(),
                "ratings": ratings,
            }

        if mode == "test":
            seq = batch["seq"].to(self.device)
            last_hidden_state, _ = self.sasrec_forward(seq)
            logits = self.get_logits(last_hidden_state)[:, -1, :]
            top_k = torch.topk(logits, limit, dim=1)

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
        """Validazione custom per riprodurre loss/NDCG del codice originale."""
        if not self.val_users_internal or self.val_border_timestamp is None:
            return {f"nDCG@{validation_top_k}": float("-inf"), "val_loss": float("inf")}

        current_device = self.device
        self.eval()
        self.to(self.eval_device)

        val_collator = MaskingCollator(
            user_actions=self.user_actions,
            sequence_length=self.max_seq_len,
            val_users=self.val_users_internal,
            pad_id=self.padding_token_id,
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
            batch = {
                key: value.to(self.eval_device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
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

            losses.append(recommendations["losses"])

        self.to(current_device)

        ndcg_value = torch.cat(ndcgs).mean().item() if ndcgs else float("-inf")
        loss_value = torch.cat(losses).mean().item() if losses else float("inf")
        return {
            f"nDCG@{validation_top_k}": ndcg_value,
            "val_loss": loss_value,
        }

    @torch.no_grad()
    def custom_test_evaluation(
        self,
        dataset,
        top_k: list[int],
        **kwargs: Any,
    ) -> dict[int, dict[str, float]]:
        """Test evaluation aligned with the original validation logic."""
        self.set_validation_context(dataset)

        if not self.val_users_internal or self.val_border_timestamp is None:
            return {
                k: {"nDCG": 0.0, "Precision": 0.0, "Recall": 0.0, "HitRate": 0.0}
                for k in top_k
            }

        current_device = self.device
        self.eval()
        self.to(self.eval_device)

        top_k = sorted(top_k)
        max_top_k = max(top_k)
        val_collator = MaskingCollator(
            user_actions=self.user_actions,
            sequence_length=self.max_seq_len,
            val_users=self.val_users_internal,
            pad_id=self.padding_token_id,
            mode="val",
            border_timestamp=self.val_border_timestamp,
            val_ndcg_at=max_top_k,
            n_items=self.n_items,
        )

        val_loader = DataLoader(
            list(self.val_users_internal),
            batch_size=self.val_batch_size,
            collate_fn=val_collator,
            shuffle=False,
        )

        ndcgs = {k: [] for k in top_k}
        precisions = {k: [] for k in top_k}
        recalls = {k: [] for k in top_k}
        hitrates = {k: [] for k in top_k}

        discounts = {
            k: torch.unsqueeze(
                1 / torch.log2(torch.arange(1, k + 1, dtype=torch.float) + 1), 0
            )
            for k in top_k
        }

        for batch in val_loader:
            batch = {
                key: value.to(self.eval_device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            recommendations = self.recommend_impl(batch, max_top_k, "val")
            items_for_ndcg = recommendations["items_for_ndcg"]
            ratings = recommendations["ratings"]

            if items_for_ndcg.numel() == 0:
                continue

            positive_ratings = ratings.clone().float()
            positive_ratings[positive_ratings < 0] = 0
            relevant_count = (positive_ratings > 0).sum(dim=1).float()
            valid_mask = relevant_count > 0

            if not valid_mask.any():
                continue

            for k in top_k:
                rec_items = items_for_ndcg[:, :k]
                gathered_scores = torch.gather(positive_ratings, 1, rec_items)
                hits = (gathered_scores > 0).sum(dim=1).float()

                dcg = torch.sum(gathered_scores * discounts[k], dim=1)
                ideal_scores = torch.topk(positive_ratings, k, dim=1).values
                idcg = torch.sum(ideal_scores * discounts[k], dim=1).clamp(min=1e-10)
                ndcg = torch.nan_to_num(dcg / idcg)[valid_mask]

                precision = (hits / k)[valid_mask]
                recall = hits[valid_mask] / relevant_count[valid_mask]
                hitrate = (hits[valid_mask] > 0).float()

                ndcgs[k].append(ndcg.cpu())
                precisions[k].append(precision.cpu())
                recalls[k].append(recall.cpu())
                hitrates[k].append(hitrate.cpu())

        self.to(current_device)

        results = {}
        for k in top_k:
            results[k] = {
                "nDCG": torch.cat(ndcgs[k]).mean().item() if ndcgs[k] else 0.0,
                "Precision": (
                    torch.cat(precisions[k]).mean().item() if precisions[k] else 0.0
                ),
                "Recall": torch.cat(recalls[k]).mean().item() if recalls[k] else 0.0,
                "HitRate": (
                    torch.cat(hitrates[k]).mean().item() if hitrates[k] else 0.0
                ),
            }
        return results


class TransformerBlock(torch.nn.Module):
    """Transformer block for SASRec."""

    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ff_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ff_dim, embedding_dim),
            torch.nn.Dropout(dropout),
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        # Self-attention with causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float("-inf"), diagonal=1
        )

        attn_output, attn_weights = self.attention(
            x, x, x, attn_mask=causal_mask, need_weights=True
        )
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        x = x * mask

        return x, attn_weights
