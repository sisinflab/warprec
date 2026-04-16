# pylint: disable = R0801, E1102
from typing import Any, Optional

import numpy as np
import torch
import narwhals as nw
from scipy.sparse import coo_matrix
from torch import Tensor

from warprec.data.entities import Sessions
from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.registry import model_registry


@model_registry.register(name="STAN")
class STAN(Recommender, SequentialRecommenderUtils):
    """Implementation of STAN model from
        Sequence and Time Aware Neighborhood for Session-basedRecommendations (SIGIR'19).

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        sessions (Sessions): Training sessions — the primary data source; we
            pull (flat_items, flat_users, user_offsets, timestamps) from it.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        k (int): Neighborhood size N (Section 4).
        lambda_1 (float): Eq. 3 decay.
        lambda_2 (float): Eq. 5 decay (seconds).
        lambda_3 (float): Eq. 7 decay.
        max_seq_len (int): Upper bound on current-session length (controls how much
            history the evaluator feeds to predict).
    """

    # Model hyperparameters
    k: int
    lambda_1: float
    lambda_2: float
    lambda_3: float
    max_seq_len: int

    def __init__(
        self,
        params: dict,
        info: dict,
        sessions: Sessions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # -----------------------------------------------------------------
        # Persist training session information (pre-computed, read-only)
        # -----------------------------------------------------------------
        # We treat each user's history as one past session
        # (WarpRec's session-based modelling convention).
        self._flat_items = sessions._flat_items.astype(np.int64)
        self._flat_users = sessions._flat_users.astype(np.int64)
        self._user_offsets = sessions._user_offsets.astype(np.int64)

        # -----------------------------------------------------------------
        # Session-level timestamp t(s_j) (most-recent event per session)
        # -----------------------------------------------------------------
        # Paper Eq. 5: w2 uses t(s) - t(s_j) where t(.) is "the timestamp of
        # the most recent item i_{j,l(s_j)} in s_j".
        self._session_timestamps = self._extract_session_timestamps(sessions)

        # -----------------------------------------------------------------
        # Binary session-by-item matrices (CSR and its CSC transpose)
        # -----------------------------------------------------------------
        # The CSR lets us compute Eq. 4's numerator for every candidate in a
        # single sparse matmul (P_cand @ s_w). The CSC is the paper's
        # inverted index from Section 5.3 footnote — the column slice for
        # item i gives the session-ids that contain i.
        csr, csc = self._build_session_item_matrices(
            self._flat_items, self._flat_users, self.n_users, self.n_items
        )
        self._session_item_csr = csr
        self._session_item_csc = csc

        # Eq. 4 denominator: |s_j| on the binary session vector = number of
        # unique items in s_j. Precomputed once here.
        self._session_unique_counts = np.asarray(
            self._session_item_csr.getnnz(axis=1), dtype=np.float64
        )

    @staticmethod
    def _extract_session_timestamps(sessions: Sessions) -> Optional[np.ndarray]:
        """Extract t(s_j) for each session (= each user in WarpRec).

        Returns an array of shape ``[n_users]`` where entry u holds the max
        timestamp observed for user u, or ``None`` if the Sessions entity has
        no timestamp column.
        """
        ts_label = sessions.timestamp_label
        df = sessions._get_processed_data()  # cached, mapped, sorted
        if ts_label not in df.columns:
            return None

        # The df is sorted by (user, timestamp) — so the last row of each
        # user's slice holds t(s_j). One vectorized gather replaces a
        # per-user Python loop.
        ts_all = df.select(nw.col(ts_label)).to_numpy().flatten().astype(np.float64)
        offsets = sessions._user_offsets
        n_users = len(offsets) - 1
        if ts_all.size == 0:
            return np.zeros(n_users, dtype=np.float64)

        lengths = offsets[1:] - offsets[:-1]
        # Clip so that empty-user entries (length == 0) land on a valid slot;
        # the np.where below overwrites them with 0 anyway.
        last_idx = np.clip(offsets[1:] - 1, 0, ts_all.size - 1)
        return np.where(lengths > 0, ts_all[last_idx], 0.0).astype(np.float64)

    @staticmethod
    def _build_session_item_matrices(
        flat_items: np.ndarray,
        flat_users: np.ndarray,
        n_users: int,
        n_items: int,
    ) -> tuple:
        """Build the binary session-by-item CSR and its CSC transpose.

        A session's vector in Eq. 4 is binary, so interaction-stream
        duplicates collapse to 1. This replaces the old per-interaction
        Python loop with a single COO → CSR conversion.

        Args:
            flat_items (np.ndarray): Flat array of item ids (one entry per interaction).
            flat_users (np.ndarray): Corresponding session (= user) ids, same length.
            n_users (int): Total number of sessions (CSR rows).
            n_items (int): Total number of items (CSR cols).

        Returns:
            tuple: (csr, csc) with shapes [n_users, n_items], binary.
        """
        # Defensive guard against out-of-range items (mirrors the old build).
        valid = (flat_items >= 0) & (flat_items < n_items)
        rows = flat_users[valid]
        cols = flat_items[valid]
        data = np.ones(rows.shape[0], dtype=np.float64)

        coo = coo_matrix((data, (rows, cols)), shape=(n_users, n_items))
        csr = coo.tocsr()
        csr.sum_duplicates()
        # Binarize: an item appears in a session either 0 or 1 times.
        csr.data = np.ones_like(csr.data)
        csc = csr.tocsc()
        return csr, csc

    def _get_user_items(self, user_id: int) -> np.ndarray:
        """Return the item sequence of user ``user_id`` in chronological order."""
        start, end = (
            int(self._user_offsets[user_id]),
            int(self._user_offsets[user_id + 1]),
        )
        return self._flat_items[start:end]

    def _compute_w1(self, current_seq: np.ndarray) -> np.ndarray:
        """Eq. 3 — w1(i, s) = exp((p(i, s) - l(s)) / lambda_1).

        Produces the real-valued weight vector for the current session. The
        last position has weight 1; earlier positions decay exponentially.
        """
        length = current_seq.shape[0]
        if length == 0:
            return np.empty(0, dtype=np.float64)
        # positions are 1-indexed in the paper (p(i, s) in [1, l(s)]).
        positions = np.arange(1, length + 1, dtype=np.float64)
        return np.exp((positions - length) / float(self.lambda_1))

    def _compute_w2(
        self,
        current_ts: float,
        neighbor_ids: np.ndarray,
    ) -> np.ndarray:
        """Eq. 5 — w2(s_j | s) = exp(-(t(s) - t(s_j)) / lambda_2).

        Returns a vector of shape ``[len(neighbor_ids)]``.
        """
        if self._session_timestamps is None:
            # ASSUMPTION: dataset without timestamps -> disable Factor-2.
            return np.ones(neighbor_ids.shape[0], dtype=np.float64)

        ts_neighbors = self._session_timestamps[neighbor_ids]
        # Paper states t(s) > t(s_j); we clip negatives to 0 so that a
        # neighbor session with ts == current_ts gets weight 1 (rather than
        # blowing up via a negative delta). This is consistent with the
        # paper's intention to "decay" the past.
        delta = np.clip(current_ts - ts_neighbors, a_min=0.0, a_max=None)
        return np.exp(-delta / float(self.lambda_2))

    def _compute_w3_and_items(
        self,
        current_items: np.ndarray,
        neighbor_seq: np.ndarray,
    ) -> tuple:
        """Eq. 7 — w3(i | s, n) = exp(-|p(i, n) - p(i*, n)| / lambda_3) * I_n(i).

        Here i* is the co-occurring item between s and n that is
        most recent in s (paper: "the item i* that occurs in both s and n,
        and is most recent w.r.t. s").

        Args:
            current_items (np.ndarray): Item sequence of the current session s.
            neighbor_seq (np.ndarray): Item sequence of the neighbor session n.

        Returns:
            tuple: (items, weights) — arrays of equal length holding the
                recommendable items in n and their per-item Eq. 7 weights.
                Returns empty arrays if no co-occurring item exists (in which
                case the neighbor contributes nothing, consistent with I_n(.)
                being zero for all items).
        """
        if neighbor_seq.size == 0 or current_items.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        # Find i* = most recent item in current_items that also appears in
        # neighbor_seq. Vectorized "in" check + take the last True index.
        mask = np.isin(current_items, neighbor_seq)
        hit = np.flatnonzero(mask)
        if hit.size == 0:
            # No co-occurring item -> I_n(i*) would be 0. Drop the neighbor.
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        i_star = int(current_items[hit[-1]])

        # Position of i* inside the neighbor session (1-indexed, paper
        # convention). If i* appears multiple times in n, use the LAST
        # occurrence.
        # ASSUMPTION: The paper defines p(i*, n) for a single position but
        # does not state which occurrence to use if i* repeats inside n.
        # Using the latest occurrence is consistent with "most recent" being
        # the operative choice throughout the paper (cf. Factor-1). We scan
        # the reversed array to avoid materializing a full index list.
        rev_hit = int((neighbor_seq[::-1] == i_star).argmax())
        pos_i_star = neighbor_seq.shape[0] - rev_hit

        # Eq. 7 for every item in n (I_n(i) == 1 here by construction because
        # we iterate items that are in n).
        positions = np.arange(1, neighbor_seq.shape[0] + 1, dtype=np.float64)
        weights = np.exp(-np.abs(positions - float(pos_i_star)) / float(self.lambda_3))
        return neighbor_seq.astype(np.int64), weights

    def _score_session(
        self,
        current_items: np.ndarray,
        current_ts: float,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Produce the STAN score vector over all items for one session.

        Implements Eqs. 4, 6, 8, 9 of the paper. Writes into the provided
        scores buffer (zeroed on entry) and returns it.

        Args:
            current_items (np.ndarray): Ordered item ids of the current
                session s (length l(s)).
            current_ts (float): t(s) — most-recent timestamp in s.
            scores (np.ndarray): Pre-allocated buffer of shape [n_items]
                that this method fills in-place.

        Returns:
            np.ndarray: The same scores buffer, now holding the score
                vector of shape [n_items].
        """
        scores.fill(0.0)
        length = current_items.shape[0]
        if length == 0:
            # Empty session — nothing to match. Return a zero vector; the
            # evaluator will still produce a ranking, equivalent to random
            # tie-breaking on an unseen user.
            return scores

        # ---- Eq. 3 & Eq. 4 ---------------------------------------------
        # s_w on the full item space. Duplicates in s accumulate their
        # Eq. 3 weights correctly via np.add.at.
        w1_vec = self._compute_w1(current_items)
        s_w = np.zeros(self.n_items, dtype=np.float64)
        np.add.at(s_w, current_items, w1_vec)

        unique_items = np.unique(current_items)

        # ---- Candidate neighbor set via inverted index -----------------
        # Paper Section 5.3 footnote: inverted index of item -> sessions.
        # Union across unique items in s via CSC column slices, then dedup.
        indptr = self._session_item_csc.indptr
        indices = self._session_item_csc.indices
        pieces = [indices[indptr[i] : indptr[i + 1]] for i in unique_items.tolist()]
        cand_ids = np.unique(np.concatenate(pieces))
        if cand_ids.size == 0:
            return scores

        # ---- Eq. 4 (sim1) vectorized over all candidates ---------------
        # Numerator:   sim1_num(s, s_j) = s_w . s_j = (P @ s_w)[j]  (binary s_j)
        # Denominator: sqrt(l(s)) * sqrt(l(s_j))   (unique cardinalities)
        p_cand = self._session_item_csr[cand_ids]
        dot = np.asarray(p_cand.dot(s_w)).ravel()
        cand_lengths = self._session_unique_counts[cand_ids]
        denom = float(np.sqrt(float(unique_items.shape[0]))) * np.sqrt(cand_lengths)
        sim1_scores = np.zeros(cand_ids.shape[0], dtype=np.float64)
        safe = denom > 0.0
        sim1_scores[safe] = dot[safe] / denom[safe]

        # ---- Eq. 5 & Eq. 6 (sim2) --------------------------------------
        w2_vec = self._compute_w2(current_ts, cand_ids)
        sim2_scores = sim1_scores * w2_vec

        # ---- Top-N neighborhood selection ------------------------------
        # Paper: "The neighborhood N(s) is then found by taking the top N
        # most similar sessions using the similarity measure sim2".
        n_nbrs = min(self.k, sim2_scores.shape[0])
        if n_nbrs <= 0:
            return scores
        # argpartition is O(n) vs. O(n log n) for argsort. We then refine.
        top_idx = np.argpartition(-sim2_scores, n_nbrs - 1)[:n_nbrs]
        # Drop neighbors with zero similarity (they cannot contribute).
        top_idx = top_idx[sim2_scores[top_idx] > 0.0]

        # ---- Eq. 7 & Eq. 8 & Eq. 9 -------------------------------------
        # scoreSTAN(i, s) = sum_{n in N(s)} sim2(s, n) * w3(i | s, n)
        for idx in top_idx:
            n_id = int(cand_ids[idx])
            n_seq = self._get_user_items(n_id)
            if n_seq.size == 0:
                continue
            items, w3_vec = self._compute_w3_and_items(current_items, n_seq)
            if items.size == 0:
                continue
            sim2_val = float(sim2_scores[idx])
            # np.add.at handles duplicates inside the neighbor session
            # correctly — Eq. 8/9 summation.
            np.add.at(scores, items, sim2_val * w3_vec)

        return scores

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        user_seq: Optional[Tensor] = None,
        seq_len: Optional[Tensor] = None,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute STAN scores for a batch of current sessions.

        Args:
            user_indices (Tensor): User indices for which to produce scores.
            *args (Any): List of arguments.
            user_seq (Optional[Tensor]): Padded sequences of item IDs for users to predict for.
            seq_len (Optional[Tensor]): Actual lengths of these sequences, before padding.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: Score matrix [batch_size, n_items] or
                [batch_size, n_samples].
        """
        batch_size = int(user_indices.shape[0])
        # Output allocated directly as float32 (the returned dtype); halves
        # memory bandwidth versus building a float64 copy first.
        all_scores = np.zeros((batch_size, self.n_items), dtype=np.float32)
        # Reusable float64 accumulation buffer — keeps Eq. 8/9 precision.
        scores_buf = np.zeros(self.n_items, dtype=np.float64)
        padding_idx = self.n_items

        user_indices_cpu = user_indices.detach().cpu().tolist()
        if user_seq is not None and seq_len is not None:
            user_seq_np = user_seq.detach().cpu().numpy()
            seq_len_np = seq_len.detach().cpu().numpy()
        else:
            user_seq_np = None
            seq_len_np = None

        for b, user_id in enumerate(user_indices_cpu):
            user_id = int(user_id)

            # Recover current session items (most-recent first order
            # preserved — the evaluator already truncates to max_seq_len).
            if user_seq_np is not None and seq_len_np is not None:
                real_len = int(seq_len_np[b])
                row = user_seq_np[b, :real_len]
                # Strip any padding ids defensively.
                current_items = row[row != padding_idx].astype(np.int64)
            else:
                current_items = self._get_user_items(user_id).astype(np.int64)

            # Current session timestamp = max timestamp observed for this
            # user (matches paper's t(s) definition). Uses cached array.
            if self._session_timestamps is not None:
                current_ts = float(self._session_timestamps[user_id])
            else:
                current_ts = 0.0  # disables Factor-2 (see _compute_w2)

            self._score_session(current_items, current_ts, scores_buf)
            all_scores[b, :] = scores_buf  # implicit float64 -> float32 cast

        predictions = torch.from_numpy(all_scores)

        if item_indices is None:
            return predictions  # [batch_size, n_items]

        # Sampled evaluation — gather only requested item slots.
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(max=self.n_items - 1),
        )
