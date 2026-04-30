# pylint: disable = R0801, E1102
from typing import Any, Optional, cast

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry


@model_registry.register(name="iALS2008")
class iALS2008(Recommender):
    """Implementation of iALS model from
    "Collaborative Filtering for Implicit Feedback Datasets" in ICDM 2008.

    Decomposes the user-item implicit feedback matrix into user-factor and
    item-factor matrices via confidence-weighted alternating least squares.

    The model treats raw observations r_ui as indicators of *preference*
    (binary) and *confidence* (monotonic in r_ui), then minimizes a
    weighted squared-error objective with L2 regularization (Eq. 3).

    Closed-form factor updates (Eq. 4 & 5) are computed in __init__,
    exploiting the sparsity trick Y^T C^u Y = Y^T Y + Y^T (C^u - I) Y
    to achieve O(f^2 N + f^3 m) per sweep.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        factors (int): Latent factor dimensionality.
        alpha (float): Confidence scaling constant.
        reg (float): L2 regularization weight (lambda).
        n_iterations (int): Number of full ALS sweeps.
        confidence_type (str): "linear" or "log".
        epsilon (float): Epsilon for the log confidence variant.
    """

    factors: int
    alpha: float
    reg: float
    n_iterations: int
    confidence_type: str
    epsilon: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        X = interactions.get_sparse()  # [n_users, n_items]
        n_users, n_items = X.shape

        # -- Compute confidence matrix entries (only non-zero stored) ---------
        # Sec. 4: c_ui = 1 + alpha * r_ui  (linear)
        # Eq. 6:  c_ui = 1 + alpha * log(1 + r_ui / epsilon)  (log)
        if self.confidence_type == "log":
            # Eq. 6 — log confidence variant (used in the paper's experiments)
            C_minus_I = X.copy()
            C_minus_I.data = self.alpha * np.log(1.0 + C_minus_I.data / self.epsilon)
        else:
            # Sec. 4 — linear confidence (paper's primary formulation)
            C_minus_I = X.copy()
            C_minus_I.data = self.alpha * C_minus_I.data

        # -- Preference matrix (binary) --------------------------------------
        # Sec. 4: p_ui = 1 if r_ui > 0, else 0
        P = X.copy()
        P.data = np.ones_like(P.data, dtype=np.float64)

        # -- Initialise latent factors ----------------------------------------
        # ASSUMPTION: Small random normal initialization — the paper does not
        # specify an initialization strategy.  This is standard practice for
        # ALS-based matrix factorization.
        rng = np.random.default_rng(seed)
        user_factors = rng.normal(0.0, 0.01, size=(n_users, self.factors)).astype(
            np.float64
        )
        item_factors = rng.normal(0.0, 0.01, size=(n_items, self.factors)).astype(
            np.float64
        )

        # -- ALS iterations ---------------------------------------------------
        # Sec. 4: "We employ a few sweeps of paired recomputation of user-
        # and item-factors, till they stabilize.  A typical number of
        # sweeps is 10."
        for _ in range(self.n_iterations):
            # --- Update user factors (Eq. 4) ---------------------------------
            # x_u = (Y^T C^u Y + lambda I)^{-1} Y^T C^u p(u)
            # Sparsity trick: Y^T C^u Y = Y^T Y + Y^T (C^u - I) Y
            user_factors = self._als_step(
                item_factors, user_factors, C_minus_I, P, self.reg
            )

            # --- Update item factors (Eq. 5) ---------------------------------
            # y_i = (X^T C^i X + lambda I)^{-1} X^T C^i p(i)
            # Transpose so items are rows
            item_factors = self._als_step(
                user_factors, item_factors, C_minus_I.T.tocsr(), P.T.tocsr(), self.reg
            )

        # Store learned factors as buffers so checkpoints preserve the model.
        self.register_buffer("user_factors", torch.from_numpy(user_factors).float())
        self.register_buffer("item_factors", torch.from_numpy(item_factors).float())

    @staticmethod
    def _als_step(
        fixed_factors: np.ndarray,
        target_factors: np.ndarray,
        C_minus_I: csr_matrix,
        P: csr_matrix,
        reg: float,
    ) -> np.ndarray:
        """Compute one half of an ALS sweep (Eq. 4 / Eq. 5).

        For each row *u* of the target factor matrix, solve:
            target_u = (F^T C^u F + lambda I)^{-1} F^T C^u p(u)

        where F is ``fixed_factors``, exploiting the decomposition:
            F^T C^u F = F^T F + F^T (C^u - I) F

        Running time per target entity: O(f^2 n_u + f^3) where n_u is
        the number of non-zero entries for that entity.

        Args:
            fixed_factors (np.ndarray): Factor matrix held fixed this step [K, f].
            target_factors (np.ndarray): Factor matrix to update [M, f].
            C_minus_I (csr_matrix): Sparse matrix of (c_ui - 1) values [M, K].
            P (csr_matrix): Sparse binary preference matrix [M, K].
            reg (float): L2 regularization weight.
        Returns:
            np.ndarray: Updated target factor matrix [M, f].
        """
        n_entities = target_factors.shape[0]
        f = fixed_factors.shape[1]

        # Precompute F^T F — O(f^2 K), shared across all entities
        # Sec. 4: "Y^T Y is independent of u and was already precomputed."
        FtF = fixed_factors.T @ fixed_factors  # [f, f]
        reg_I = reg * np.eye(f, dtype=np.float64)

        updated = np.empty_like(target_factors)

        for u in range(n_entities):
            # Indices and values of non-zero entries for entity u
            # These correspond to items with r_ui > 0
            row = C_minus_I.getrow(u)
            indices = row.indices
            c_minus_1 = row.data  # c_ui - 1 for non-zero entries

            if len(indices) == 0:
                # No interactions — factor determined purely by regularization
                # ASSUMPTION: For entities with no interactions, factors are
                # set to zero (regularization dominates).
                updated[u] = 0.0
                continue

            # F_u = fixed_factors[indices]  — the factor rows for observed items
            F_u = fixed_factors[indices]  # [n_u, f]

            # Sec. 4 sparsity trick:
            # F^T C^u F = F^T F + F^T (C^u - I) F
            # F^T (C^u - I) F = F_u^T diag(c_ui - 1) F_u
            FtCuF = FtF + (F_u.T * c_minus_1) @ F_u  # [f, f]

            # A = F^T C^u F + lambda I
            A = FtCuF + reg_I  # [f, f]

            # Sec. 4: F^T C^u p(u)
            # C^u p(u) has only n_u non-zero entries: c_ui * p_ui = c_ui
            # (since p_ui = 1 for observed, 0 otherwise)
            # c_ui = (c_ui - 1) + 1 = c_minus_1 + 1
            p_u = P.getrow(u).toarray().ravel()[indices]  # p_ui values (all 1)
            c_u = c_minus_1 + 1.0  # full c_ui for observed entries
            rhs = F_u.T @ (c_u * p_u)  # [f]

            # Eq. 4: x_u = A^{-1} rhs
            updated[u] = np.linalg.solve(A, rhs)

        return updated

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute predicted preference scores.

        Sec. 5: p_hat_ui = x_u^T y_i

        Args:
            user_indices (Tensor): Batch of user indices [batch_size].
            *args (Any): Additional positional arguments.
            item_indices (Optional[Tensor]): Optional item indices [batch_size, k] for sampled
                evaluation.  If ``None``, scores for all items are returned.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tensor: Score tensor [batch_size, n_items] or [batch_size, k].
        """
        # Sec. 5: p_hat_ui = x_u^T y_i
        users = user_indices.cpu().numpy()
        user_factors = cast(Tensor, self.user_factors)
        item_factors = cast(Tensor, self.item_factors)
        X_u = user_factors[users]  # [batch_size, factors]
        predictions = X_u @ item_factors.T  # [batch_size, n_items]

        if item_indices is None:
            return predictions  # [batch_size, n_items]

        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(max=self.n_items - 1),
        )  # [batch_size, k]
