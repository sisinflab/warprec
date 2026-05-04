# pylint: disable = R0801, E1102
from typing import Any, Optional, no_type_check

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry


@model_registry.register(name="iALS")
class iALS(Recommender):
    """Implementation of iALS model from
    "Revisiting the Performance of iALS on Item Recommendation Benchmarks" in RecSys 2022.

    The paper revisits iALS for top-n recommendation using a binary
    user-item matrix, an all-pairs unobserved loss weighted by ``alpha0``,
    and a frequency-scaled L2 regularizer controlled by ``nu``.

    The original implementation of iALS is described in
    "Collaborative Filtering for Implicit Feedback Datasets" in ICDM 2008, and
    optimizes a different confidence-weighted objective. The 2022 paper shows
    that the original iALS performs poorly on standard benchmarks, and that the
    modified iALS objective is competitive with state-of-the-art methods.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        factors (int): Latent factor dimensionality.
        alpha0 (float): Weight of the all-pairs unobserved loss term.
        reg (float): L2 regularization weight (lambda).
        n_iterations (int): Number of full ALS sweeps.
        nu (float): Frequency scaling exponent for the regularizer.
    """

    factors: int
    alpha0: float
    reg: float
    n_iterations: int
    nu: float

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

        X = interactions.get_sparse().tocsr()  # [n_users, n_items]
        n_users, n_items = X.shape

        # The paper defines S as a set of observed user-item pairs. Any
        # non-zero interaction is therefore treated as a binary positive.
        S = X.copy().astype(np.float64)
        S.data = np.ones_like(S.data, dtype=np.float64)

        # -- Initialise latent factors ----------------------------------------
        # Appendix A.2, Eq. 6: sigma = sigma_star / sqrt(d), with sigma_star=0.1
        rng = np.random.default_rng(seed)
        sigma = 0.1 / np.sqrt(self.factors)
        user_factors = rng.normal(0.0, sigma, size=(n_users, self.factors)).astype(
            np.float64
        )
        item_factors = rng.normal(0.0, sigma, size=(n_items, self.factors)).astype(
            np.float64
        )

        # -- ALS iterations ---------------------------------------------------
        # The paper optimizes the binary objective by alternating least squares.
        for _ in range(self.n_iterations):
            user_factors = self._als_step(
                item_factors, user_factors, S, self.reg, self.alpha0, self.nu
            )

            item_factors = self._als_step(
                user_factors,
                item_factors,
                S.T.tocsr(),
                self.reg,
                self.alpha0,
                self.nu,
            )

        # Store learned factors as buffers so checkpoints preserve the model.
        self.register_buffer("user_factors", torch.from_numpy(user_factors).float())
        self.register_buffer("item_factors", torch.from_numpy(item_factors).float())

    @staticmethod
    def _als_step(
        fixed_factors: np.ndarray,
        target_factors: np.ndarray,
        observed: csr_matrix,
        reg: float,
        alpha0: float,
        nu: float,
    ) -> np.ndarray:
        """Compute one half of an ALS sweep for the paper's loss.

        For one entity e with observed opposite-side embeddings F_e, the
        paper's objective yields the normal equations:

            (alpha0 * F^T F + F_e^T F_e + lambda * #e^nu * I) x_e = sum(F_e)

        where #e = |obs(e)| + alpha0 * |opposite side|.

        Args:
            fixed_factors (np.ndarray): Factor matrix held fixed this step [K, f].
            target_factors (np.ndarray): Factor matrix to update [M, f].
            observed (csr_matrix): Sparse binary interaction matrix [M, K].
            reg (float): L2 regularization weight.
            alpha0 (float): All-pairs unobserved weight.
            nu (float): Frequency scaling exponent for the regularizer.
        Returns:
            np.ndarray: Updated target factor matrix [M, f].
        """
        n_entities = target_factors.shape[0]
        n_opposite_entities = fixed_factors.shape[0]
        f = fixed_factors.shape[1]

        # Shared all-pairs term from L_I in Eq. 4.
        FtF = alpha0 * (fixed_factors.T @ fixed_factors)  # [f, f]
        reg_I = np.eye(f, dtype=np.float64)

        updated = np.empty_like(target_factors)

        for u in range(n_entities):
            row = observed.getrow(u)
            indices = row.indices
            n_interactions = len(indices)

            if n_interactions == 0:
                # With no observed positives, the RHS is zero and the solution
                # is the all-zero vector.
                updated[u] = 0.0
                continue

            F_u = fixed_factors[indices]  # [n_u, f]

            freq_weight = n_interactions + alpha0 * n_opposite_entities
            reg_multiplier = freq_weight**nu
            A = FtF + (F_u.T @ F_u) + (reg * reg_multiplier) * reg_I

            # The observed term L_S contributes one copy of each positive
            # embedding to the RHS.
            rhs = F_u.sum(axis=0)  # [f]

            updated[u] = np.linalg.solve(A, rhs)

        return updated

    @no_type_check
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute predicted preference scores.

        The paper uses the standard matrix-factorization score
        p_hat_ui = x_u^T y_i.

        Args:
            user_indices (Tensor): Batch of user indices [batch_size].
            *args (Any): Additional positional arguments.
            item_indices (Optional[Tensor]): Optional item indices [batch_size, k] for sampled
                evaluation.  If ``None``, scores for all items are returned.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tensor: Score tensor [batch_size, n_items] or [batch_size, k].
        """
        user_indices = user_indices.to(self.user_factors.device)
        X_u = self.user_factors[user_indices]  # [batch_size, factors]

        if item_indices is None:
            return X_u @ self.item_factors.T  # [batch_size, n_items]

        item_indices = item_indices.to(self.item_factors.device).clamp(
            max=self.n_items - 1
        )
        selected_item_factors = self.item_factors[
            item_indices
        ]  # [batch_size, k, factors]
        return (X_u.unsqueeze(1) * selected_item_factors).sum(dim=-1)  # [batch_size, k]
