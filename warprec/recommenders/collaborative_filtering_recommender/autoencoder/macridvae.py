# pylint: disable = R0801, E1102
from typing import Tuple, Any, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z using reparameterization trick."""

    def forward(self, z_mean: Tensor, z_log_var: Tensor) -> Tensor:
        """Forward pass of the sampler.

        Args:
            z_mean (Tensor): The mean value.
            z_log_var (Tensor): The log variance value.

        Returns:
            Tensor: The sampled mean value.
        """
        if self.training:
            epsilon = torch.randn_like(z_log_var)
            return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        return z_mean


class MacridEncoder(nn.Module):
    """Encoder module for MacridVAE.

    It processes the input interaction matrix masked by concept probabilities.
    To be efficient, it processes (Batch * K) items in parallel.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        layers: list[Any] = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.Tanh())
            curr_dim = h_dim

        self.dense_proj = nn.Sequential(*layers)
        self.dense_mean = nn.Linear(curr_dim, latent_dim)
        self.dense_log_var = nn.Linear(curr_dim, latent_dim)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass of the MacridEncoder.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Mean value and log variance value.
        """
        # Normalize input as per MacridVAE paper (L2 norm)
        i_normalized = F.normalize(inputs, p=2, dim=1)
        i_drop = self.dropout(i_normalized)

        x = self.dense_proj(i_drop)

        z_mean = self.dense_mean(x)
        # Normalize mean to hypersphere
        z_mean = F.normalize(z_mean, p=2, dim=1)

        z_log_var = self.dense_log_var(x)

        return z_mean, z_log_var


@model_registry.register(name="MacridVAE")
class MacridVAE(IterativeRecommender):
    """Implementation of MacridVAE algorithm from
        Learning Disentangled Representations for Recommendation (NeurIPS 2019).

    For further details, please refer to the `paper <https://arxiv.org/abs/1910.14238>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of user and item.
        encoder_hidden_dims (List[int]): The hidden dimensions of the encoder MLP.
        k_fac (int): The number of macro concepts (K).
        tau (float): The temperature for cosine similarity.
        corruption (float): The input corruption rate.
        nogb (bool): If True, use Softmax instead of Gumbel-Soft
        std (float): The standard deviation for reparameterization.
        anneal_cap (float): The maximum value for KL annealing.
        total_anneal_steps (int): The number of annealing steps.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The weight decay for optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition (Same as MultiVAE)
    DATALOADER_TYPE = DataLoaderType.INTERACTION_LOADER

    # Model hyperparameters
    embedding_size: int
    encoder_hidden_dims: List[int]
    k_fac: int
    tau: float
    corruption: float
    nogb: bool
    std: float
    anneal_cap: float
    total_anneal_steps: int
    reg_weight: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float

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

        # Store the training matrix for prediction
        self.train_matrix = interactions.get_sparse()

        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.k_embedding = nn.Embedding(self.k_fac, self.embedding_size)

        # Macrid Encoder
        self.encoder = MacridEncoder(
            input_dim=self.n_items,
            hidden_dims=self.encoder_hidden_dims,
            latent_dim=self.embedding_size,
            dropout_rate=self.corruption,
        )

        # Sampling layer
        self.sampling = Sampling()

        # Initialize weights and regularization
        self.reg_loss = EmbLoss()
        self.apply(self._init_weights)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_interaction_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def train_step(self, batch: Any, epoch: int, *args: Any, **kwargs: Any) -> Tensor:
        rating_matrix = batch[0]

        beta_anneal = (
            min(self.anneal_cap * epoch / self.total_anneal_steps, self.anneal_cap)
            if self.total_anneal_steps > 0
            else self.anneal_cap
        )

        logits, z_mean, z_log_var = self.forward(rating_matrix)

        # Calculate Reconstruction loss
        recon_loss = -(F.log_softmax(logits, dim=1) * rating_matrix).sum(dim=1).mean()

        # KL Divergence (Corrected for Prior N(0, sigma_0^2))
        var_prior = self.std**2 + 1e-10  # Small epsilon for numerical stability

        # Term 1: log(var_prior / var_posterior) = log(var_prior) - z_log_var
        term1 = torch.log(torch.tensor(var_prior, device=self.device)) - z_log_var

        # Term 2: (var_posterior + mu^2) / var_prior
        term2 = (z_log_var.exp() + z_mean.pow(2)) / var_prior

        # KL = 0.5 * sum( term1 + term2 - 1 )
        kl_loss = beta_anneal * (0.5 * torch.sum(term1 + term2 - 1, dim=[1, 2]).mean())

        # Calculate L2 loss
        reg_tensors = []
        for name, param in self.encoder.named_parameters():  # Include encoder params
            if name.endswith("weight"):
                reg_tensors.append(param)

        reg_loss = self.reg_weight * self.reg_loss(
            self.item_embedding.weight, self.k_embedding.weight * reg_tensors
        )

        return recon_loss + kl_loss + reg_loss

    def forward(self, rating_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass handling Macro-Disentanglement logic.

        Args:
            rating_matrix (Tensor): (Batch, n_items)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                logits: (Batch, n_items) - Log probabilities of reconstruction
                z_mean: (Batch, K, D)
                z_log_var: (Batch, K, D)
        """
        batch_size = rating_matrix.size(0)

        # Concept Assignment (Macro)
        # Normalize embeddings for Cosine Similarity
        items_norm = F.normalize(self.item_embedding.weight, dim=1)  # (M, D)
        cores_norm = F.normalize(self.k_embedding.weight, dim=1)  # (K, D)

        # Similarity: (M, K)
        cates_logits = torch.matmul(items_norm, cores_norm.t()) / self.tau

        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            if self.training:
                cates = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            else:
                cates = torch.softmax(cates_logits, dim=-1)

        # Encoder (Vectorized)
        # We mask the input rating matrix with concept probabilities.
        # Input: (B, M). Cates: (M, K).
        # We want (B, K, M) where x_{u,k} = x_u * c_k

        # (B, 1, M) * (1, K, M) -> (B, K, M)
        x_k = rating_matrix.unsqueeze(1) * cates.t().unsqueeze(0)

        # Flatten to (B*K, M) for parallel processing in MLP
        x_k_flat = x_k.reshape(batch_size * self.k_fac, self.n_items)

        # Encode
        z_mean_flat, z_log_var_flat = self.encoder(x_k_flat)

        # Reshape back to (B, K, D)
        z_mean = z_mean_flat.view(batch_size, self.k_fac, -1)
        z_log_var = z_log_var_flat.view(batch_size, self.k_fac, -1)

        # Sample z
        z = self.sampling(z_mean, z_log_var)  # (B, K, D)

        # Decoder
        # Reconstruct based on cosine similarity between z_u^k and h_i
        # z: (B, K, D), items: (M, D)
        z_norm = F.normalize(z, dim=-1)

        # (B, K, D) @ (D, M) -> (B, K, M)
        concept_scores = torch.matmul(z_norm, items_norm.t()) / self.tau

        # Weight by concept probability c_{i,k}
        # (B, K, M) * (1, K, M) -> (B, K, M)
        weighted_scores = torch.exp(concept_scores) * cates.t().unsqueeze(0)

        # Sum over concepts K -> (B, M)
        probs = weighted_scores.sum(dim=1)

        # Logits for numerical stability in loss
        logits = torch.log(probs + 1e-10)

        return logits, z_mean, z_log_var

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the the encoder and decoder modules.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Compute predictions and convert to Tensor
        train_batch = self.train_matrix[user_indices.tolist(), :].toarray()
        predictions, _, _ = self.forward(
            torch.from_numpy(train_batch).float().to(self.device)
        )

        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(
                max=self.n_items - 1
            ),  # [batch_size, pad_seq]
        )
