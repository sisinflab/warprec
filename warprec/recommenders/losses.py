import torch
from torch import nn, Tensor


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking.

    For further details, check the `paper <https://arxiv.org/abs/1205.2618>`_.

    Args:
        gamma (float): Small value to avoid division by zero
    """

    def __init__(self, gamma: float = 1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        """Compute the BPR loss.

        Args:
            pos_score (Tensor): Positive item scores.
            neg_score (Tensor): Negative item scores.

        Returns:
            Tensor: The computed BPR loss.
        """
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class EmbeddingLoss(nn.Module):
    """EmbeddingLoss, regularization on embeddings.

    This loss aims to penalize the magnitude of embedding weights,
    promoting smaller weights and potentially preventing overfitting.
    It is based on the L1 or L2 norm of the embedding vectors.

    Args:
        norm (int): The norm to use for loss calculation.
                    Commonly 1 (L1) or 2 (L2).

    Raises:
        ValueError: If the norm is not 1 or 2.
    """

    def __init__(self, norm: int = 2):
        super(EmbeddingLoss, self).__init__()
        if norm not in [1, 2]:
            raise ValueError("Norm must be either 1 (L1) or 2 (L2).")
        self.norm = norm

    def forward(self, *embeddings: nn.Embedding) -> Tensor:
        """Compute the EmbeddingLoss loss.

        Args:
            *embeddings (nn.Embedding): The list of embedding layers.

        Returns:
            Tensor: The computed Embedding loss.
        """
        emb_loss = torch.tensor(0.0, device=embeddings[0].weight.device)
        num_embeddings = len(embeddings)

        for embedding_layer in embeddings:
            emb_loss += torch.norm(embedding_layer.weight, p=self.norm)

        emb_loss /= num_embeddings
        return emb_loss
