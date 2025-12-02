import torch
import torch.nn.functional as F
from torch import nn, Tensor


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking.

    For further details, check the `paper <https://arxiv.org/abs/1205.2618>`_.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        """Compute the BPR loss.

        Args:
            pos_score (Tensor): Positive item scores.
            neg_score (Tensor): Negative item scores.

        Returns:
            Tensor: The computed BPR loss.
        """
        # Compute the distance of positive
        # and negative scores
        distance = pos_score.unsqueeze(1) - neg_score

        # Compute the softplus function of the negative distance
        loss = F.softplus(-distance)  # pylint: disable=not-callable

        # Return the mean of the bpr losses computed
        return loss.mean()


class EmbLoss(nn.Module):
    """L2 Regularization Loss for Embeddings.

    Computes the L2 regularization term (squared L2 norm) for a variable number
    of embedding tensors. This is standard for preventing overfitting in
    embedding-based recommender systems.

    The loss is scaled by 1/2 to simplify the gradient update (derivative of x^2 is 2x).

    Formula:
        If reduction is 'mean':
            L_reg = (1 / 2) * (1 / batch_size) * sum(||E||^2)
        If reduction is 'sum':
            L_reg = (1 / 2) * sum(||E||^2)

    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'mean' | 'sum'. Default: 'mean'.

    Raises:
        ValueError: If the reduction is not supported.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"Invalid reduction type: {reduction}. Supported: 'mean', 'sum'."
            )
        self.reduction = reduction

    def forward(self, *embeddings: Tensor) -> Tensor:
        """Compute the L2 regularization loss.

        Args:
            *embeddings (Tensor): Variable number of embedding tensors.

        Returns:
            Tensor: The computed regularization loss.
        """
        # Edge case: No embedding passed
        if not embeddings:
            return torch.tensor(0.0)

        # PyTorch will handle the casting to Tensor
        l2_reg = torch.tensor(0.0, device=embeddings[0].device)
        for emb in embeddings:
            l2_reg = l2_reg + emb.pow(2).sum()

        # Scale by 1/2 (standard convention in RecSys)
        l2_reg = 0.5 * l2_reg

        if self.reduction == "mean":
            batch_size = embeddings[0].size(0)
            return l2_reg / batch_size
        return l2_reg


class MultiDAELoss(nn.Module):
    """MultiDAELoss, used to train MultiDAE model.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/3178876.3186150>`_.
    """

    def __init__(self):
        super(MultiDAELoss, self).__init__()

    def forward(self, rating_matrix: Tensor, reconstructed: Tensor) -> Tensor:
        """Compute loss for MultiDAE model.

        Args:
            rating_matrix (Tensor): The original ratings.
            reconstructed (Tensor): The reconstructed Tensor from
                MultiDAE model.

        Returns:
            Tensor: The loss value computed.
        """
        return -(F.log_softmax(reconstructed, dim=1) * rating_matrix).sum(dim=1).mean()


class MultiVAELoss(nn.Module):
    def __init__(self):
        super(MultiVAELoss, self).__init__()

    def forward(
        self,
        rating_matrix: Tensor,
        reconstructed: Tensor,
        kl_loss: Tensor,
        anneal: float,
    ) -> Tensor:
        """Compute loss for MultiVAE model.

        Args:
            rating_matrix (Tensor): The original ratings.
            reconstructed (Tensor): The reconstructed Tensor from
                MultiVAE model.
            kl_loss (Tensor): The KL loss computed during the
                forward step.
            anneal (float): The anneal value based on epoch.

        Returns:
            Tensor: The loss value computed.
        """
        log_softmax = F.log_softmax(reconstructed, dim=1)
        neg_ll = -torch.mean(torch.sum(log_softmax * rating_matrix, dim=1))
        return neg_ll + anneal * kl_loss
