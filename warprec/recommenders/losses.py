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
