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
