from typing import Any, Iterable

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

    def forward(self, *params: Any) -> Tensor:
        """Compute the EmbeddingLoss loss.

        Args:
            *params (Any): A list containing either Tensors, Parameters or Embedding.
                Any other type of data will be ignored.

        Returns:
            Tensor: The computed Embedding loss.
        """
        # This list will store all the tensors to regularize
        tensors_to_regularize: list[Tensor] = []

        # Call the support function to handle the list
        self._collect_tensors(params, tensors_to_regularize)

        # Edge case: list is empty
        if not tensors_to_regularize:
            return torch.tensor(0.0)

        # Initialize the total loss that will be computed on the list
        total_loss = torch.tensor(0.0, device=tensors_to_regularize[0].device)

        # Compute total loss iterating on all the tensors
        for tensor in tensors_to_regularize:
            if self.norm == 2:
                # L2 regularization, we also square the result
                total_loss += torch.norm(tensor, p=2).pow(2)
            else:
                # L1 regularization
                total_loss += torch.norm(tensor, p=1)

        return total_loss

    def _collect_tensors(self, items: Iterable[Any], tensor_list: list[Tensor]):
        """Helper function to handle different type of input data.

        Args:
            items (Iterable[Any]): Any type of iterable object.
            tensor_list (list[Tensor]): The list that will be populated.
        """
        for item in items:
            if isinstance(item, nn.Embedding):
                # In case of an embedding layer, we append it's weights
                tensor_list.append(item.weight)
            elif isinstance(item, torch.Tensor):
                # In case of a tensor, we append the tensor itself
                tensor_list.append(item)
            elif isinstance(item, Iterable) and not isinstance(item, str):
                # In case of another list (or iterable), we
                # recursively call the helper function
                self._collect_tensors(item, tensor_list)
            # Anything else will be ignored
