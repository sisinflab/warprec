from typing import Any, List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender, Recommender
from warprec.recommenders.losses import BPRLoss
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry


@model_registry.register(name="MyEASE")
class MyEASE(Recommender):
    """EASE (Steck, 2019): a closed-form item-item model, fitted in the constructor.

    Args:
        params (dict): The hyperparameters; only the annotated ones are read.
        info (dict): The dataset information, 'n_users' and 'n_items' among it.
        interactions (Interactions): The training interactions.
        *args (Any): Passed on to the base class.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Everything else a pipeline passes, ignored here.

    Attributes:
        l2 (float): The L2 regularisation of the item-item weights.
    """

    l2: float

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
        self.train_matrix = interactions.get_sparse()

        X = self.train_matrix
        gram = (X.T @ X).toarray() + self.l2 * np.identity(self.n_items)
        weights = np.linalg.inv(gram)
        weights /= -np.diag(weights)
        np.fill_diagonal(weights, 0.0)

        # A buffer moves with the model to its device and is saved in its state.
        self.register_buffer("weights", torch.from_numpy(weights))

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Score items for a batch of users.

        Args:
            user_indices (Tensor): The users, shape [batch].
            *args (Any): Unused.
            item_indices (Optional[Tensor]): None to score every item, or the
                candidates of each user, shape [batch, candidates]. Candidate
                lists are padded with the index n_items.
            **kwargs (Any): Unused.

        Returns:
            Tensor: Shape [batch, n_items], or [batch, candidates].
        """
        history = self.train_matrix[user_indices.tolist()].toarray()
        scores = torch.from_numpy(history).to(self.weights) @ self.weights
        if item_indices is None:
            return scores

        # The padding index has no column; its score is discarded by the
        # evaluator, so any in-range value will do.
        candidates = item_indices.to(scores.device).clamp(max=self.n_items - 1)
        return scores.gather(1, candidates)


@model_registry.register(name="MyBPR")
class MyBPR(IterativeRecommender):
    """Matrix factorisation trained with the BPR loss, without regularisation.

    Args:
        params (dict): The hyperparameters; only the annotated ones are read.
        info (dict): The dataset information, 'n_users' and 'n_items' among it.
        *args (Any): Passed on to the base class.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Everything else a pipeline passes, ignored here.

    Attributes:
        embedding_size (int): The size of the user and item factors.
        batch_size (int): The number of training triples per batch.
        epochs (int): The number of passes over the training data.
        learning_rate (float): The optimiser's learning rate.
    """

    embedding_size: int
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # Item n_items is the padding index of candidate lists and sequences.
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        self.apply(self._init_weights)
        self.bpr_loss = BPRLoss()
        self.epoch_losses: List[float] = []

    def get_dataloader(
        self, interactions: Interactions, sessions: Sessions, **kwargs: Any
    ) -> DataLoader:
        """Batches of (user, positive item, sampled negative item) triples.

        Args:
            interactions (Interactions): The training interactions.
            sessions (Sessions): The training sessions, unused here.
            **kwargs (Any): DataLoader options the pipeline sets (workers...).

        Returns:
            DataLoader: The training loader.
        """
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size, **kwargs
        )

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """The score of each (user, item) pair.

        Args:
            user (Tensor): The users, shape [batch].
            item (Tensor): The items, shape [batch].

        Returns:
            Tensor: The scores, shape [batch].
        """
        return (self.user_embedding(user) * self.item_embedding(item)).sum(dim=1)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """The loss of one batch; Lightning does the backward pass and the step.

        WarpRec logs the returned loss as 'train_loss', its mean over the epoch.

        Args:
            batch (Any): One batch from get_dataloader.
            batch_idx (int): Its position in the epoch.

        Returns:
            Tensor: The scalar loss.
        """
        user, positive, negative = batch
        return self.bpr_loss(self.forward(user, positive), self.forward(user, negative))

    def on_train_epoch_end(self):
        """Keep the epoch's mean loss and write it to the WarpRec log."""
        self.epoch_losses.append(self.trainer.callback_metrics["train_loss"].item())
        logger.msg(
            f"{self.name} epoch {self.current_epoch + 1}: "
            f"loss {self.epoch_losses[-1]:.4f}"
        )

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Score items for a batch of users.

        Args:
            user_indices (Tensor): The users, shape [batch].
            *args (Any): Unused.
            item_indices (Optional[Tensor]): None to score every item, or the
                candidates of each user, shape [batch, candidates], padded with
                the index n_items.
            **kwargs (Any): Unused.

        Returns:
            Tensor: Shape [batch, n_items], or [batch, candidates].
        """
        users = self.user_embedding(user_indices)  # [batch, e]
        if item_indices is None:
            items = self.item_embedding.weight[:-1]  # [n_items, e], no padding row
            return users @ items.T
        items = self.item_embedding(item_indices)  # [batch, candidates, e]
        return torch.einsum("be,bce->bc", users, items)
