# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_
from scipy.sparse import csr_matrix

from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.recommenders.losses import BPRLoss
from elliotwo.utils.registry import model_registry


@model_registry.register(name="BPR")
class BPR(Recommender):
    """Implementation of BPR algorithm from
        BPR: Bayesian Personalized Ranking from Implicit Feedback 2012

    For further details, check the `paper <https://arxiv.org/abs/1205.2618>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items or users value was not passed through the info dict.

    Attributes:
        embedding_size (int): The embedding size of user and item.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Model hyperparameters
    embedding_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, *args, **kwargs)
        self._name = "BPR"

        # Get information from dataset info
        users = info.get("users", None)
        if not users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

        # Embeddings
        self.user_embedding = nn.Embedding(users, self.embedding_size)
        self.item_embedding = nn.Embedding(items, self.embedding_size)

        # Init embedding weights
        self.apply(self._init_weights)

        # Loss and optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss = BPRLoss()

        # Move to device
        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on triplets of (user, positive_item, negative_item).

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        # Get the dataloader from interactions
        dataloader = interactions.get_pos_neg_dataloader()

        # Training loop
        self.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                user, pos_item, neg_item = [x.to(self._device) for x in batch]

                # Forward pass and loss computation
                self.optimizer.zero_grad()
                pos_item_score = self.forward(user, pos_item)
                neg_item_score = self.forward(user, neg_item)
                loss: Tensor = self.loss(pos_item_score, neg_item_score)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if report_fn is not None:
                report_fn(self)

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass of the BPR model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.

        Returns:
            Tensor: The predicted score for each pair of positive and negative items.
        """
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        return torch.mul(user_e, item_e).sum(dim=1)

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction using the learned embeddings.

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        batch_size, num_items = interaction_matrix.shape
        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", interaction_matrix.shape[0])
        block_size = 200  # Better memory management

        preds = []
        for start in range(
            0, num_items, block_size
        ):  # We proceed with the evaluation in blocks
            end = min(start + block_size, num_items)
            items_block = torch.arange(start, end, device=self._device).unsqueeze(0)
            items_block = items_block.expand(batch_size, -1).reshape(
                -1
            )  # [batch_size * block]
            users_block = torch.arange(
                start_idx, end_idx, device=self._device
            ).unsqueeze(1)
            users_block = users_block.expand(-1, end - start).reshape(
                -1
            )  # [batch_size * block]

            user_e = self.user_embedding(users_block)
            item_e = self.item_embedding(items_block)
            preds_block = torch.mul(user_e, item_e).sum(dim=1)
            preds.append(
                preds_block.view(batch_size, end - start)
            )  # [batch_size x block]
        predictions = torch.cat(preds, dim=1)  # [batch_size x num_items]

        coo = interaction_matrix.tocoo()
        user_indices = torch.from_numpy(coo.row).to(self._device)
        item_indices = torch.from_numpy(coo.col).to(self._device)
        predictions[user_indices, item_indices] = -torch.inf

        return predictions
