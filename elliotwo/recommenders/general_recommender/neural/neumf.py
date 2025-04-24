# pylint: disable = R0801, E1102
from typing import List, Optional, Callable, Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_
from scipy.sparse import csr_matrix
from elliotwo.recommenders.layers import MLP
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.utils.registry import model_registry


@model_registry.register(name="NeuMF")
class NeuMF(Recommender):
    """Implementation of NeuMF algorithm from
        Neural Collaborative Filtering 2017.

    For further details, check the `paper <https://arxiv.org/abs/1708.05031>`_.

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
        mf_embedding_size (int): The MF embedding size.
        mlp_embedding_size (int): The MLP embedding size.
        mlp_hidden_size (List[int]): The MLP hidden layer size list.
        mf_train (bool): Wether or not to train MF embedding.
        mlp_train (bool): Wether or not to train MLP embedding.
        dropout (float): The dropout probability.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples per positive interaction.
    """

    # Model hyperparameters
    mf_embedding_size: int
    mlp_embedding_size: int
    mlp_hidden_size: List[int]
    mf_train: bool
    mlp_train: bool
    dropout: float
    epochs: int
    learning_rate: float
    neg_samples: int

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
        self._name = "NeuMF"

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

        # MF embeddings
        self.user_mf_embedding = nn.Embedding(users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(items, self.mf_embedding_size)

        # MLP embeddings
        self.user_mlp_embedding = nn.Embedding(users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(items, self.mlp_embedding_size)

        # MLP layers
        self.mlp_layers = MLP(
            [2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout
        )

        # Final prediction layer
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1
            )
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        else:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        # Init embedding weights
        self.apply(self._init_weights)

        # Loss and optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        # Move to device
        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on triplets of (user, item, rating).
        If the negative sampling has been set, the dataloader will contain also
        some negative samples for each user positive interaction.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        # Get the dataloader from interactions
        dataloader = interactions.get_item_rating_dataloader(
            num_negatives=self.neg_samples
        )

        # Training loop
        self.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                user, item, rating = [x.to(self._device) for x in batch]

                # Forward pass and loss computation
                self.optimizer.zero_grad()
                predictions = self.forward(user, item)
                loss: Tensor = self.loss(predictions, rating)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if report_fn is not None:
                report_fn(self)

    # pylint: disable = E0606
    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass of the NeuMF model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.

        Returns:
            Tensor: The predicted score for each pair (user, item).
        """
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        output: Tensor = None

        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)

        if self.mlp_train:
            mlp_input = torch.cat((user_mlp_e, item_mlp_e), -1)
            mlp_output = self.mlp_layers(mlp_input)

        if self.mf_train and self.mlp_train:
            combined = torch.cat((mf_output, mlp_output), -1)
            output = self.predict_layer(combined)
        elif self.mf_train:
            output = self.predict_layer(mf_output)
        else:
            output = self.predict_layer(mlp_output)

        return output.squeeze(-1)

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
            preds_block = self.sigmoid(self.forward(users_block, items_block))
            preds.append(
                preds_block.view(batch_size, end - start)
            )  # [batch_size x block]
        predictions = torch.cat(preds, dim=1)  # [batch_size x num_items]

        coo = interaction_matrix.tocoo()
        user_indices = torch.from_numpy(coo.row).to(self._device)
        item_indices = torch.from_numpy(coo.col).to(self._device)
        predictions[user_indices, item_indices] = -torch.inf

        return predictions
