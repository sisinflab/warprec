# pylint: disable = R0801, E1102
from typing import List, Optional, Callable, Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_
from scipy.sparse import csr_matrix
from warprec.recommenders.layers import MLP, CNN
from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.enums import Activations
from warprec.utils.registry import model_registry


@model_registry.register(name="ConvNCF")
class ConvNCF(Recommender):
    """Implementation of ConvNCF algorithm from
        Outer Product-based Neural Collaborative Filtering 2018.

    For further details, check the `paper <https://www.ijcai.org/proceedings/2018/0460.pdf>`_.

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
        embedding_size (int): The embedding size for users and items.
        cnn_channels (List[int]): The list of output channels for each CNN layer.
        cnn_kernels (List[int]): The list of kernel sizes for each CNN layer.
        cnn_strides (List[int]): The list of stride sizes for each CNN layer.
        dropout_prob (float): The dropout probability for the prediction layer.
        reg_embedding (float): The regularization for embedding weights.
        reg_cnn_mlp (float): The regularization for embedding cnn and mlp layers.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Model hyperparameters
    embedding_size: int
    cnn_channels: List[int]
    cnn_kernels: List[int]
    cnn_strides: List[int]
    dropout_prob: float
    reg_embedding: float
    reg_cnn_mlp: float
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
        self._name = "ConvNCF"

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

        # Set block size
        self.block_size = kwargs.get("block_size", 50)

        # Ray Tune converts lists to tuples
        self.cnn_channels = list(self.cnn_channels)
        self.cnn_kernels = list(self.cnn_kernels)
        self.cnn_strides = list(self.cnn_strides)

        # Embedding layers
        self.user_embedding = nn.Embedding(users, self.embedding_size)
        self.item_embedding = nn.Embedding(items, self.embedding_size)

        # CNN layers
        self.cnn_layers = CNN(
            self.cnn_channels,
            self.cnn_kernels,
            self.cnn_strides,
            activation=Activations.RELU,
        )

        # Prediction layer (MLP)
        # The input of the prediction layer is the output
        # of the CNN, so self.cnn_channels[-1]
        self.predict_layers = MLP(
            [self.cnn_channels[-1], 1], self.dropout_prob, activation=None
        )  # We set no activation for last layer

        # Init embedding weights
        self.apply(self._init_weights)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Move to device
        self.to(self._device)

        print(f"aaaaa: {self.block_size}")

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def _bpr_loss(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        """ConvNCFBPRLoss, based on Bayesian Personalized Ranking.

        This is a variation of the normal BPR loss, used in the original paper.

        Args:
            pos_score (Tensor): Positive item scores.
            neg_score (Tensor): Negative item scores.

        Returns:
            Tensor: The computed ConvNCFBPR loss.
        """
        distance = pos_score - neg_score
        loss = torch.sum(
            torch.nn.functional.softplus(-distance)
        )  # Log-sigmoid loss for BPR (using softplus)
        return loss

    def _reg_loss(self) -> Tensor:
        """Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.

        Returns;
            Tensor: The computed regularization loss.
        """
        loss_1 = self.reg_embedding * self.user_embedding.weight.pow(2).sum()
        loss_2 = self.reg_embedding * self.item_embedding.weight.pow(2).sum()
        loss_3 = torch.tensor(0.0, device=self._device)

        # Regularization for CNN layers
        for name, parm in self.cnn_layers.named_parameters():
            if name.endswith("weight"):
                loss_3 += self.reg_cnn_mlp * parm.pow(2).sum()

        # Regularization for prediction layers (MLP)
        for name, parm in self.predict_layers.named_parameters():
            if name.endswith("weight"):
                loss_3 += self.reg_cnn_mlp * parm.pow(2).sum()

        return loss_1 + loss_2 + loss_3

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
        # ConvNCF uses pairwise training, so we need positive and negative items.
        dataloader = interactions.get_pos_neg_dataloader()

        # Training loop
        self.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                user, pos_item, neg_item = [x.to(self._device) for x in batch]

                # Forward pass for positive and negative items
                self.optimizer.zero_grad()
                pos_item_score = self.forward(user, pos_item)
                neg_item_score = self.forward(user, neg_item)

                # Loss computation (BPR + Regularization)
                bpr_loss = self._bpr_loss(pos_item_score, neg_item_score)
                reg_loss = self._reg_loss()
                total_loss = bpr_loss + reg_loss

                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            if report_fn is not None:
                report_fn(self, loss=epoch_loss)

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass of the ConvNCF model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.

        Returns:
            Tensor: The predicted score for each pair (user, item).
        """
        user_e = self.user_embedding(user)  # [batch_size, embedding_size]
        item_e = self.item_embedding(item)  # [batch_size, embedding_size]

        # Outer product to create interaction map
        # user_e.unsqueeze(2) -> [batch_size, embedding_size, 1]
        # item_e.unsqueeze(1) -> [batch_size, 1, embedding_size]
        # torch.bmm(user_e.unsqueeze(2), item_e.unsqueeze(1)) -> [batch_size, embedding_size, embedding_size]
        interaction_map = torch.bmm(user_e.unsqueeze(2), item_e.unsqueeze(1))

        # Add a channel dimension for CNN input: [batch_size, 1, embedding_size, embedding_size]
        interaction_map = interaction_map.unsqueeze(1)

        # CNN layers
        cnn_output = self.cnn_layers(
            interaction_map
        )  # [batch_size, cnn_channels[-1], H', W']

        # Sum across spatial dimensions (H', W')
        # This reduces the feature map to [batch_size, cnn_channels[-1]]
        cnn_output = cnn_output.sum(axis=(2, 3))

        # Prediction layer (MLP)
        prediction = self.predict_layers(cnn_output)  # [batch_size, 1]

        return prediction.squeeze(-1)  # [batch_size]

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """
        Prediction using the learned embeddings (optimized version).

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_indices = torch.arange(
            kwargs.get("start", 0),
            kwargs.get("end", interaction_matrix.shape[0]),
            device=self._device,
        ).long()

        num_users_in_batch = len(user_indices)
        num_items = interaction_matrix.shape[1]

        # Extract embedding of specific users in batch
        user_e_batch = self.user_embedding(user_indices)

        # Process in blocks for memory efficiency
        all_scores = []
        for item_start_idx in range(0, num_items, self.block_size):
            item_end_idx = min(item_start_idx + self.block_size, num_items)

            # Process embeddings of items in block
            item_indices_block = torch.arange(
                item_start_idx, item_end_idx, device=self._device
            ).long()
            item_e_block = self.item_embedding(item_indices_block)
            num_items_in_block = len(item_indices_block)

            # Expand embeddings to match batch_size and block_size
            user_e_exp = user_e_batch.unsqueeze(1).expand(-1, num_items_in_block, -1)
            item_e_exp = item_e_block.unsqueeze(0).expand(num_users_in_batch, -1, -1)

            # Compute outer product (batch_users x block_items)
            interaction_map = torch.bmm(
                user_e_exp.reshape(-1, self.embedding_size, 1),
                item_e_exp.reshape(-1, 1, self.embedding_size),
            )
            interaction_map = interaction_map.unsqueeze(1)

            # Pass through neural layers
            cnn_output = self.cnn_layers(interaction_map)
            cnn_output = cnn_output.sum(axis=(2, 3))
            prediction = self.predict_layers(cnn_output)

            # Reshape scores to match expected format
            scores_block = prediction.reshape(num_users_in_batch, num_items_in_block)
            all_scores.append(scores_block)

        # Concat predictions
        predictions = torch.cat(all_scores, dim=1)

        # Mask seen items
        coo = interaction_matrix.tocoo()
        user_indices_in_batch = torch.from_numpy(coo.row).to(self._device)
        item_indices = torch.from_numpy(coo.col).to(self._device)
        predictions[user_indices_in_batch, item_indices] = -torch.inf

        return predictions
