# pylint: disable = R0801, E1102
from typing import Any, Optional, Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_
from scipy.sparse import csr_matrix

from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry


@model_registry.register(name="FISM")
class FISM(Recommender):
    r"""Implementation of FISM model from
    FISM: Factored Item Similarity Models for Top-N Recommender Systems (KDD 2013).

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/2487575.2487589>`_.

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
        embedding_size (int): The number of factors for item feature embeddings.
        reg_1 (float): Regularization coefficient for the item source embeddings (beta).
        reg_2 (float): Regularization coefficient for the item destination embeddings (lambda).
        alpha (float): The alpha parameter, a value between 0 and 1, used in the similarity calculation.
        split_to (int): Parameter for splitting items into chunks during prediction (for memory management).
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
    """

    # Model specific parameters
    embedding_size: int
    reg_1: float
    reg_2: float
    alpha: float
    split_to: int
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
        super().__init__(params, *args, device=device, seed=seed, info=info, **kwargs)
        self._name = "FISM"

        # Get information from dataset info
        self.n_users = info.get("users", None)
        if not self.n_users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        self.n_items = info.get("items", None)
        if not self.n_items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        self.block_size = kwargs.get("block_size", 50)

        # Embeddings and biases
        self.item_src_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0
        )  # +1 for padding
        self.item_dst_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0
        )  # +1 for padding
        self.user_bias = nn.Parameter(torch.zeros(self.n_users))
        self.item_bias = nn.Parameter(torch.zeros(self.n_items + 1))  # +1 for padding

        # Parameters initialization
        self.apply(self._init_weights)

        # Define the loss
        self.bceloss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # These will be set in the fit method
        self.history_matrix: Tensor | None = None
        self.history_lens: Tensor | None = None
        self.history_mask: Tensor | None = None

        # Handle groups
        self.group = torch.chunk(torch.arange(1, self.n_items + 1), self.split_to)

        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)  # Standard deviation

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass for calculating scores for specific user-item pairs.

        Args:
            user (Tensor): User indices.
            item (Tensor): Item indices.

        Returns:
            Tensor: Predicted scores.
        """
        # Adjust item indices for embedding (0 is padding, so items are 1-indexed)
        item_emb_idx = item + 1

        user_inter = self.history_matrix[user]
        item_num = self.history_lens[user].unsqueeze(1)
        batch_mask_mat = self.history_mask[user]

        user_history = self.item_src_embedding(
            user_inter
        )  # batch_size x max_len x embedding_size
        target = self.item_dst_embedding(item_emb_idx)  # batch_size x embedding_size

        user_bias = self.user_bias[user]  # batch_size
        item_bias = self.item_bias[item_emb_idx]  # batch_size

        # (batch_size, max_len, embedding_size) @ (batch_size, embedding_size, 1) -> (batch_size, max_len, 1)
        similarity = torch.bmm(user_history, target.unsqueeze(2)).squeeze(
            2
        )  # batch_size x max_len

        # Apply mask to similarity
        similarity = batch_mask_mat * similarity

        # coeff = N_u ^ (-alpha)
        # Add a small epsilon to item_num to prevent division by zero for users with no history
        coeff = torch.pow(item_num.squeeze(1).float() + 1e-6, -self.alpha)  # batch_size

        # Scores = coeff * sum(similarity) + user_bias + item_bias
        scores = coeff * torch.sum(similarity, dim=1) + user_bias + item_bias
        return scores

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method of FISM model.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        self.train()

        # Prepare history information
        self.history_matrix, self.history_lens, self.history_mask = (
            interactions.get_history()
        )

        # Get training data (user_ids, item_ids, ratings)
        train_coo = interactions.get_sparse().tocoo()
        user_tensor = torch.tensor(train_coo.row, dtype=torch.long, device=self._device)
        item_tensor = torch.tensor(train_coo.col, dtype=torch.long, device=self._device)
        label_tensor = torch.tensor(
            train_coo.data, dtype=torch.float, device=self._device
        )  # Assuming ratings are 1.0 for positive interactions

        dataset_size = user_tensor.size(0)
        batch_size = interactions.batch_size
        num_batches = (dataset_size + batch_size - 1) // batch_size

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            # Shuffle data for each epoch
            indices = torch.randperm(dataset_size, device=self._device)
            shuffled_user_tensor = user_tensor[indices]
            shuffled_item_tensor = item_tensor[indices]
            shuffled_label_tensor = label_tensor[indices]

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, dataset_size)

                self.optimizer.zero_grad()

                batch_user = shuffled_user_tensor[start_idx:end_idx]
                batch_item = shuffled_item_tensor[start_idx:end_idx]
                batch_label = shuffled_label_tensor[start_idx:end_idx]

                output = self.forward(batch_user, batch_item)

                # Loss computation and backpropagation
                loss = self.bceloss(output, batch_label.float()) + self._reg_loss()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if report_fn:
                report_fn(self, loss=epoch_loss, epoch=epoch)

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction using the learned embeddings."""
        # Get user batching parameters from kwargs
        start_user = kwargs.get("start", 0)
        end_user = kwargs.get("end", self.n_users)

        # Retrieve training set to get history data
        if (
            self.history_matrix is None
            or self.history_lens is None
            or self.history_mask is None
        ):
            train_set: Interactions = kwargs.get("train_set", None)
            if train_set is None:
                raise RuntimeError(
                    "The model has not been fit yet. Call fit() before predict() "
                    "or pass the training set as kwargs."
                )
            self.history_matrix, self.history_lens, self.history_mask = (
                train_set.get_history()
            )

        # Retrieve embedding and bias
        all_item_dst_emb = self.item_dst_embedding.weight[1:]
        all_item_bias = self.item_bias[1:]

        # Tensor of indices
        user_batch_indices = torch.arange(start_user, end_user, device=self._device)

        # Pre-compute indices and coefficient
        user_batch_indices = torch.arange(start_user, end_user, device=self._device)
        user_history_indices = self.history_matrix[user_batch_indices]
        item_counts = self.history_lens[user_batch_indices]
        batch_user_bias = self.user_bias[user_batch_indices]
        user_history_emb = self.item_src_embedding(user_history_indices)
        coeff = torch.pow(item_counts.float() + 1e-6, -self.alpha).unsqueeze(1)

        # Iter in blocks for efficiency
        block_predictions = []
        for start_item in range(0, self.n_items, self.block_size):
            end_item = min(start_item + self.block_size, self.n_items)

            # Item indices for this block
            item_indices_block = torch.arange(start_item, end_item, device=self._device)

            # Extract embeddings based on the block
            chunk_item_dst_emb = all_item_dst_emb[item_indices_block]
            chunk_item_bias = all_item_bias[item_indices_block]

            # Compute similarity on the block
            similarity_matrix = torch.matmul(user_history_emb, chunk_item_dst_emb.T)
            sum_similarity = torch.sum(similarity_matrix, dim=1)

            chunk_scores = (
                coeff * sum_similarity
                + batch_user_bias.unsqueeze(1)
                + chunk_item_bias.unsqueeze(0)
            )
            block_predictions.append(chunk_scores)

        # Concat the scores of the different groups
        predictions = torch.cat(block_predictions, dim=1)

        # Mask previously seen items
        coo = interaction_matrix.tocoo()
        user_indices_in_batch = torch.from_numpy(coo.row).to(self._device)
        item_indices = torch.from_numpy(coo.col).to(self._device)
        predictions[user_indices_in_batch, item_indices] = -torch.inf

        # We return raw scores without sigmoid
        return predictions

    def _reg_loss(self) -> Tensor:
        """Compute the regularization loss for embedding layers.

        Returns:
            Tensor: The sum of the two regularization losses.
        """
        loss_1 = self.reg_1 * self.item_src_embedding.weight.pow(2).sum()
        loss_2 = self.reg_2 * self.item_dst_embedding.weight.pow(2).sum()
        return loss_1 + loss_2
