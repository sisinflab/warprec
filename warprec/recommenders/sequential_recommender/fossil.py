# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Optional, Callable, Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_
from scipy.sparse import csr_matrix

from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss
from warprec.data.dataset import Interactions, Sessions
from warprec.utils.registry import model_registry


@model_registry.register(name="FOSSIL")
class FOSSIL(Recommender, SequentialRecommenderUtils):
    """Implementation of FOSSIL algorithm from
    "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation." in ICDM 2016.

    FOSSIL uses similarity of the items as main purpose and uses high MC as a way of sequential preference improve of
    ability of sequential recommendation.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If essential values like 'items' or 'max_seq_len' are not passed
                    through the info dict.

    Attributes:
        embedding_size (int): The dimension of the item embeddings.
        order_len (int): The number of last items to consider for high-order Markov chains.
        reg_weight (float): The L2 regularization weight.
        alpha (float): The parameter for calculating similarity.
        weight_decay (float): The value of weight decay used in the optimizer.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    # Model hyperparameters
    embedding_size: int
    order_len: int
    reg_weight: float
    alpha: float
    weight_decay: float
    epochs: int
    learning_rate: float
    neg_samples: int
    max_seq_len: int

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
        self._name = "FOSSIL"

        # Get information from dataset info
        self.n_items = info.get("items", None)
        self.n_users = info.get("users", None)
        if not self.n_items or not self.n_users:
            raise ValueError(
                "Both 'items' and 'users' must be provided to correctly initialize the model."
            )

        # Define the layers
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0
        )
        self.user_lambda = nn.Embedding(
            self.n_users, self.order_len
        )  # User specific weights for Markov chains
        self.lambda_ = nn.Parameter(
            torch.zeros(self.order_len)
        )  # Global weights for Markov chains

        # Initialize weights
        self.apply(self._init_weights)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Loss function
        self.loss: nn.Module
        if self.neg_samples > 0:
            self.loss = BPRLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _inverse_seq_item_embedding(
        self, seq_item_embedding: Tensor, seq_item_len: Tensor
    ) -> Tensor:
        """Inverts and pads sequence item embeddings to create a "short" item embedding.

        This method effectively shifts and gathers specific item embeddings from the end
        of each sequence, effectively creating a "short" representation of the sequence
        from its tail, padded with zeros at the beginning. This is often used in models
        where the most recent interactions are of particular interest for higher-order
        Markov chains or similar sequential processing.

        Args:
            seq_item_embedding (Tensor): A tensor representing the embeddings of items in a sequence.
                                         Expected shape: (batch_size, sequence_length, embedding_dim).
            seq_item_len (Tensor): A tensor representing the actual lengths of the sequences.
                                   Expected shape: (batch_size,).

        Returns:
            Tensor: A tensor representing the "short" item embeddings, extracted from the
                    tail of the original sequences and padded.
                    Expected shape: (batch_size, order_len, embedding_dim).
        """
        # Create a tensor of zeros with the same shape and type as seq_item_embedding
        # This will be prepended to the sequence embeddings to act as padding or initial state.
        zeros = torch.zeros_like(seq_item_embedding, dtype=torch.float).to(
            self._device
        )  # (batch_size, sequence_length, embedding_dim)

        # Concatenate zeros to the beginning of seq_item_embedding along dimension 1 (sequence_length)
        # This effectively shifts the original embeddings to the right, creating a padded sequence.
        item_embedding_zeros = torch.cat(
            [zeros, seq_item_embedding], dim=1
        )  # (batch_size, 2 * sequence_length, embedding_dim)

        # Iterate 'order_len' times to gather specific items from the padded sequence
        embedding_list = []
        for i in range(self.order_len):
            # Calculate the index for gathering. This index is relative to the padded sequence.
            # The indices are designed to gather the last `order_len` items from the original sequence
            # within the context of the `item_embedding_zeros` tensor.
            embedding = self._gather_indexes(
                item_embedding_zeros,
                self.max_seq_len + seq_item_len - self.order_len + i,
            )  # (batch_size, embedding_dim)
            embedding_list.append(embedding.unsqueeze(1))

        # Concatenate all the gathered embeddings along dimension 1
        # This stacks the 'order_len' individual item embeddings for each sequence.
        short_item_embedding = torch.cat(
            embedding_list, dim=1
        )  # (batch_size, order_len, embedding_dim)
        return short_item_embedding

    def _get_high_order_Markov(
        self, high_order_item_embedding: Tensor, user: Tensor
    ) -> Tensor:
        """Calculates a weighted high-order Markov embedding based on user and item interactions.

        This method applies user-specific and general lambda weights to the high-order item embeddings,
        effectively creating a weighted sum that represents a more personalized high-order Markov state.

        Args:
            high_order_item_embedding (Tensor): A tensor representing the high-order embeddings of items.
                                                Expected shape: (batch_size, num_items, embedding_dim).
            user (Tensor): A tensor representing the user embedding or features.
                           Expected shape: (batch_size, user_feature_dim).

        Returns:
            Tensor: A tensor representing the aggregated high-order Markov embedding after applying
                    the lambda weights and summing along the item dimension.
                    Expected shape: (batch_size, embedding_dim).
        """
        # Calculate user-specific lambda and unsqueeze dimensions for broadcasting
        user_lambda = self.user_lambda(user).unsqueeze(dim=2)  # (batch_size, 1, 1)

        # Unsqueeze general lambda for broadcasting
        lambda_ = self.lambda_.unsqueeze(dim=0).unsqueeze(
            dim=2
        )  # (1, num_lambda_weights, 1)

        # Add user-specific and general lambda values
        lambda_ = torch.add(user_lambda, lambda_)  # (batch_size, num_items, 1)

        # Apply the combined lambda weights to the high-order item embeddings
        high_order_item_embedding = torch.mul(
            high_order_item_embedding, lambda_
        )  # (batch_size, num_items, embedding_dim)

        # Sum the weighted embeddings along the item dimension
        high_order_item_embedding = high_order_item_embedding.sum(
            dim=1
        )  # (batch_size, embedding_dim)

        return high_order_item_embedding

    def _get_similarity(
        self, seq_item_embedding: Tensor, seq_item_len: Tensor
    ) -> Tensor:
        """Calculates a weighted similarity based on sequence item embeddings and their lengths.

        This method computes a coefficient based on the inverse power of sequence item lengths,
        then multiplies this coefficient with the sum of sequence item embeddings. This effectively
        down-weights the influence of longer sequences in the final similarity calculation.

        Args:
            seq_item_embedding (Tensor): A tensor representing the embeddings of items in a sequence.
                                         Expected shape: (batch_size, sequence_length, embedding_dim).
            seq_item_len (Tensor): A tensor representing the actual lengths of the sequences.
                                   Expected shape: (batch_size,).

        Returns:
            Tensor: A tensor representing the similarity score for each sequence after applying
                    the length-based weighting.
                    Expected shape: (batch_size, embedding_dim).
        """
        # Calculate the coefficient based on sequence length
        coeff = torch.pow(
            seq_item_len.unsqueeze(1).float(), -self.alpha
        )  # (batch_size,  1)

        # Multiply the coefficient with the summed embeddings to compute similarity
        similarity = torch.mul(
            coeff, seq_item_embedding.sum(dim=1)
        )  # (batch_size, embedding_size)
        return similarity

    def _reg_loss(
        self, user_embedding: Tensor, item_embedding: Tensor, seq_output: Tensor
    ) -> Tensor:
        """Calculates the L2 regularization loss."""
        reg_term = (
            self.reg_weight * torch.norm(user_embedding, p=2)
            + self.reg_weight * torch.norm(item_embedding, p=2)
            + self.reg_weight * torch.norm(seq_output, p=2)
        )
        return reg_term

    def forward(
        self, item_seq: Tensor, item_seq_len: Tensor, user_id: Tensor
    ) -> Tensor:
        """Forward pass of the FOSSIL model.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences [batch_size,].
            user_id (Tensor): User IDs for each sequence [batch_size,].

        Returns:
            Tensor: The combined embedding for prediction [batch_size, embedding_size].
        """
        seq_item_embedding = self.item_embedding(item_seq)

        high_order_seq_item_embedding = self._inverse_seq_item_embedding(
            seq_item_embedding, item_seq_len
        )
        # batch_size * order_len * embedding

        high_order = self._get_high_order_Markov(high_order_seq_item_embedding, user_id)
        similarity = self._get_similarity(seq_item_embedding, item_seq_len)

        return high_order + similarity

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method for FOSSIL.

        The training will be conducted on sequential data:
        (user_id, item_sequence, sequence_length, positive_next_item, negative_next_item (optional)).

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.

        Raises:
            ValueError: If the Sessions object is not provided.
        """
        sessions = kwargs.get("sessions")
        if not isinstance(sessions, Sessions):
            raise ValueError("Sessions must be provided correctly to train the model.")

        dataloader = sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            num_negatives=self.neg_samples,
            user_id=True,  # We need user ids for FOSSIL
        )

        self.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                if self.neg_samples > 0:
                    user_id, item_seq, item_seq_len, pos_item, neg_item = [
                        x.to(self._device) for x in batch
                    ]
                else:
                    user_id, item_seq, item_seq_len, pos_item = [
                        x.to(self._device) for x in batch
                    ]
                    neg_item = None

                self.optimizer.zero_grad()

                seq_output = self.forward(item_seq, item_seq_len, user_id)

                pos_items_emb = self.item_embedding(pos_item)
                user_lambda_emb = self.user_lambda(
                    user_id
                )  # Embedding for regularization

                total_loss: Tensor
                if self.neg_samples > 0:
                    neg_items_emb = self.item_embedding(neg_item)
                    pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
                    neg_score = torch.sum(
                        seq_output.unsqueeze(1) * neg_items_emb, dim=-1
                    )
                    total_loss = self.loss(pos_score, neg_score)
                    total_loss += self._reg_loss(
                        user_lambda_emb, pos_items_emb, seq_output
                    )
                else:
                    test_item_emb = self.item_embedding.weight
                    logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                    total_loss = self.loss(logits, pos_item)
                    total_loss += self._reg_loss(
                        user_lambda_emb, pos_items_emb, seq_output
                    )

                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            if report_fn is not None:
                report_fn(self, loss=epoch_loss, epoch=epoch)

    @torch.no_grad()
    def predict(
        self,
        interaction_matrix: csr_matrix,
        user_seq: Tensor,
        seq_len: Tensor,
        user_id: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned session embeddings (full sort prediction).

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            user_id (Tensor): The user IDs corresponding to user_seq.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)
        user_id = user_id.to(self._device)
        print(user_id.min(), user_id.max())

        # Get the combined output embedding for each user
        seq_output = self.forward(
            user_seq, seq_len, user_id
        )  # [num_users, embedding_size]

        # Get embeddings for all items
        all_item_embeddings = self.item_embedding.weight[
            1:
        ]  # [n_items, embedding_size]

        # Calculate scores for all items
        # Scores = dot product of session embedding with all item embeddings
        predictions = torch.matmul(
            seq_output, all_item_embeddings.transpose(0, 1)
        )  # [num_users, n_items]

        # Mask seen items
        coo = interaction_matrix.tocoo()
        user_indices_in_batch = torch.from_numpy(coo.row).to(self._device)
        item_indices = torch.from_numpy(coo.col).to(self._device)
        predictions[user_indices_in_batch, item_indices] = -torch.inf

        return predictions
