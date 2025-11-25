##########################
Sequential Recommenders
##########################

The **Sequential Recommenders** module of WarpRec focuses on models that leverage the temporal order of user interactions to predict future behaviors.
Unlike general recommenders, which often treat interactions as independent events, sequential models explicitly capture the dynamics of user preferences within a session or across time.
These models are particularly effective for tasks such as next-item prediction in e-commerce or personalized content recommendation in streaming services.

=============
CNN-Based
=============

CNN-based sequential recommenders apply convolutional operations to user interaction histories, treating sequences as structured data.
These models can capture both short-term dependencies and long-term patterns through different convolutional filters.

- Caser (Convolutional Sequence Embedding Recommendation):
  Treats a user’s interaction history as a 2D "image" and applies horizontal and vertical convolutional filters.
  Caser models local patterns (short-term interests) as well as long-term user preferences, making it effective in session-based recommendation scenarios.

.. code-block:: yaml

    models:
        Caser:
            embedding_size: 64
            n_h: 8
            n_v: 4
            dropout_prob: 0.5
            weight_decay: 0.0
            batch_size: 512
            epochs: 30
            learning_rate: 0.001
            neg_samples: 1
            max_seq_len: 20

=============
Markov-Chains
=============

Some sequential recommenders combine Markovian assumptions with item similarity to balance short-term context with long-term personalization.
These models are especially suited to scenarios where user behavior exhibits both immediate intent and broader preferences.

- FOSSIL (FactOrized Sequential Prediction with Item SImilarity ModeLs):
  Integrates a first-order Markov Chain for short-term user behavior with a factored item similarity model (inspired by SLIM) to address data sparsity and capture long-term dependencies.

.. code-block:: yaml

    models:
        FOSSIL:
            embedding_size: 64
            order_len: 8
            alpha: 0.001
            weight_decay: 0.0
            batch_size: 512
            epochs: 100
            learning_rate: 0.001
            neg_samples: 1
            max_seq_len: 100

=============
RNN-Based
=============

Recurrent Neural Networks (RNNs) process sequences step by step, maintaining a hidden state to capture temporal dependencies.
They are effective for modeling evolving user interests within sessions.

- GRU4Rec (Gated Recurrent Unit for Recommendation):
  One of the earliest deep learning approaches for session-based recommendation.
  It leverages GRUs to model user interaction sequences, focusing on short-term behavior and next-item prediction.

.. code-block:: yaml

    models:
        GRU4Rec:
            embedding_size: 128
            hidden_size: 128
            num_layers: 1
            dropout_prob: 0.2
            weight_decay: 0.0001
            batch_size: 512
            epochs: 100
            learning_rate: 0.001
            neg_samples: 2
            max_seq_len: 200

=============
Transformer-Based
=============

Transformer-inspired recommenders employ self-attention to capture dependencies across an entire sequence simultaneously.
They excel at modeling both short-term and long-term user preferences without relying on recurrence or convolution.

- SASRec (Self-Attentive Sequential Recommendation):
  A Transformer-based model that uses stacked self-attention blocks to capture item dependencies in user sequences.
  SASRec effectively models dynamic user preferences in sparse datasets, learning both short- and long-term interests.

.. code-block:: yaml

    models:
        SASRec:
            embedding_size: 128
            n_layers: 2
            n_heads: 4
            inner_size: 512
            dropout_prob: 0.3
            attn_dropout_prob: 0.3
            learning_rate: 0.001
            weight_decay: 0.0
            batch_size: 512
            epochs: 100
            neg_samples: 1
            max_seq_len: 200

- gSASRec (General Self-Attentive Sequential Recommendation):
  Extends SASRec by introducing general self-attention.
  This enables better modeling of diverse or evolving user interests.

.. code-block:: yaml

    models:
        gSASRec:
            embedding_size: 128
            n_layers: 2
            n_heads: 4
            inner_size: 512
            dropout_prob: 0.3
            attn_dropout_prob: 0.3
            learning_rate: 0.001
            gbce_t: 0.5
            weight_decay: 0.0
            batch_size: 512
            epochs: 100
            neg_samples: 1
            max_seq_len: 200
            reuse_item_embeddings: True

===============================
Summary of Available Sequential Models
===============================

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Category
     - Model
     - Description
   * - CNN-Based
     - Caser
     - Convolutional model capturing local and global sequential patterns.
   * - Markov-Chains
     - FOSSIL
     - Combines Markov Chains with factored item similarity for sequential prediction.
   * - RNN-Based
     - GRU4Rec
     - Session-based recommender using GRUs for short-term preference modeling.
   * - Transformer-Based
     - SASRec
     - Transformer-inspired model learning short- and long-term user preferences.
   * -
     - gSASRec
     - General self-attention model for diverse and evolving user behaviors.
