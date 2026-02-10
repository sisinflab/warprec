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
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
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
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 1
        max_seq_len: 200

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
        hidden_size: 64
        num_layers: 2
        dropout_prob: 0.1
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 1
        max_seq_len: 200

=============
Transformer-Based
=============

Transformer-inspired recommenders employ self-attention to capture dependencies across an entire sequence simultaneously.
They excel at modeling both short-term and long-term user preferences without relying on recurrence or convolution.

- BERT4Rec (Bidirectional Encoder Representations from Transformers for Recommendation):
  Applies a bidirectional Transformer architecture to sequential recommendation.
  Instead of predicting the next item, it is trained on a "cloze" task, where it predicts randomly masked items in a sequence, allowing it to learn context from both past and future interactions.

.. code-block:: yaml

  models:
    BERT4Rec:
      embedding_size: 128
      n_layers: 2
      n_heads: 8
      inner_size: 512
      dropout_prob: 0.1
      attn_dropout_prob: 0.1
      mask_prob: 0.2
      reg_weight: 0.001
      weight_decay: 0.0001
      batch_size: 2048
      epochs: 200
      learning_rate: 0.001
      neg_samples: 1
      max_seq_len: 200

- CORE (Consistent Representation Encoder):
  A session-based recommendation framework that unifies the representation space for both encoding and decoding.
  Unlike standard deep encoders that project session embeddings into a different space than item embeddings, CORE encodes sessions as a weighted sum of item embeddings (using a Transformer to learn the weights).
  It also employs Robust Distance Measuring (RDM) based on cosine similarity to prevent overfitting.

.. code-block:: yaml

    models:
      CORE:
        embedding_size: 64
        dnn_type: "trm"
        n_layers: 2
        n_heads: 8
        inner_size: 256
        hidden_dropout_prob: 0.1
        attn_dropout_prob: 0.1
        layer_norm_eps: 1e-12
        initializer_range: 0.02
        session_dropout: 0.1
        item_dropout: 0.1
        temperature: 0.07
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 1
        max_seq_len: 50

- gSASRec (General Self-Attentive Sequential Recommendation):
  Extends SASRec by introducing general self-attention.
  This enables better modeling of diverse or evolving user interests.

.. code-block:: yaml

    models:
      gSASRec:
        embedding_size: 128
        n_layers: 2
        n_heads: 8
        inner_size: 512
        dropout_prob: 0.1
        attn_dropout_prob: 0.1
        gbce_t: 0.5
        reuse_item_embeddings: True
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 1
        max_seq_len: 200

- LightSANs (Low-Rank Decomposed Self-Attention Networks):
  A sequential recommender that improves upon standard self-attention (like SASRec) by introducing low-rank decomposed self-attention to reduce complexity and decoupled position encoding to better model sequential relations.

.. code-block:: yaml

  models:
    LightSANs:
      embedding_size: 128
      n_layers: 2
      n_heads: 8
      k_interests: 5
      inner_size: 512
      dropout_prob: 0.5
      attn_dropout_prob: 0.5
      reg_weight: 0.001
      weight_decay: 0.0001
      batch_size: 2048
      epochs: 200
      learning_rate: 0.001
      neg_samples: 1
      max_seq_len: 200

- LinRec (Linear Attention Mechanism for Long-term Sequential Recommender Systems):
  Proposed to address the quadratic complexity of standard Transformers.
  LinRec employs a linear attention mechanism (O(N)) using L2 normalization and ELU activation.
  This allows for efficient modeling of long-term user sequences while maintaining the ability to capture complex dependencies, making it significantly faster than standard SASRec on long sequences.

.. code-block:: yaml

    models:
      LinRec:
        embedding_size: 128
        n_layers: 2
        n_heads: 8
        inner_size: 512
        dropout_prob: 0.1
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 1
        max_seq_len: 200

- SASRec (Self-Attentive Sequential Recommendation):
  A Transformer-based model that uses stacked self-attention blocks to capture item dependencies in user sequences.
  SASRec effectively models dynamic user preferences in sparse datasets, learning both short- and long-term interests.

.. code-block:: yaml

    models:
      SASRec:
        embedding_size: 128
        n_layers: 2
        n_heads: 8
        inner_size: 512
        dropout_prob: 0.1
        attn_dropout_prob: 0.1
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 1
        max_seq_len: 200

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
     - BERT4Rec
     - Bidirectional Transformer model trained on a masked item prediction task.
   * -
     - CORE
     - Unifies encoding/decoding spaces using linear combination of embeddings and RDM.
   * -
     - gSASRec
     - General self-attention model for diverse and evolving user behaviors.
   * -
     - LightSANs
     - Efficient self-attention with low-rank decomposition and decoupled positioning.
   * -
     - LinRec
     - Linear attention mechanism (O(N)) for efficient long-term recommendation.
   * -
     - SASRec
     - Transformer-inspired model learning short- and long-term user preferences.
