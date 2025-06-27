# 🔄 Sequential Recommenders

The `Sequential Recommenders` module of WarpRec focuses on models that leverage the temporal order of user interactions to predict future behaviors. Unlike general recommenders that often treat interactions as independent events, sequential models capture dynamic user preferences and context within a session or over time. This makes them particularly effective for tasks like next-item prediction in e-commerce or content streaming.

- [Caser](caser.py): A neural recommender model that captures sequential patterns in user behavior by treating a user’s interaction history as an "image" and applying convolutional filters. It learns both local (short-term) and global (long-term) user preferences by combining horizontal and vertical convolutional layers, making it effective for session-based and sequential recommendation tasks.

```yaml
models:
    Caser:
        embedding_size: 64
        n_h: 8
        n_v: 4
        dropout_prob: 0.5
        weight_decay: 0.0
        epochs: 3
        learning_rate: 0.001
        neg_samples: 1
        max_seq_len: 20
...
```

- [FOSSIL](fossil.py): A recommendation model that combines sequential modeling with item similarity. It leverages a first-order Markov Chain to capture short-term user behavior and integrates a factored item similarity model (inspired by SLIM) to better handle data sparsity and model long-term preferences.

```yaml
models:
    FOSSIL:
        embedding_size: 64
        order_len: 8
        reg_weight: 0.001
        alpha: 0.001
        weight_decay: 0.0
        epochs: 100
        learning_rate: 0.001
        neg_samples: 1
        max_seq_len: 100
...
```

- [GRU4Rec](gru4rec.py): An early and influential recurrent neural network-based model for session-based recommendation. It utilizes Gated Recurrent Units (GRUs) to capture sequential patterns within user sessions, predicting the next item a user is likely to interact with. GRU4Rec is known for its ability to model short-term user interests effectively.

```yaml
models:
    GRU4Rec:
        embedding_size: 128
        hidden_size: 128
        num_layers: 1
        dropout_prob: 0.2
        weight_decay: 0.0001
        epochs: 100
        learning_rate: 0.001
        neg_samples: 2
        max_seq_len: 200
...
```

- [SASRec (Self-Attentive Sequential Recommendation)](sasrec.py): A sequential recommendation model based on self-attention mechanisms, inspired by the Transformer architecture. It uses stacked self-attention blocks to capture dependencies between previously interacted items, effectively modeling users' dynamic preferences. SASRec is capable of learning both short- and long-term patterns without relying on RNNs or CNNs, making it well-suited for sparse recommendation scenarios.

```yaml
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
        epochs: 100
        neg_samples: 1
        max_seq_len: 200
...
```
