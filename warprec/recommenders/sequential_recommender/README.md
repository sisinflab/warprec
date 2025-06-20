# 🔄 Sequential Recommenders

The `Sequential Recommenders` module of WarpRec focuses on models that leverage the temporal order of user interactions to predict future behaviors. Unlike general recommenders that often treat interactions as independent events, sequential models capture dynamic user preferences and context within a session or over time. This makes them particularly effective for tasks like next-item prediction in e-commerce or content streaming.

- [GRU4Rec](gru4rec.py): An early and influential recurrent neural network-based model for session-based recommendation. It utilizes Gated Recurrent Units (GRUs) to capture sequential patterns within user sessions, predicting the next item a user is likely to interact with. GRU4Rec is known for its ability to model short-term user interests effectively.

```yaml
models:
    GRU4Rec:
        embedding_size: 128
        hidden_size: 128
        num_layers: 1
        dropout_prob: 0.2
        weight_decay: 0.0001
        epochs: 20
        learning_rate: 0.001
        neg_samples: 2
...
```
