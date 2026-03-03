# Stash

The `Stash` acts as a flexible container for user-defined data structures. In the field of Recommender Systems, which is constantly evolving, providing a one-size-fits-all solution for every scenario is inherently difficult. The `Stash` mechanism allows developers to persist arbitrary data that can be retrieved later during training or evaluation.

The main way to interact with the `Stash` is through the **WarpRec Callback System**, which provides hooks at key stages of the data processing and training pipeline. You can find the full documentation of the callback system [here](../guides/callbacks.md). Another way to access the `Stash` is through a custom script that directly interacts with WarpRec's internal components.
