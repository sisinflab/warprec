# Splitter Configuration

The **Splitter Configuration** module defines how a dataset is partitioned into **training**, **validation**,
and **test sets** prior to model training.
Proper splitting is crucial to build a reliable evaluation pipeline and ensure fair comparison between models.

WarpRec provides multiple splitting strategies that can be tailored to your experimental needs, including:

- Temporal strategies
- Random strategies
- Timestamp slicing
- K-Fold Cross Validation

!!! important
    - Temporal strategies require that **timestamps** are present in the dataset loaded by the reader.
    - **Test set** is required for train and design pipelines; **validation set** is optional.
    - Each strategy requires the parameter it reads (`ratio`, `k`, `timestamp` or `folds`, as listed below); a strategy missing it is refused when the configuration is loaded, and so is a `SplitSpec` built from Python.
    - **seed** seeds the random strategies (`random_holdout`, `random_leave_k_out` and the cold-start ones), separately for `test_splitting` and `validation_splitting`. An integer from `0` to `2**32 - 1`; `null` is refused when the configuration is loaded. Defaults to `42`.

## Supported Splitting Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| `temporal_holdout` | Temporal | Split by timestamp; most recent interactions become test set. |
| `temporal_leave_k_out` | Temporal | Leave the last K interactions per user for testing. |
| `random_holdout` | Random | Randomly sample a ratio of interactions for the test set. |
| `random_leave_k_out` | Random | Randomly leave K interactions per user for testing. |
| `timestamp_slicing` | Temporal | Split at a specific timestamp boundary. |
| `k_fold_cross_validation` | Cross-validation | K-fold partitioning (validation only). |
| `item_cold_start` | Cold-start | Hold out every interaction of a ratio of the **items**. |
| `user_cold_start` | Cold-start | Hold out every interaction of a ratio of the **users**. |

!!! note "Interactions that share a timestamp"
    Temporal order is the **timestamp, then the order of the rows in the file**: of two interactions of a user recorded in the same second, the one further down the file counts as the more recent. `temporal_holdout` and `temporal_leave_k_out` hold out the last rows in that order, so the split is the same on every backend and every run. Without a timestamp column, file order alone is used. `timestamp_slicing` compares timestamps only, so every interaction at the boundary timestamp lands on the same side.

    Every strategy returns the training and the evaluation set with their rows **in file order**, so whatever comes next breaks ties the same way: a temporal validation split taken out of a random test split's training set, the sequences of the [sequential models](../recommenders/sequential.md) and the `first` and `last` [duplicate policies](reader.md).

**1. Temporal Holdout**

Orders interactions by **timestamp** and reserves a portion of the most recent interactions as the test set.

```yaml
splitter:
  test_splitting:
    strategy: temporal_holdout
    ratio: 0.1
```

**2. Temporal Leave-K-Out**

Orders interactions by **timestamp** and leaves exactly **K** interactions per user for the test set.
Users with K or fewer interactions are dropped from both training and test.

```yaml
splitter:
  test_splitting:
    strategy: temporal_leave_k_out
    k: 1
```

**3. Random Holdout**

Randomly selects a portion of interactions to include in the test set.

```yaml
splitter:
  test_splitting:
    strategy: random_holdout
    ratio: 0.1
```

**4. Random Leave-K-Out**

Randomly selects **K** interactions per user to include in the test set.
Users with K or fewer interactions are dropped from both training and test.

```yaml
splitter:
  test_splitting:
    strategy: random_leave_k_out
    k: 1
```

**5. Timestamp Slicing**

Splits the dataset based on a specific **timestamp**:

- Interactions before the timestamp -> training set
- Interactions after the timestamp -> test set

WarpRec also supports the special keyword `best` to automatically select an optimal timestamp.

```yaml
splitter:
  test_splitting:
    strategy: timestamp_slicing
    timestamp: 10009287 | best
```

**6. K-Fold Cross Validation**

Partitions the dataset into **K folds**, using K-1 folds for training and the remaining fold for validation.
The process is repeated K times to exhaust all possible fold combinations.
Each user's rows are dealt to the folds round-robin in file order, so the assignment is deterministic and `seed` is not read.

```yaml
splitter:
  validation_splitting:
    strategy: k_fold_cross_validation
    folds: 10
```

!!! note
    - This strategy is applicable **only for validation sets**, not test sets.
    - Provides less biased and more accurate evaluation metrics, but requires additional training time.

**7. Cold-Start**

```yaml
splitter:
  test_splitting:
    strategy: item_cold_start
    ratio: 0.1
```

!!! important
    `ratio` is a fraction of **entities**, not of interactions, unlike every other
    strategy. The sampled items keep no training history at all, so only a model that
    scores from attributes can reach them.

    Pair it with `evaluation.candidates: cold` and read the result against the random
    floor WarpRec logs. See [Cold-Start Protocols](../evaluation/cold-start.md).

!!! note
    `ratio` is required: a cold-start strategy without it is refused when the
    configuration is loaded. As `validation_splitting`, a cold-start strategy is
    accepted only when `test_splitting` holds out the same side (`item_cold_start`
    for both, or `user_cold_start` for both). Otherwise the entities it holds out
    are dropped when the validation set is aligned with its train set, which would
    leave the validation set empty, so the configuration is refused at load.

## Example Splitter Configuration

This example demonstrates a full configuration:

- Test set split using **temporal holdout** (10% of the most recent interactions)
- Validation set using **10-fold cross validation**

```yaml
splitter:
  test_splitting:
    strategy: temporal_holdout
    ratio: 0.1
  validation_splitting:
    strategy: k_fold_cross_validation
    folds: 10
```
