# Filtering Configuration

The **Filtering Configuration** module defines the preprocessing strategies applied to the dataset
*before* the splitting phase. Filtering runs only with `reader.loading_strategy: dataset`; data read
already split (`loading_strategy: split`) is not filtered, and the section is ignored there.
Filtering is a fundamental step when the dataset contains redundant or low-quality interactions,
or when its size exceeds the available computational resources.

By applying filters, WarpRec ensures that the resulting dataset is both computationally manageable
and more representative of the target recommendation task.

## General Configuration Format

Filtering strategies must be declared under the `filtering` section of the configuration file.
Each strategy is specified by name, followed by its parameters (if required).

```yaml
filtering:
    strategy_name_1:
        arg_name_1: value_1
    strategy_name_2:
        arg_name_1: value_1
        arg_name_2: value_2
...
```

!!! important
    - Strategies are executed **top to bottom** in the exact order they are listed.
    - A strategy can appear only once: if a name is repeated, the YAML loader silently keeps its last parameters, at the position of its first occurrence.
    - An unknown strategy name, or a missing or invalid required parameter, raises when the configuration is loaded. A key the strategy does not take is ignored.
    - The rating-based strategies also work with `rating_type: implicit`: the rating column is still read, and the 1s are stored only when the dataset is built, after filtering.

## Supported Filtering Strategies

WarpRec currently supports the following filtering strategies:

| Strategy | Category | Description |
|----------|----------|-------------|
| `MinRating` | Rating-based | Remove interactions below a rating threshold. |
| `UserAverage` | Rating-based | Keep interactions rated above the user's average rating. |
| `ItemAverage` | Rating-based | Keep interactions rated above the item's average rating. |
| `UserMin` | Frequency-based | Remove users with fewer than N interactions. |
| `UserMax` | Frequency-based | Remove users with more than N interactions. |
| `ItemMin` | Frequency-based | Remove items with fewer than N interactions. |
| `ItemMax` | Frequency-based | Remove items with more than N interactions. |
| `IterativeKCore` | Core-based | Iterative k-core filtering until convergence. |
| `NRoundsKCore` | Core-based | k-core filtering for a fixed number of rounds. |
| `UserHeadN` | Temporal | Retain the first N interactions per user. |
| `UserTailN` | Temporal | Retain the last N interactions per user. |
| `DropUser` | Entity removal | Remove specific users by ID. |
| `DropItem` | Entity removal | Remove specific items by ID. |

**1. MinRating**

Removes all interactions where the rating value is strictly below the specified threshold.

```yaml
filtering:
    MinRating:
        min_rating: 3.0
```

**2. UserAverage**

Keeps only the interactions rated strictly above the corresponding user's average rating,
so a rating equal to the mean is removed as well.

```yaml
filtering:
    UserAverage: {}   # No parameters required
```

**3. ItemAverage**

Keeps only the interactions rated strictly above the corresponding item's average rating,
so a rating equal to the mean is removed as well.

```yaml
filtering:
    ItemAverage: {}   # No parameters required
```

**4. UserMin**

Removes all interactions involving users with fewer interactions than the given threshold.

```yaml
filtering:
    UserMin:
        min_interactions: 5
```

**5. UserMax**

Removes all interactions involving users with more interactions than the given threshold.
This is particularly useful for **cold-start user analysis**.

```yaml
filtering:
    UserMax:
        max_interactions: 2
```

**6. ItemMin**

Removes all interactions involving items with fewer interactions than the given threshold.

```yaml
filtering:
    ItemMin:
        min_interactions: 5
```

**7. ItemMax**

Removes all interactions involving items with more interactions than the given threshold.
Useful for analyzing **cold-start item scenarios**.

```yaml
filtering:
    ItemMax:
        max_interactions: 2
```

**8. IterativeKCore**

Applies `UserMin` and `ItemMin` iteratively until no further interactions can be removed
(i.e., until a stable state is reached).

```yaml
filtering:
    IterativeKCore:
        min_interactions: 5
```

**9. NRoundsKCore**

Applies `UserMin` and `ItemMin` for a fixed number of iterations.
This is a simplified variant of `IterativeKCore` that does not require full convergence.

```yaml
filtering:
    NRoundsKCore:
        rounds: 3
        min_interactions: 5
```

!!! tip
    `IterativeKCore` ensures dataset stability, but may be computationally expensive.
    `NRoundsKCore` is recommended when deterministic runtime is preferred over convergence.

**10. UserHeadN**

Selects and retains the first N interactions for each user.
If timestamps are available, interactions are sorted chronologically before selection.
If no timestamps are provided, the original ordering of interactions is preserved.

```yaml
filtering:
    UserHeadN:
        num_interactions: 30
```

**11. UserTailN**

Selects and retains the last N interactions for each user.
If timestamps are available, interactions are sorted chronologically before selection.
If no timestamps are provided, the original ordering of interactions is preserved.

```yaml
filtering:
    UserTailN:
        num_interactions: 30
```

**12. DropUser**

Filter out all interactions involving specific users identified by their user IDs.

```yaml
filtering:
    DropUser:
        user_ids_to_filter: [123, 456, 789]
```

**13. DropItem**

Filter out all interactions involving specific items identified by their item IDs.

```yaml
filtering:
    DropItem:
        item_ids_to_filter: [123, 456, 789]
```

## Example Filtering Pipeline

The following example demonstrates a pipeline where:

1. All ratings below 3.0 are removed.
2. Users with fewer than 10 interactions are filtered out.

```yaml
filtering:
    MinRating:
        min_rating: 3.0
    UserMin:
        min_interactions: 10
```
