# Rating

**Rating metrics** are specifically designed for recommender systems that predict **explicit user ratings** (e.g., 1-5 stars). These metrics quantify the **accuracy of these numerical predictions** by comparing them to the actual user ratings.

!!! info "API Reference"

    For class signatures and source code, see the [Rating Metrics API Reference](../../api-reference/metrics/rating.md).

## MAE

**Mean Absolute Error (MAE).** The **average absolute difference** between predicted and actual ratings.

$$
\text{MAE} = \frac{1}{N} \sum_{(u,i)} |\hat{r}_{ui} - r_{ui}|
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Mean_absolute_error).

```yaml
evaluation:
    metrics: [MAE]
```

## MSE

**Mean Squared Error (MSE).** The **average of the squared differences** between predicted and actual ratings.

$$
\text{MSE} = \frac{1}{N} \sum_{(u,i)} (\hat{r}_{ui} - r_{ui})^2
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Mean_squared_error).

```yaml
evaluation:
    metrics: [MSE]
```

## RMSE

**Root Mean Squared Error (RMSE).** The **square root of the MSE**, providing an error measure in the same units as the ratings.

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{(u,i)} (\hat{r}_{ui} - r_{ui})^2}
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Root_mean_square_deviation).

```yaml
evaluation:
    metrics: [RMSE]
```
