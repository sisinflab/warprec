# Multiobjective

**Multiobjective metrics** evaluate model performance across multiple, often conflicting, objectives simultaneously (e.g., Accuracy vs. Diversity). These metrics aggregate individual scores to assess the overall trade-off or distance from an ideal state.

!!! info "API Reference"

    For class signatures and source code, see the [Multiobjective Metrics API Reference](../../api-reference/metrics/multiobjective.md).

## EucDistance

**Euclidean Distance (EucDistance@K).** Computes the **Euclidean Distance** between the model's performance and an Utopia Point (ideal performance). A lower value indicates the model is closer to the desired objectives.

$$
\text{EucDistance} = \sqrt{\sum_{i=1}^{n} \left( s_i - u_i \right)^2}
$$

where $s_i$ is the score of sub-metric $i$ and $u_i$ is the utopia (ideal) value for that metric.

For further details, please refer to this [link](https://www.sciencedirect.com/science/article/abs/pii/S0020025516318035?fr=RR-2&ref=pdf_download&rr=9d06d2a68bb6e889) and this [paper](https://dl.acm.org/doi/10.1145/3583780.3615010).

```yaml
evaluation:
    top_k: [10, 20, 50]
    complex_metrics:
        - name: EucDistance
          params:
              metric_names: [Precision, Recall, nDCG]
              utopia_points: [1.0, 1.0, 1.0]
```

## Hypervolume

**Hypervolume (Hypervolume@K).** Measures the **volume of the objective space** dominated by the model's performance relative to a Nadir Point (worst-case reference). A higher volume indicates a better trade-off between metrics.

$$
\text{HV} = \prod_{i=1}^{n} \max\!\left(0,\; d_i\right), \quad d_i = \begin{cases} s_i - n_i & \text{if higher is better} \\ n_i - s_i & \text{otherwise} \end{cases}
$$

where $s_i$ is the score of sub-metric $i$ and $n_i$ is the nadir (worst-case) reference value.

For further details, please refer to this [book](https://link.springer.com/chapter/10.1007/978-3-540-70928-2_64) and this [paper](https://dl.acm.org/doi/10.1145/3583780.3615010).

```yaml
evaluation:
    top_k: [10, 20, 50]
    complex_metrics:
        - name: Hypervolume
          params:
              metric_names: [Precision, Recall, nDCG]
              nadir_points: [0.0, 0.0, 0.0]
              higher_is_better: [True, True, True]
```
