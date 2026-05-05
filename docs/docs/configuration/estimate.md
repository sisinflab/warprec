# Estimate Configuration

The **Estimate Configuration** section controls how WarpRec samples batches during the `estimate` pipeline.

This block does not change the model architecture or evaluation protocol. Instead, it defines how much lightweight profiling WarpRec performs before extrapolating time and memory usage.

## Available Keywords

- **train_batches**: Number of measured training batches used to estimate iterative-model training cost. Defaults to `100`.
- **eval_batches**: Number of measured evaluation batches used to estimate inference and evaluation cost. Defaults to `100`.
- **warmup_batches**: Number of initial batches ignored before measurements start. Defaults to `10`.

!!! important
    `train_batches` and `eval_batches` must be strictly positive. `warmup_batches` must be greater than or equal to zero.

## Example Estimate Configuration

```yaml
estimate:
    warmup_batches: 10
    train_batches: 100
    eval_batches: 100
```

## Practical Guidance

- Increase `train_batches` or `eval_batches` when you want more stable estimates on highly variable workloads.
- Reduce them when you want faster, cheaper pre-flight checks.
- Use a small positive `warmup_batches` value to avoid measuring one-time startup effects such as kernel warmup or initial memory allocation.
