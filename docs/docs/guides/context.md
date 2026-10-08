# 5 · Context

A contextual dataset records the situation of each interaction next to the user
and the item: when it happened, on what device, in which mood. This guide derives
such fields from MovieLens-100K and follows them through WarpRec:

- declaring contextual columns, and the three kinds of field WarpRec reads them
  as: categorical, numeric and multi-valued;
- how each field is encoded, and where to read the encoding back;
- `Transactions` (one row per record) next to `Interactions` (the collapsed
  matrix), and the `duplicates` policy that decides how the matrix collapses;
- `training.sequence_pooling` for multi-valued fields;
- the contextual evaluation, one ranking per (user, item, context) row, and the
  `evaluation.mask_seen` policies;
- training a factorization machine on all four fields with the design pipeline,
  evaluating it in full and sampled mode, and comparing the pooling policies.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/05-context/context.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in under two minutes on a laptop CPU, most of it training the model five times. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/context.yml`](https://github.com/sisinflab/warprec/blob/main/guides/05-context/configs/context.yml)
- [`configs/fm-sampled.yml`](https://github.com/sisinflab/warprec/blob/main/guides/05-context/configs/fm-sampled.yml)
- [`configs/fm.yml`](https://github.com/sisinflab/warprec/blob/main/guides/05-context/configs/fm.yml)

## Reference

- [Context-Aware Recommenders](../recommenders/context.md)
- [Training Configuration](../configuration/training.md)
- [Evaluation Configuration](../configuration/evaluation.md)
