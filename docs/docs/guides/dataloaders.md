# 8 · Dataloaders

A model never reads the `Dataset` directly while it trains: it asks one of the
dataset's entities for a PyTorch `DataLoader`, and the evaluator does the same
for the evaluation split. This guide opens every loader WarpRec builds and looks
at what one batch holds. It covers:

- the configuration keys that reach the loaders, run through
  `load_design_configuration` and `initialize_datasets`;
- the four loaders over the interaction matrix (dense user rows, pointwise,
  contrastive triplets, positive pairs) and which models read each;
- the row loader that keeps contexts attached to their interaction;
- the five session loaders the sequential models train on, and the history
  lookup they are scored with;
- negative sampling: `uniform` against `popularity` and `neg_alpha`, measured;
- seeding: what a seed fixes and what changes between epochs;
- the four evaluation loaders: full, sampled, and their contextual variants.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/08-dataloaders/dataloaders.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in under a minute on MovieLens-100K (`guide_data.movielens_100k()` downloads and converts it exactly as guide 1 shows). Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/loaders.yml`](https://github.com/sisinflab/warprec/blob/main/guides/08-dataloaders/configs/loaders.yml)

## Reference

- [Training Configuration](../configuration/training.md)
- [Evaluation Configuration](../configuration/evaluation.md)
