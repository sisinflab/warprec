# 10 · Hyperparameter search

The train pipeline turns every hyperparameter written as a list into a search
space, runs one Ray Tune trial per configuration, keeps the best one, tests it
and writes everything to disk. This guide covers:

- the search-space syntax, and what the configuration accepts, rewrites or
  rejects;
- the search strategies (`grid`, `random`, `hopt`, `optuna`, `bohb`) and the
  schedulers that stop trials early (`fifo`, `asha`, `median`, `bohb`), next to
  per-trial early stopping;
- a closed-form sweep (EASE over `l2`) and an iterative one (BPR with Optuna
  and ASHA) in one run, and how to read what the writer produced;
- cross-validation: scoring each configuration as a mean over folds, and
  retraining the winner for the epochs the folds settled on;
- the optimizer, learning-rate scheduler, precision, gradient clipping and
  per-trial resources.

The sweeps are deliberately tiny (three EASE values, six BPR trials of at most
eight epochs on 16- or 32-dimensional embeddings, six cross-validation trials):
they show the mechanics, they do not tune anything.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/10-hyperparameter-search/hyperparameter-search.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in five to ten minutes on a laptop CPU, almost all of it Ray starting a worker process per trial. It reads MovieLens-100K through `guide_data.movielens_100k()`, which downloads and converts it exactly as guide 1 shows. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/bad-space.yml`](https://github.com/sisinflab/warprec/blob/main/guides/10-hyperparameter-search/configs/bad-space.yml)
- [`configs/cross-validation.yml`](https://github.com/sisinflab/warprec/blob/main/guides/10-hyperparameter-search/configs/cross-validation.yml)
- [`configs/sweep.yml`](https://github.com/sisinflab/warprec/blob/main/guides/10-hyperparameter-search/configs/sweep.yml)

## Reference

- [Training Pipeline](../pipelines/training.md)
- [Models Configuration](../configuration/models.md)
