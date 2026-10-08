# 3 · Splitting

A model is trained on one part of the data and evaluated on another. This guide
shows how WarpRec makes those parts from a single file:

- the eight splitting strategies, the parameters each reads, and a check on
  MovieLens-100K that each split has the property it promises;
- alignment: what happens to test rows whose user or item training never saw;
- seeds, and what else a split depends on;
- a validation split on top of the test split, as a holdout or as k folds, and
  which rows each resulting dataset trains and evaluates on.

Everything is done through the `Splitter` API, then through the `splitter`
section of a configuration file.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/03-splitting/splitting.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in under a minute; it downloads MovieLens-100K (5 MB) into `guides/data/` if guide 1 has not. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/leave-one-out.yml`](https://github.com/sisinflab/warprec/blob/main/guides/03-splitting/configs/leave-one-out.yml)
- [`configs/validation-holdout.yml`](https://github.com/sisinflab/warprec/blob/main/guides/03-splitting/configs/validation-holdout.yml)
- [`configs/validation-kfold.yml`](https://github.com/sisinflab/warprec/blob/main/guides/03-splitting/configs/validation-kfold.yml)

## Reference

- [Splitter Configuration](../configuration/splitting.md)
- [Cold-Start Protocols](../evaluation/cold-start.md)
