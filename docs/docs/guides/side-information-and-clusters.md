# 4 · Side information, clusters and the stash

Interactions are not the only thing a `Dataset` can carry. This guide adds three
more kinds of data to MovieLens-100K and follows each one to the models and
metrics that read it:

- **item side information** (genres and decade of release): the file WarpRec
  expects, how it trims the catalogue, `keep_unseen_items`, and the two views
  the `Dataset` builds from it, a feature matrix and per-column indices;
- **user and item clusters** (users by gender, items by genre): the files, and
  how the cluster labels are renumbered for the fairness metrics;
- **the stash**: any structure of your own, handed to every model and every
  metric as a keyword argument.

It ends by running two side-information models next to their collaborative
counterparts, evaluated with metrics that read the features, the clusters and
the stash.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/04-side-information-and-clusters/side-information-and-clusters.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in about a minute. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/clusters.yml`](https://github.com/sisinflab/warprec/blob/main/guides/04-side-information-and-clusters/configs/clusters.yml)
- [`configs/models.yml`](https://github.com/sisinflab/warprec/blob/main/guides/04-side-information-and-clusters/configs/models.yml)
- [`configs/side-keep-unseen.yml`](https://github.com/sisinflab/warprec/blob/main/guides/04-side-information-and-clusters/configs/side-keep-unseen.yml)
- [`configs/side.yml`](https://github.com/sisinflab/warprec/blob/main/guides/04-side-information-and-clusters/configs/side.yml)

## Reference

- [Reader Configuration](../configuration/reader.md)
- [Stash](../data-management/stash.md)
- [Hybrid Recommenders](../recommenders/hybrid.md)
