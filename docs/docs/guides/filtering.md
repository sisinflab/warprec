# 2 · Filtering

Filtering removes interactions, users and items from the data WarpRec has read,
before it is split. This guide runs every filter on MovieLens-100K and shows
what each one keeps, with the size of the data before and after:

- where filtering runs, and the two ways to call it: the filter classes with
  `apply_filtering`, and the `filtering` section of a configuration;
- rating thresholds: `MinRating`, `UserAverage`, `ItemAverage`;
- activity bounds: `UserMin`, `UserMax`, `ItemMin`, `ItemMax`;
- k-cores: `IterativeKCore` and `NRoundsKCore`, and how they differ;
- per-user history length: `UserHeadN`, `UserTailN`;
- drop lists: `DropUser`, `DropItem`;
- why the order of the filters matters.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/02-filtering/filtering.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in a few seconds; it downloads MovieLens-100K (5 MB) into `guides/data/` if guide 1 has not. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/filtering.yml`](https://github.com/sisinflab/warprec/blob/main/guides/02-filtering/configs/filtering.yml)
- [`configs/rating-then-users.yml`](https://github.com/sisinflab/warprec/blob/main/guides/02-filtering/configs/rating-then-users.yml)
- [`configs/users-then-rating.yml`](https://github.com/sisinflab/warprec/blob/main/guides/02-filtering/configs/users-then-rating.yml)

## Reference

- [Filtering Configuration](../configuration/filtering.md)
