# 1 · Reading your dataset

This guide takes a public dataset from its download to a WarpRec `Dataset`, the
object every model trains on and every evaluation reads. It covers:

- preparing a file WarpRec can read, and the shapes of file it accepts;
- the `LocalReader`: separators, headers, column names and types, parquet,
  and the two dataframe backends;
- building a `Dataset` and reading what it holds: id mappings, the interaction
  matrix and the `info()` every model is built from;
- loading data that is already split, including validation folds, and writing
  a split out so it can be loaded that way later.

Every step is done twice: through the Python API, then through the `reader`
section of a configuration file, which is what the pipelines run.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/01-reading-data/reading-data.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in under a minute; it downloads MovieLens-100K (5 MB) into `guides/data/`. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/labels.yml`](https://github.com/sisinflab/warprec/blob/main/guides/01-reading-data/configs/labels.yml)
- [`configs/reader-dataset.yml`](https://github.com/sisinflab/warprec/blob/main/guides/01-reading-data/configs/reader-dataset.yml)
- [`configs/reader-folds.yml`](https://github.com/sisinflab/warprec/blob/main/guides/01-reading-data/configs/reader-folds.yml)
- [`configs/reader-split.yml`](https://github.com/sisinflab/warprec/blob/main/guides/01-reading-data/configs/reader-split.yml)

## Reference

- [Reader Configuration](../configuration/reader.md)
- [Readers](../data-management/reader.md)
- [Writers](../data-management/writer.md)
