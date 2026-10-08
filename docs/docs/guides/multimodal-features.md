# 7 · Multimodal features

Multimodal models score from precomputed item vectors (an image encoder's
output, a text embedding) beside the interactions. This guide gives
MovieLens-100K two such modalities and covers:

- producing a feature matrix for the items, here a text embedding of the
  title and genres;
- the three file layouts `reader.multimodal` reads (a numpy array with its
  row-order file, a tabular file, a parquet file) and why the array needs the
  row-order file;
- what happens to items a feature file does not cover, and `normalize`;
- the `MultiModalFeatures` object the dataset carries, and what models see
  of it through `info()`;
- choosing modalities per model, and a short VBPR run next to BPR.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/07-multimodal-features/multimodal-features.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in one to two minutes on a laptop CPU; it downloads MovieLens-100K (5 MB) into `guides/data/`. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/multimodal.yml`](https://github.com/sisinflab/warprec/blob/main/guides/07-multimodal-features/configs/multimodal.yml)
- [`configs/train.yml`](https://github.com/sisinflab/warprec/blob/main/guides/07-multimodal-features/configs/train.yml)
- [`configs/unknown-modality.yml`](https://github.com/sisinflab/warprec/blob/main/guides/07-multimodal-features/configs/unknown-modality.yml)

## Reference

- [Multimodal Recommenders](../recommenders/multimodal.md)
- [Reader Configuration](../configuration/reader.md)
