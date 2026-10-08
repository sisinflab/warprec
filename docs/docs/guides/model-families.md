# 9 · Model families

WarpRec ships 87 models, and every one of them is found, built, trained, saved
and loaded the same way. This guide shows that common shape and trains one model
of each family on the same split. It covers:

- the model registry: how the models are organised, how a name in a
  configuration becomes a class, and how its hyperparameters are validated;
- the constructor every model shares, and the difference between models that
  fit in their constructor and models that a Lightning trainer trains;
- one model per family (unpersonalized, collaborative closed-form and
  iterative, sequential, content-based, hybrid) evaluated together;
- the design pipeline, which does all of the above from one configuration;
- saving a model with `get_state()` and restoring it with `from_checkpoint`.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/09-model-families/model-families.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in about two minutes on a laptop CPU, most of it training BPR and SASRec twice. It reads MovieLens-100K through `guide_data.movielens_100k()`, which downloads and converts it exactly as guide 1 shows. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/design.yml`](https://github.com/sisinflab/warprec/blob/main/guides/09-model-families/configs/design.yml)

## Reference

- [Module Overview](../recommenders/index.md)
- [Design Pipeline](../pipelines/design.md)
- [Models Configuration](../configuration/models.md)
