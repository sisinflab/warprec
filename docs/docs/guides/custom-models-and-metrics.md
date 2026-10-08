# 15 · Custom models and metrics

A model or a metric of your own is a class in a Python file of your own. WarpRec
imports the file, the class registers itself under a name, and from then on the
name works everywhere a built-in's does: in a configuration, in every pipeline,
next to the built-ins in an `Evaluator`. This guide writes three such files in
`user_code/` and covers:

- `general.custom_modules`: how WarpRec imports your files, and the names it
  gives them;
- a closed-form model, fitted in its constructor, whose `predict` serves both
  full and sampled evaluation;
- an iterative model: `get_dataloader`, `forward`, `training_step`, `predict`,
  trained by Lightning;
- a parameter class, so a configuration's hyperparameters are validated, and
  what happens without one;
- a metric, used bare under `metrics` and with a parameter under
  `complex_metrics`;
- all of it run by the design pipeline, then from Python next to the built-ins
  the two models re-implement, which they match exactly.

The [Extending WarpRec](https://warprec.readthedocs.io/en/latest/extending/models/)
pages describe the same interfaces in prose.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/15-custom-models-and-metrics/custom-models-and-metrics.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in one to two minutes on a laptop CPU. It reads MovieLens-100K through `guide_data.movielens_100k()`, which downloads and converts it exactly as guide 1 shows. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/design.yml`](https://github.com/sisinflab/warprec/blob/main/guides/15-custom-models-and-metrics/configs/design.yml)
- [`configs/train.yml`](https://github.com/sisinflab/warprec/blob/main/guides/15-custom-models-and-metrics/configs/train.yml)
- [`.ruff_cache/`](https://github.com/sisinflab/warprec/blob/main/guides/15-custom-models-and-metrics/.ruff_cache)
- [`user_code/`](https://github.com/sisinflab/warprec/blob/main/guides/15-custom-models-and-metrics/user_code)

## Reference

- [Custom Recommender Implementation Guide](../extending/models.md)
- [Custom Metric Implementation Guide](../extending/metrics.md)
