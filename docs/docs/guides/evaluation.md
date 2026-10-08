# 12 · Evaluation

Every pipeline ends by handing a trained model to an `Evaluator`. This guide
drives the evaluator directly, on three fast models fitted to MovieLens-100K,
and shows the `evaluation` section of a configuration next to each step. It
covers:

- the `Evaluator`: metrics × cutoffs in one pass, what `compute_results()`
  returns, and how a per-user vector becomes the number a run reports;
- full against sampled evaluation, with `uniform` and `popularity` negatives;
- `mask_seen`, and what a ranking looks like when training items stay in it;
- a tour of the metric families, including the ones that read side
  information, clusters or parameters;
- per-user results, `save_per_user`, `validation_metric` and
  `full_evaluation_on_report`;
- significance tests between models, with corrections for multiple testing.

Running whole pipelines is guide 10's subject; models are guide 9's.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/12-evaluation/evaluation.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in about a minute on MovieLens-100K (`guide_data.movielens_100k()` downloads and converts it exactly as guide 1 shows). Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/complete.yml`](https://github.com/sisinflab/warprec/blob/main/guides/12-evaluation/configs/complete.yml)
- [`configs/evaluation.yml`](https://github.com/sisinflab/warprec/blob/main/guides/12-evaluation/configs/evaluation.yml)
- [`configs/mask-seen.yml`](https://github.com/sisinflab/warprec/blob/main/guides/12-evaluation/configs/mask-seen.yml)
- [`configs/metrics.yml`](https://github.com/sisinflab/warprec/blob/main/guides/12-evaluation/configs/metrics.yml)
- [`configs/sampled.yml`](https://github.com/sisinflab/warprec/blob/main/guides/12-evaluation/configs/sampled.yml)

## Reference

- [Evaluation System](../evaluation/index.md)
- [Sampled Evaluation](../evaluation/sampled.md)
- [Statistical Significance](../evaluation/statistical-significance.md)
- [Evaluation Configuration](../configuration/evaluation.md)
