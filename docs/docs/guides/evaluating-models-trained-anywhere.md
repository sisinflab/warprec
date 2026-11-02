# 14 · Evaluating Models Trained Anywhere

A model does not have to be trained in the run that evaluates it. This guide
saves two WarpRec models, evaluates them again from their files, and puts a
recommendation list computed outside WarpRec next to them. It covers:

- what a saved model holds, and how the train pipeline saves one
  (`meta.save_model`, `writer.save_split`);
- the evaluation pipeline with `meta.load_from`: new metrics and cut-offs
  without training, with the same numbers as the model in memory, and what
  loading requires;
- `ProxyRecommender`: evaluating a file another framework wrote, its format,
  and what happens to users and items the file leaves out or WarpRec does not
  know;
- `meta.save_recs`: the file WarpRec writes, how `mask_seen` shapes it, and
  reading it back;
- significance tests between saved models and an external one.

Training is guide 9's subject and the `Evaluator` guide 12's; this guide reuses
both.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/14-evaluating-models-trained-anywhere/evaluating-models-trained-anywhere.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in about two minutes on MovieLens-100K (`guide_data.movielens_100k()` downloads and converts it exactly as guide 1 shows). Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/design.yml`](https://github.com/sisinflab/warprec/blob/main/guides/14-evaluating-models-trained-anywhere/configs/design.yml)
- [`configs/eval-checkpoints.yml`](https://github.com/sisinflab/warprec/blob/main/guides/14-evaluating-models-trained-anywhere/configs/eval-checkpoints.yml)
- [`configs/eval-compare.yml`](https://github.com/sisinflab/warprec/blob/main/guides/14-evaluating-models-trained-anywhere/configs/eval-compare.yml)
- [`configs/eval-recs.yml`](https://github.com/sisinflab/warprec/blob/main/guides/14-evaluating-models-trained-anywhere/configs/eval-recs.yml)
- [`configs/train.yml`](https://github.com/sisinflab/warprec/blob/main/guides/14-evaluating-models-trained-anywhere/configs/train.yml)

## Reference

- [Evaluation Pipeline](../pipelines/evaluation.md)
- [Cross-Framework Evaluation](../recommenders/cross-framework-evaluation.md)
