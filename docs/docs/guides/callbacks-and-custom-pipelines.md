# 16 · Callbacks and custom pipelines

There are two ways to add code of your own to an experiment. A **callback**
keeps the configuration-driven pipelines and runs your code at fixed points
of them; a **pipeline of your own** calls WarpRec's components from Python
and decides everything itself. This guide covers:

- a `WarpRecCallback` subclass in `user_code/`, configured under
  `general.callback`: `on_data_reading` changes the frame just read,
  `on_dataset_creation` fills the stash of every dataset, `on_training_complete`
  and `on_evaluation_complete` see each model, and `on_trial_result`, a Ray
  Tune hook, records every validation report;
- where each hook runs: all of them in your process for the design pipeline,
  but split between a Ray worker and your process for the train pipeline, and
  what that means for state the callback keeps;
- a custom metric that reads what the callback put in the stash, end to end;
- a pipeline written against the Python API with no configuration at all:
  read, filter, split, build the datasets, select hyperparameters on
  validation, train an iterative model with Lightning as the design pipeline
  does, test, run a significance test and write the results.

The [Callbacks](https://warprec.readthedocs.io/en/latest/extending/callbacks/)
and [Stash](https://warprec.readthedocs.io/en/latest/extending/stash/) pages
describe the same interfaces in prose.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/16-callbacks-and-custom-pipelines/callbacks-and-custom-pipelines.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in three to five minutes on a laptop CPU, most of it Ray starting processes for the train pipeline. It reads MovieLens-100K through `guide_data.movielens_100k()`, which downloads and converts it exactly as guide 1 shows. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/design.yml`](https://github.com/sisinflab/warprec/blob/main/guides/16-callbacks-and-custom-pipelines/configs/design.yml)
- [`configs/train.yml`](https://github.com/sisinflab/warprec/blob/main/guides/16-callbacks-and-custom-pipelines/configs/train.yml)
- [`user_code/`](https://github.com/sisinflab/warprec/blob/main/guides/16-callbacks-and-custom-pipelines/user_code)

## Reference

- [Callbacks](../extending/callbacks.md)
- [Add Data to the Stash](../extending/stash.md)
- [Callbacks](../pipelines/callbacks.md)
