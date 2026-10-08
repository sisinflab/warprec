# 11 · Running experiments

Guide 10 tuned models with the train pipeline. This guide is about running such
experiments for real, when they take hours and something gets in the way:

- naming a run, and the run-state manifest WarpRec keeps for it;
- pausing a train run with a signal and resuming it, and what a changed
  configuration does to a resume;
- what a finished run leaves on disk;
- dashboards: CodeCarbon, MLflow and Weights & Biases;
- the estimate pipeline, which sizes an experiment before it runs;
- the swarm pipeline, which tunes every model at once.

The train and swarm runs go through the command line, `python -m warprec.run
-c <config> -p <pipeline>`, started from the notebook as a subprocess: a pause
is a signal sent to a process, and that is what this guide sends.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/11-running-experiments/running-experiments.ipynb){ .md-button .md-button--primary }

## Running it

`pip install "warprec[dashboard]" jupyter` (the `dashboard` extra brings CodeCarbon, MLflow and W&B). Runs in 10 to 20 minutes on a laptop CPU, depending on its load; each run starts and stops its own local Ray instance. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/baselines.yml`](https://github.com/sisinflab/warprec/blob/main/guides/11-running-experiments/configs/baselines.yml)
- [`configs/dashboards.yml`](https://github.com/sisinflab/warprec/blob/main/guides/11-running-experiments/configs/dashboards.yml)
- [`configs/estimate.yml`](https://github.com/sisinflab/warprec/blob/main/guides/11-running-experiments/configs/estimate.yml)
- [`configs/swarm.yml`](https://github.com/sisinflab/warprec/blob/main/guides/11-running-experiments/configs/swarm.yml)

## Reference

- [Pause & Resume](../pipelines/pause-resume.md)
- [Run Configuration](../configuration/run.md)
- [Dashboard Configuration](../configuration/dashboard.md)
- [Estimate Pipeline](../pipelines/estimate.md)
- [Swarm Pipeline](../pipelines/swarm.md)
