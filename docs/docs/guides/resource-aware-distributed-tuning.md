# 18 · Resource-Aware Distributed Tuning

A search on GPUs is fastest when each model gets the resources it can use:
a light model wastes a whole GPU, a heavy one is slowed down by sharing it.
This guide runs the same searches on one machine with two GPUs, changing only
the resource lines of their configurations, and measures what each setting
costs and buys. It covers:

- what Ray sees on the machine, and how many trials WarpRec runs at once for
  a given `cpu_per_trial` and `gpu_per_trial`;
- one GPU against two, and two trials packed on each GPU
  (`gpu_per_trial: 0.5`), for a light model (BPR) and a heavy one (LightGCN);
- pruning the weaker trials with ASHA;
- one trial trained on two GPUs with DDP (`gpu_per_trial: 2`);
- a swarm run giving each model its own resources;
- the energy and carbon of every run, measured with CodeCarbon.

Guide 10 covers search spaces, strategies and schedulers, guide 11 the swarm
pipeline and the dashboards, guide 15 moving a model from the design pipeline
to Ray.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/resource-aware-distributed-tuning.ipynb){ .md-button .md-button--primary }

## Running it

a Linux machine with two CUDA GPUs (this notebook was run on two RTX 4090) and `pip install "warprec[dashboard]" jupyter`. Runs in about an hour. MovieLens-100K is small for these GPUs: the numbers show how the settings change a search, not what the hardware can do. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/bpr-asha.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/bpr-asha.yml)
- [`configs/bpr-packed.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/bpr-packed.yml)
- [`configs/bpr.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/bpr.yml)
- [`configs/lightgcn-ddp.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/lightgcn-ddp.yml)
- [`configs/lightgcn-packed.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/lightgcn-packed.yml)
- [`configs/lightgcn-single.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/lightgcn-single.yml)
- [`configs/lightgcn.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/lightgcn.yml)
- [`configs/swarm.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/swarm.yml)

## Reference

- [General Configuration](../configuration/general.md)
- [Models Configuration](../configuration/models.md)
- [Swarm Pipeline](../pipelines/swarm.md)
- [Dashboard Configuration](../configuration/dashboard.md)
