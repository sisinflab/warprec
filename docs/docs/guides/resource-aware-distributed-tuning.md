# 18 · Resource-Aware Distributed Tuning

A hyperparameter search explores many configurations, and scheduling every
trial the same way wastes hardware: a lightweight model leaves a whole GPU
mostly idle, a compute-heavy one is slowed down when it has to share one.
WarpRec gives each model the resources that fit its computational profile,
declared in two lines of its configuration. This guide runs the same searches
on one machine with two GPUs, changing only those lines, and measures what
each setting costs and buys. It covers:

- what Ray sees on the machine, and how many trials WarpRec runs at once for
  a given `cpu_per_trial` and `gpu_per_trial`;
- one GPU against two, and two trials packed on each GPU
  (`gpu_per_trial: 0.5`), for a lightweight model (BPR) and a compute-heavy
  one (LightGCN);
- pruning the weaker trials with ASHA, on top of packing;
- one trial trained on two GPUs with DDP (`gpu_per_trial: 2`), for each model;
- the energy and carbon of every run, measured with CodeCarbon.

Guide 10 covers search spaces, strategies and schedulers, guide 11 the swarm
pipeline and the dashboards, guide 15 moving a model from the design pipeline
to Ray.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/resource-aware-distributed-tuning.ipynb){ .md-button .md-button--primary }

## Running it

a Linux machine with two CUDA GPUs (this notebook was run on two RTX 4090 and 32 CPU cores) and `pip install "warprec[dashboard]" jupyter`. Runs in about 25 minutes. MovieLens-100K is small for these GPUs: even LightGCN, given large embeddings and four layers of propagation, leaves them mostly idle. The numbers show how to measure what each setting buys, and why the answer depends on how much work a trial gives its GPU. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/bpr-asha.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/bpr-asha.yml)
- [`configs/bpr-ddp.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/bpr-ddp.yml)
- [`configs/bpr-packed.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/bpr-packed.yml)
- [`configs/bpr-single.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/bpr-single.yml)
- [`configs/bpr.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/bpr.yml)
- [`configs/lightgcn-ddp.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/lightgcn-ddp.yml)
- [`configs/lightgcn-packed.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/lightgcn-packed.yml)
- [`configs/lightgcn-single.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/lightgcn-single.yml)
- [`configs/lightgcn.yml`](https://github.com/sisinflab/warprec/blob/main/guides/18-resource-aware-distributed-tuning/configs/lightgcn.yml)

## Reference

- [General Configuration](../configuration/general.md)
- [Models Configuration](../configuration/models.md)
- [Dashboard Configuration](../configuration/dashboard.md)
