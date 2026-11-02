# WarpRec guides

Jupyter notebooks that take WarpRec apart one piece at a time, on real public
data. Each guide downloads its dataset, prepares it, and runs every step twice:
through the Python API, and through the configuration file the pipelines read.
The notebooks are committed with their outputs, so they can be read here without
running anything.

```bash
pip install warprec jupyter        # guide 17 needs "warprec[mcp]", guide 18 "warprec[dashboard]"
jupyter lab guides/
```

Open each notebook from its own folder: its `configs/` refer to the data relative
to it. Datasets are downloaded from their publishers into `guides/data/` the first
time a guide needs them ([`guide_data.py`](guide_data.py) holds the downloaders),
and everything a guide writes goes to `guides/runs/`; both are ignored by git.

| # | Guide | What it covers |
|---|---|---|
| | **Data** | |
| 1 | [From Raw Files to a Dataset](01-from-raw-files-to-a-dataset/from-raw-files-to-a-dataset.ipynb) | Files WarpRec reads, the reader, the `Dataset`, ready-made splits and folds |
| 2 | [Reproducible Data Filtering](02-reproducible-data-filtering/reproducible-data-filtering.ipynb) | All thirteen filters, and why their order matters |
| 3 | [Leak-Free Data Splitting](03-leak-free-data-splitting/leak-free-data-splitting.ipynb) | All eight strategies, alignment, seeds, validation holdout and folds |
| 4 | [Enriching Items and Users](04-enriching-items-and-users/enriching-items-and-users.ipynb) | Item features, user and item clusters, the stash |
| 5 | [Context-Aware Data](05-context-aware-data/context-aware-data.ipynb) | Contextual fields, `Transactions`, duplicates, contextual evaluation |
| 6 | [Knowledge-Graph Data](06-knowledge-graph-data/knowledge-graph-data.ipynb) | `reader.knowledge`, coverage, the `KnowledgeGraph`, KaHFM |
| 7 | [Multimodal Item Features](07-multimodal-item-features/multimodal-item-features.ipynb) | `reader.multimodal`, file layouts, coverage, VBPR |
| 8 | [Seeded Data Loading](08-seeded-data-loading/seeded-data-loading.ipynb) | Every training and evaluation loader, negative sampling, seeding |
| | **Models and training** | |
| 9 | [One Interface, Every Model Family](09-one-interface-every-model-family/one-interface-every-model-family.ipynb) | The registry, the model contract, one model per family, the design pipeline, checkpoints |
| 10 | [Distributed Hyperparameter Search](10-distributed-hyperparameter-search/distributed-hyperparameter-search.ipynb) | The train pipeline on Ray: search spaces, strategies, schedulers, cross-validation |
| 11 | [Resumable, Measured Experiments](11-resumable-measured-experiments/resumable-measured-experiments.ipynb) | Run names, pause and resume, dashboards, the estimate and swarm pipelines |
| | **Evaluation** | |
| 12 | [Cross-Paradigm Benchmarking](12-cross-paradigm-benchmarking/cross-paradigm-benchmarking.ipynb) | The `Evaluator`, full and sampled, `mask_seen`, every metric family, significance across model families |
| 13 | [Cold-Start, Debiased and Re-Ranked Evaluation](13-cold-start-debiased-and-re-ranked-evaluation/cold-start-debiased-and-re-ranked-evaluation.ipynb) | Cold-start protocols, IPS and SNIPS estimators, MMR and Calibration |
| 14 | [Evaluating Models Trained Anywhere](14-evaluating-models-trained-anywhere/evaluating-models-trained-anywhere.ipynb) | The eval pipeline on checkpoints, `ProxyRecommender`, saved recommendations |
| | **Extending and deploying** | |
| 15 | [From Prototyping to Cluster Scaling](15-from-prototyping-to-cluster-scaling/from-prototyping-to-cluster-scaling.ipynb) | Your own model, parameter class and metric through `custom_modules`, from the design pipeline to Ray |
| 16 | [Extending the Lifecycle with Callbacks](16-extending-the-lifecycle-with-callbacks/extending-the-lifecycle-with-callbacks.ipynb) | Every callback hook, where it runs, and a pipeline written against the API |
| 17 | [From Offline Validation to Verified Deployment](17-from-offline-validation-to-verified-deployment/from-offline-validation-to-verified-deployment.ipynb) | Serving checkpoints on Ray Serve: REST, sequential and contextual requests, MCP, served lists checked against the run |
| | **Scaling** | |
| 18 | [Resource-Aware Distributed Tuning](18-resource-aware-distributed-tuning/resource-aware-distributed-tuning.ipynb) | Trial footprints, packing trials on GPUs, ASHA, DDP across two GPUs, time and energy per setup |

[`cluster/`](cluster/) holds a Ray cluster configuration for Google Cloud.

The same guides, with links to the reference pages each one relies on, are in the
[documentation](https://warprec.readthedocs.io/en/latest/guides/).
