# Guides

The guides are Jupyter notebooks that take WarpRec apart one piece at a time, on
real public data. Each one downloads its dataset, prepares it, and runs every step
twice: through the Python API, where you can see what each object holds, and
through the configuration file the pipelines read. The notebooks are committed
with their outputs, so they can be read on GitHub without running anything.

They live in the [`guides/`](https://github.com/sisinflab/warprec/tree/main/guides)
folder of the repository, one folder per guide, each with its notebook and the
configuration files it runs.

## Running them

```bash
git clone https://github.com/sisinflab/warprec.git
cd warprec
pip install warprec jupyter        # guide 17 needs "warprec[mcp]"
jupyter lab guides/
```

Open each notebook from its own folder: the configuration files refer to the data
relative to it. Datasets are downloaded from their publishers into `guides/data/`
the first time a guide needs them, and everything the guides write goes to
`guides/runs/`; neither is committed. Most guides use MovieLens-100K and run in a
minute or two on a laptop CPU; the ones that run the training pipeline on a local
Ray instance or start a server (10, 11, 16 and 17) take five to fifteen minutes.

## The guides

| # | Guide | What it covers |
|---|---|---|
| | **Data** | |
| 1 | [Reading data](reading-data.md) | Files WarpRec reads, the reader, the `Dataset`, ready-made splits and folds |
| 2 | [Filtering](filtering.md) | All thirteen filters, and why their order matters |
| 3 | [Splitting](splitting.md) | All eight strategies, alignment, seeds, validation holdout and folds |
| 4 | [Side information and clusters](side-information-and-clusters.md) | Item features, user and item clusters, the stash |
| 5 | [Context](context.md) | Contextual fields, `Transactions`, duplicates, contextual evaluation |
| 6 | [Knowledge graphs](knowledge-graphs.md) | `reader.knowledge`, coverage, the `KnowledgeGraph`, KaHFM |
| 7 | [Multimodal features](multimodal-features.md) | `reader.multimodal`, file layouts, coverage, VBPR |
| 8 | [Dataloaders](dataloaders.md) | Every training and evaluation loader, negative sampling, seeding |
| | **Models and training** | |
| 9 | [Model families](model-families.md) | The registry, the model contract, one model per family, the design pipeline, checkpoints |
| 10 | [Hyperparameter search](hyperparameter-search.md) | The train pipeline on Ray: search spaces, strategies, schedulers, cross-validation |
| 11 | [Running experiments](running-experiments.md) | Run names, pause and resume, dashboards, the estimate and swarm pipelines |
| | **Evaluation** | |
| 12 | [Evaluation](evaluation.md) | The `Evaluator`, full and sampled, `mask_seen`, every metric family, significance |
| 13 | [Cold start, debiasing and re-ranking](cold-start-debiasing-reranking.md) | Cold-start protocols, IPS and SNIPS estimators, MMR and Calibration |
| 14 | [Evaluating existing models](evaluating-existing-models.md) | The eval pipeline on checkpoints, `ProxyRecommender`, saved recommendations |
| | **Extending and deploying** | |
| 15 | [Custom models and metrics](custom-models-and-metrics.md) | Your own model, parameter class and metric through `custom_modules` |
| 16 | [Callbacks and custom pipelines](callbacks-and-custom-pipelines.md) | Every callback hook, where it runs, and a pipeline written against the API |
| 17 | [Serving](serving.md) | Serving checkpoints on Ray Serve: REST, sequential and contextual requests, MCP |

The guides build on each other in this order, but each one runs on its own.
The [Ray cluster configuration](https://github.com/sisinflab/warprec/tree/main/guides/cluster)
for Google Cloud sits next to them; [Cluster Management](../cloud/cluster-management.md)
explains it.
