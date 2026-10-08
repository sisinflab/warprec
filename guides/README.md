# WarpRec guides

Jupyter notebooks that take WarpRec apart one piece at a time, on real public
data. Each guide downloads its dataset, prepares it, and runs every step twice:
through the Python API, and through the configuration file the pipelines read.
The notebooks are committed with their outputs, so they can be read here without
running anything.

```bash
pip install warprec jupyter        # guide 17 needs "warprec[mcp]"
jupyter lab guides/
```

Open each notebook from its own folder: its `configs/` refer to the data relative
to it. Datasets are downloaded from their publishers into `guides/data/` the first
time a guide needs them ([`guide_data.py`](guide_data.py) holds the downloaders),
and everything a guide writes goes to `guides/runs/`; both are ignored by git.

| # | Guide | What it covers |
|---|---|---|
| | **Data** | |
| 1 | [Reading data](01-reading-data/reading-data.ipynb) | Files WarpRec reads, the reader, the `Dataset`, ready-made splits and folds |
| 2 | [Filtering](02-filtering/filtering.ipynb) | All thirteen filters, and why their order matters |
| 3 | [Splitting](03-splitting/splitting.ipynb) | All eight strategies, alignment, seeds, validation holdout and folds |
| 4 | [Side information and clusters](04-side-information-and-clusters/side-information-and-clusters.ipynb) | Item features, user and item clusters, the stash |
| 5 | [Context](05-context/context.ipynb) | Contextual fields, `Transactions`, duplicates, contextual evaluation |
| 6 | [Knowledge graphs](06-knowledge-graphs/knowledge-graphs.ipynb) | `reader.knowledge`, coverage, the `KnowledgeGraph`, KaHFM |
| 7 | [Multimodal features](07-multimodal-features/multimodal-features.ipynb) | `reader.multimodal`, file layouts, coverage, VBPR |
| 8 | [Dataloaders](08-dataloaders/dataloaders.ipynb) | Every training and evaluation loader, negative sampling, seeding |
| | **Models and training** | |
| 9 | [Model families](09-model-families/model-families.ipynb) | The registry, the model contract, one model per family, the design pipeline, checkpoints |
| 10 | [Hyperparameter search](10-hyperparameter-search/hyperparameter-search.ipynb) | The train pipeline on Ray: search spaces, strategies, schedulers, cross-validation |
| 11 | [Running experiments](11-running-experiments/running-experiments.ipynb) | Run names, pause and resume, dashboards, the estimate and swarm pipelines |
| | **Evaluation** | |
| 12 | [Evaluation](12-evaluation/evaluation.ipynb) | The `Evaluator`, full and sampled, `mask_seen`, every metric family, significance |
| 13 | [Cold start, debiasing and re-ranking](13-cold-start-debiasing-reranking/cold-start-debiasing-reranking.ipynb) | Cold-start protocols, IPS and SNIPS estimators, MMR and Calibration |
| 14 | [Evaluating existing models](14-evaluating-existing-models/evaluating-existing-models.ipynb) | The eval pipeline on checkpoints, `ProxyRecommender`, saved recommendations |
| | **Extending and deploying** | |
| 15 | [Custom models and metrics](15-custom-models-and-metrics/custom-models-and-metrics.ipynb) | Your own model, parameter class and metric through `custom_modules` |
| 16 | [Callbacks and custom pipelines](16-callbacks-and-custom-pipelines/callbacks-and-custom-pipelines.ipynb) | Every callback hook, where it runs, and a pipeline written against the API |
| 17 | [Serving](17-serving/serving.ipynb) | Serving checkpoints on Ray Serve: REST, sequential and contextual requests, MCP |

[`cluster/`](cluster/) holds a Ray cluster configuration for Google Cloud.

The same guides, with links to the reference pages each one relies on, are in the
[documentation](https://warprec.readthedocs.io/en/latest/guides/).
