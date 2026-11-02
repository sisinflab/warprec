# 17 · From Offline Validation to Verified Deployment

A model saved by WarpRec can be put behind an HTTP API with one command and one
configuration file, without the dataset or the training configuration. This
guide trains three models on MovieLens-100K, serves them together and queries
them the way an application or an LLM agent would. It covers:

- what `meta.save_model` writes for serving, from the train pipeline and from
  the writer's `write_model`: seen items, histories, context vocabularies and
  the facts of the training run;
- the serving configuration (`serve.yml`): endpoints, item names and genres
  from `items.tsv`, unknown users, batching, the API key, and how the schema
  rejects a key it does not know;
- starting `warprec.serve` and the REST API: model cards, recommendations for
  a user, for an anonymous session and in a context, scoring your own
  candidates, item search, popular items, filters, and the errors;
- checking that a served model recommends what the run evaluated: the same
  lists, re-ranker included, and the same nDCG, where scoring it by hand does
  not;
- the MCP endpoint, called with an MCP client as an agent would;
- `--export`, which turns the same file into a Ray Serve config for a cluster.

Guide 10 covers the train pipeline, guide 5 contexts and guide 14 the
checkpoint file itself.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/17-from-offline-validation-to-verified-deployment/from-offline-validation-to-verified-deployment.ipynb){ .md-button .md-button--primary }

## Running it

`pip install "warprec[mcp]" jupyter` (the `mcp` extra includes the `serving` one). Runs in eight to twelve minutes on a laptop CPU, most of it training; the server listens on port 8117 while the guide runs. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/context.yml`](https://github.com/sisinflab/warprec/blob/main/guides/17-from-offline-validation-to-verified-deployment/configs/context.yml)
- [`configs/serve.yml`](https://github.com/sisinflab/warprec/blob/main/guides/17-from-offline-validation-to-verified-deployment/configs/serve.yml)
- [`configs/train.yml`](https://github.com/sisinflab/warprec/blob/main/guides/17-from-offline-validation-to-verified-deployment/configs/train.yml)
- [`configs/verify.yml`](https://github.com/sisinflab/warprec/blob/main/guides/17-from-offline-validation-to-verified-deployment/configs/verify.yml)

## Reference

- [Serving Models](../serving/index.md)
- [Serving Configuration](../configuration/serving.md)
- [Deploying Served Models](../cloud/deployment.md)
