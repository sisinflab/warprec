# Serving Models

A model saved by the [training pipeline](../pipelines/training.md) can be served directly from its checkpoint with one command and one configuration file. Serving runs on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html):

- concurrent requests to a model are **batched** into one forward pass;
- every model runs in its own **replicas**, which can be scaled by hand or by an autoscaling policy;
- models can share a GPU, each asking for a fraction of it;
- the same configuration runs on a laptop, joins an existing Ray cluster, or is **exported** for `serve deploy` and KubeRay.

General, sequential, graph-based and context-aware models are all served from the checkpoint alone. The dataset, the experiment directory and the training configuration are not needed.

## Install

```bash
pip install "warprec[serving]"   # the REST API
pip install "warprec[mcp]"       # the REST API and the MCP endpoint for LLM agents
```

## From Training to Serving

**1. Save the model.** Set `save_model` in the model's `meta` section of the training configuration:

```yaml
models:
  SASRec:
    meta:
      save_model: true
    embedding_size: 64
    max_seq_len: 50
    # ...
```

```bash
python -m warprec.run -c train.yml -p train
```

The best model of the run is written to `{local_experiment_path}/{dataset_name}/serialized/{Model}_{name}.pth`. Besides the weights, the file carries everything serving needs from the training data: the items each user has seen, each user's recent history for a sequential model, and the context vocabulary for a context-aware one.

**2. Describe what to serve** in a serving configuration (see [Serving Configuration](../configuration/serving.md) for every key):

```yaml
server:
  port: 8000
  api_key: change-me
endpoints:
  - name: sasrec
    checkpoint: experiments/ml-1m/serialized/SASRec_brave-otter.pth
    item_metadata:
      path: data/ml-1m/movies.dat
      sep: "::"
      header: false
      encoding: latin-1
```

**3. Serve it.**

```bash
python -m warprec.serve -c serve.yml
# or, once installed with pip:
warprec.serve -c serve.yml
```

The command blocks until it is stopped. Ctrl-C at a terminal and `SIGTERM` from a process manager or a container both shut it down cleanly.

## API

Every route but `/healthz` requires the `X-API-Key` header when an API key is configured.

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness check. |
| GET | `/v1/models` | The served models and what each accepts. |
| GET | `/v1/models/{name}` | One served model. |
| POST | `/v1/models/{name}/recommend` | Top-k recommendations. |
| POST | `/v1/models/{name}/score` | Scores of given candidate items, for re-ranking. |
| * | `/mcp` | MCP tools, when `server.mcp` is `true`. |

**List the models.**

```bash
curl -H "X-API-Key: change-me" localhost:8000/v1/models
```

```json
[{"name": "sasrec", "model": "SASRec", "kind": "sequential", "n_users": 6040, "n_items": 3706,
  "needs_user": false, "warprec_version": "1.16.0", "params": {"embedding_size": 64, "...": "..."},
  "context": null}]
```

**Recommend for a known user.** Ids are the dataset's own, as numbers or strings.

```bash
curl -X POST localhost:8000/v1/models/sasrec/recommend -H "X-API-Key: change-me" \
     -H "Content-Type: application/json" -d '{"user_id": 1, "k": 3}'
```

```json
{"model": "sasrec", "fallback": false,
 "items": [{"item_id": 2858, "score": 7.91, "name": "American Beauty (1999)"},
           {"item_id": 1196, "score": 7.64, "name": "Star Wars: Episode V - The Empire Strikes Back (1980)"},
           {"item_id": 260, "score": 7.55, "name": "Star Wars: Episode IV - A New Hope (1977)"}]}
```

**Recommend for a session.** A sequential model also answers from a history of items, oldest first, without a known user. With an item catalogue, items may be given by their exact name.

```bash
curl -X POST localhost:8000/v1/models/sasrec/recommend -H "X-API-Key: change-me" \
     -H "Content-Type: application/json" \
     -d '{"history": ["Toy Story (1995)", "Aladdin (1992)"], "k": 5, "exclude": [2355]}'
```

**Score candidates** that another system retrieved. Scores come back in request order, and nothing is masked.

```bash
curl -X POST localhost:8000/v1/models/sasrec/score -H "X-API-Key: change-me" \
     -H "Content-Type: application/json" -d '{"user_id": 1, "items": [1, 260, 2858]}'
```

```json
{"model": "sasrec", "scores": [{"item_id": 1, "score": 5.12, "name": "Toy Story (1995)"},
                               {"item_id": 260, "score": 7.55, "name": "Star Wars: Episode IV - A New Hope (1977)"},
                               {"item_id": 2858, "score": 7.91, "name": "American Beauty (1999)"}]}
```

Errors carry a `detail` message and a status: `404` for an unknown model or user, `422` for a request the model cannot answer, `401` for a missing or wrong API key.

## How Requests Are Answered

- **Ids.** Requests and responses use the dataset's own user and item ids. They are matched as strings, so `1` and `"1"` are the same user.
- **Seen items.** With `mask_seen` (the default), items the user interacted with in training are left out, as are the items of a request's `history` and `exclude`. When masking leaves fewer than `k` items, fewer are returned: a masked item is never used to pad the list.
- **Sequential models.**
    - A `user_id` alone uses the user's training history.
    - A `history` alone is an anonymous session.
    - Both together score the given history.
    - Caser, FOSSIL and STAN mix a user embedding into the sequence, so they need a known `user_id` with a history (`needs_user: true` in the model listing).
    - A `history` sent to a non-sequential model is refused.
- **Unknown users.** A user absent from the training data gets `404`, or, with `unknown_user: popular`, the items most interacted with in training and `fallback: true` in the response.
- **Context-aware models** (AFM, DCN, DCNv2, DeepFM, FM, NFM, Wide&Deep, xDeepFM) need the situation of every request.
    - `context` is a dictionary by field, or a list in the field order the model listing shows.
    - The listing also shows each field's type and the values it knows.
    - A categorical field takes one value seen in training, a multi-valued field takes a list of them, and a numeric field takes a number.
    - A value never seen in training is refused, listing the known ones, rather than silently scored as unknown.
    - Seen-item masking stays per user, not per context, as when recommendations are written to file.

```bash
curl -X POST localhost:8000/v1/models/fm/recommend -H "X-API-Key: change-me" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 12, "k": 5, "context": {"daytime": "morning", "weather": "sunny", "isweekend": "workday", "homework": "home"}}'
```

## MCP

With `server.mcp: true` and the `mcp` extra installed, the same server exposes two [Model Context Protocol](https://modelcontextprotocol.io/) tools under `/mcp`:

- `list_models` tells an agent which models exist and what each accepts.
- `recommend` asks one of them, by user, session or context.

The API key protects `/mcp` as well. A client configuration looks like this:

```json
{"mcpServers": {"warprec": {"url": "http://localhost:8000/mcp/", "headers": {"X-API-Key": "change-me"}}}}
```

## Scaling and GPUs

Each endpoint is its own Ray Serve deployment, configured independently:

- **batching**: concurrent requests are grouped into one forward pass of up to `max_batch_size` requests, waiting at most `batch_wait_timeout_s` for a batch to fill. A replica accepts as many requests at once as a batch holds, unless `deployment.max_ongoing_requests` says otherwise.
- **deployment.num_replicas** fixes the number of replicas, while **deployment.autoscaling_config** lets Ray Serve add and remove them with the load.
- **deployment.ray_actor_options** sets the resources of each replica. With `device: cuda` a replica asks for one GPU; `num_gpus: 0.25` places four replicas on one GPU instead.
- **server.ray_address: auto** joins the Ray cluster the machine belongs to, rather than starting a local one.

## Deploying to a Cluster

```bash
python -m warprec.serve -c serve.yml --export serve_app.yaml
serve deploy serve_app.yaml
```

`--export` writes a standard Ray Serve configuration file instead of starting the server:

- Its application points at `warprec.serving.app:app_builder`, with the serving configuration inlined.
- Paths become absolute, so the checkpoints must exist at the same paths on the cluster.
- The API key is left out of the file; set `WARPREC_API_KEY` in the cluster's environment instead.

The `applications` section of the file can also be pasted into a KubeRay `RayService`.

## Security

- **Checkpoints are pickles.** Loading one can run code, so only serve files from a trusted source.
- **Set an API key** whenever the server is reachable beyond the local machine. The `WARPREC_API_KEY` environment variable keeps it out of configuration files.

## Limitations

- Checkpoints saved before this version serve without seen-item masking or popularity fallback. Graph-based and context-aware models among them must be saved again.
- Checkpoints are read from the local file system.
