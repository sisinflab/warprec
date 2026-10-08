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

Serving a model needs the same extras as training it: graph-based models such as LightGCN or NGCF also need `graph`, for example `pip install "warprec[serving,graph]"`. A missing one shows up as a `ModuleNotFoundError` in the replica's log, and the server stops with `Deploying application warprec failed`.

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
| GET | `/v1/models` | The cards of all served models. |
| GET | `/v1/models/{name}` | One model's card: how to ask it, what it knows, how it was trained. |
| GET | `/v1/models/{name}/context` | The context a context-aware model accepts, with an example. |
| GET | `/v1/models/{name}/items?q=` | Items whose name contains `q`, with their attributes. |
| POST | `/v1/models/{name}/items/lookup` | Items by id or name. |
| POST | `/v1/models/{name}/popular` | The items with the most training interactions. |
| POST | `/v1/models/{name}/recommend` | Top-k recommendations. |
| POST | `/v1/models/{name}/score` | Scores of given candidate items, for re-ranking. |
| * | `/mcp` | MCP tools, resources and prompts, when `server.mcp` is `true`. |

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
           {"item_id": 1210, "score": 7.48, "name": "Star Wars: Episode VI - Return of the Jedi (1983)"}]}
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

## What a Model Says About Itself

Every endpoint describes itself, so a client or an agent needs nothing but the server to use it.

**The model card,** `GET /v1/models/{name}`, holds:

- `how_to_ask`: what a request must and may contain, in sentences.
- `example_request`: a request the endpoint answers, to start from.
- `catalogue`: whether items have names, and the attributes they carry with their most common values.
- `training`: what the model was trained on and how it scored.
    - dataset, date, and numbers of users, items and interactions;
    - evaluation strategy and test metrics, such as `nDCG@10`.
- `context` and `example_context`: for a context-aware model.
- The configured `description` and `item_noun`, the model class, its hyperparameters and the WarpRec version.

```bash
curl -s -H "X-API-Key: change-me" localhost:8000/v1/models/sasrec | python3 -m json.tool
```

**The catalogue.** With [`item_metadata`](../configuration/serving.md), items have names and attributes, such as genres. These are returned with every item.

- Names are matched ignoring case.
- A name that is not in the catalogue gets a 422 suggesting the closest ones ("did you mean 'Toy Story (1995)'?").
- Search answers whether a model knows an item:

```bash
curl -s -H "X-API-Key: change-me" "localhost:8000/v1/models/sasrec/items?q=toy%20story"
curl -s -H "X-API-Key: change-me" "localhost:8000/v1/models/sasrec/items?q=Kung%20Fu%20Panda"   # no matches: not in its training data
curl -s -X POST localhost:8000/v1/models/sasrec/items/lookup -H "X-API-Key: change-me" \
     -H "Content-Type: application/json" -d '{"items": ["heat (1995)", 260]}'
```

**The context.** `GET /v1/models/{name}/context` answers "what context can I give you?", with three things per field:

- its accepted values, most frequent in training first, with their counts;
- the training range, for a numeric field;
- the configured description.

It also gives an example context that the model accepts.

```bash
curl -s -H "X-API-Key: change-me" localhost:8000/v1/models/fm/context | python3 -m json.tool
```

Checkpoints saved before WarpRec recorded training facts still serve. Their card has `training: null`, and their context values come without counts.

## Popular Items, Filters and Explanations

**Popular items.** `POST /v1/models/{name}/popular` lists the items with the most training interactions. It is the same for everyone, and suits a first visit:

```bash
curl -s -X POST localhost:8000/v1/models/sasrec/popular -H "X-API-Key: change-me" \
     -H "Content-Type: application/json" -d '{"k": 5, "filter": {"genres": "Comedy"}}'
```

**Filters.** `recommend` and `popular` accept `filter`, such as `{"genres": "Comedy"}`.

- Only items with those attributes are returned, ranked as the model ranks them.
- Values are compared ignoring case.
- A list matches any of its values, for example `{"genres": ["Action", "Crime"]}`.
- Several attributes must all match.
- An unknown attribute or value gets a 422 that lists or suggests the accepted ones.

**Explanations.** `explain: true` on `recommend` adds `because` to each item: up to two of the request's own items (the user's training items, or the session) that training users most often consumed together with it, each with `co_occurrences`, the number of training users who consumed both. For user 1 above, the first item becomes:

```json
{"item_id": 2858, "score": 7.91, "name": "American Beauty (1999)",
 "because": [{"item_id": 608, "name": "Fargo (1996)", "co_occurrences": 1840},
             {"item_id": 2762, "name": "Sixth Sense, The (1999)", "co_occurrences": 1787}]}
```

!!! warning
    `because` is evidence from the training data, not the model's reasoning: WarpRec models do not expose why they rank an item. Present it as "people who liked X also liked this", not as the reason the model chose it.

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

With `server.mcp: true` and the `mcp` extra installed, the same server speaks the [Model Context Protocol](https://modelcontextprotocol.io/) under `/mcp`. The API key protects it as well.

**Instructions.** On connecting, a client receives instructions it passes to its model before any tool is called. They state what the server does, list each endpoint with its configured `description`, and say which tool answers which question.

**Tools:**

| Tool | Answers |
|---|---|
| `list_models` | Which models are there, what each serves and needs? |
| `describe_model` | How do I ask this model? What does it know, and how was it trained? |
| `describe_context` | What context can I provide, and which values are accepted? |
| `search_items` | Is this model trained on "Kung Fu Panda"? Which "Toy Story" films does it know? |
| `get_items` | What are these items, and what attributes do they have? |
| `popular_items` | What is popular, possibly among comedies only? |
| `recommend` | What should this user, or this session, try next, possibly filtered and explained? |
| `score_items` | Which of these candidates would the user like most? |

**Resources:** `warprec://models` (the list) and `warprec://models/{name}` (a model card), for clients that browse resources instead of calling tools.

**Prompts:**
- `recommend_for_me` guides a conversation that ends in explained, personal recommendations.
- `explore_catalogue` guides an exploration of what a model knows.

A client configuration looks like this:

```json
{"mcpServers": {"warprec": {"url": "http://localhost:8000/mcp/", "headers": {"X-API-Key": "change-me"}}}}
```

## Scaling and GPUs

Each endpoint is its own Ray Serve deployment, configured independently:

- **batching**: concurrent requests are grouped into one forward pass of up to `max_batch_size` requests, waiting at most `batch_wait_timeout_s` for a batch to fill. A replica accepts as many requests at once as a batch holds, unless `deployment.max_ongoing_requests` says otherwise.
- **deployment.num_replicas** fixes the number of replicas, while **deployment.autoscaling_config** lets Ray Serve add and remove them with the load.
- **deployment.ray_actor_options** sets the resources of each replica. With `device: cuda` a replica asks for one GPU; `num_gpus: 0.25` places four replicas on one GPU instead.
- **server.ray_address: auto** joins the Ray cluster the machine belongs to, rather than starting a local one.

## Deploying

`warprec.serve` runs the same way everywhere Ray runs:
- one machine;
- a Docker container;
- a Kubernetes `Deployment`;
- a Ray cluster on virtual machines;
- a KubeRay `RayService`.

For a cluster, `--export` writes the application as a standard Ray Serve config file:

```bash
python -m warprec.serve -c serve.yml --export serve_app.yaml
```

[Deploying Served Models](../cloud/deployment.md) walks through every option, with the image, the manifests and the trade-offs.

## Security

- **Checkpoints are pickles.** Loading one can run code, so only serve files from a trusted source.
- **Set an API key** whenever the server is reachable beyond the local machine. The `WARPREC_API_KEY` environment variable keeps it out of configuration files.

## Limitations

- Checkpoints saved before this version serve without seen-item masking or popularity fallback. Graph-based and context-aware models among them must be saved again.
- Checkpoints are read from the local file system.
