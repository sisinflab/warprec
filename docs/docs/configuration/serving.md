# Serving Configuration

The **Serving Configuration** describes which trained models are served and how. Unlike the other sections in this chapter, it is **its own file**, not a section of the training configuration. It is passed to the serving command:

```bash
python -m warprec.serve -c serve.yml
```

Each endpoint points at a checkpoint written by the [training pipeline](../pipelines/training.md) with `meta.save_model: true` (see [Models](models.md)). Serving does not need the experiment directory, the dataset or the training configuration: the checkpoint carries everything. See [Serving Models](../serving/index.md) for the full workflow and the API.

Unknown keys are rejected rather than ignored, so a typo fails at startup instead of silently falling back to a default. The reference file `config/serve_config.yml` lists every key with its default.

## Server

The `server` section is optional. Every key has a default.

- **host**: The address the HTTP proxy binds to. Use `0.0.0.0` to accept connections from other machines. Defaults to `127.0.0.1`.
- **port**: The port the HTTP proxy listens on, between 1 and 65535. Defaults to `8000`.
- **route_prefix**: The path every route is mounted under. Must start with `/`. Defaults to `/`.
- **api_key**: The value every request must carry in the `X-API-Key` header, except `GET /healthz`. The `WARPREC_API_KEY` environment variable overrides it. `null` disables authentication. Defaults to `null`.
- **mcp**: Whether to expose the models as [MCP](https://modelcontextprotocol.io/) tools under `/mcp`. Requires the `mcp` extra. Defaults to `false`.
- **ray_address**: The Ray cluster to run on. `null` starts a local Ray instance; `auto` joins the cluster this machine belongs to; any other value is passed to `ray.init(address=...)`. Defaults to `null`.

## Endpoints

`endpoints` is a list with at least one entry. Each entry serves one checkpoint.

- **name**: The name the model is served under, used in its URL (`/v1/models/<name>/...`). Must be unique, start with a letter or digit and contain only letters, digits, `-` and `_`. `gateway` is reserved. Required.
- **checkpoint**: The `.pth` file to serve. Relative paths resolve against the working directory. The file must exist. Required.
- **device**: Where the model runs: `cpu`, `mps`, `cuda` or `cuda:N`. Defaults to `cpu`.
- **default_k**: How many items a request gets when it does not set `k`. Must not exceed `max_k`. Defaults to `10`.
- **max_k**: The largest `k` a request may ask for. Defaults to `100`.
- **mask_seen**: Whether items the user interacted with in training are left out of their recommendations. Defaults to `true`.
- **unknown_user**: What a user that was not in the training data gets. `error` answers `404`; `popular` answers with the items most interacted with in training, and marks the response with `fallback: true`. Defaults to `error`.
- **item_metadata**: Optional. A delimited file that names the items, so that responses carry a `name` next to each `item_id` and requests may refer to items by name.
    - **path**: The file. It must exist. Required.
    - **sep**: The column separator. Multi-character separators such as `::` are supported. Defaults to `,`.
    - **header**: Whether the first row holds column names. Defaults to `true`.
    - **id_column**: The column of item ids, by position (an integer) or, with a header, by name. Defaults to `0`.
    - **name_column**: The column of item names, by position or name. Defaults to `1`.
    - **encoding**: The file encoding, for example `latin-1` for MovieLens. Defaults to `utf-8`.
- **batching**: How concurrent requests to this endpoint are grouped into one forward pass.
    - **max_batch_size**: The most requests scored together. Defaults to `64`.
    - **batch_wait_timeout_s**: How long, in seconds, the first request of a batch waits for others to join it. Defaults to `0.005`.
- **deployment**: Ray Serve options for this endpoint, passed through unchanged.
    - **num_replicas**: A fixed number of replicas, or `auto` for Ray Serve's default autoscaling. Cannot be combined with a fixed `autoscaling_config`. Defaults to `1`.
    - **max_ongoing_requests**: The most requests one replica handles at once. Defaults to Ray Serve's own default.
    - **autoscaling_config**: A Ray Serve autoscaling policy, for example `{min_replicas: 1, max_replicas: 8, target_ongoing_requests: 16}`. Defaults to `null`.
    - **ray_actor_options**: Resources per replica, for example `{num_cpus: 2, num_gpus: 0.25}`. Defaults to `null`.

!!! important
    Ray gives a replica no GPU unless it asks for one, and CUDA then sees no device at all. When `device` is `cuda` or `cuda:N` and `ray_actor_options` does not set `num_gpus`, WarpRec asks for one whole GPU per replica. Set a fraction, such as `num_gpus: 0.25`, to place several replicas on the same GPU.

## Example

```yaml
server:
  port: 8000
  api_key: change-me
endpoints:
  - name: sasrec
    checkpoint: experiments/ml-1m/serialized/SASRec_example.pth
    device: cuda
    item_metadata:
      path: data/ml-1m/movies.dat
      sep: "::"
      header: false
      encoding: latin-1
    deployment:
      ray_actor_options: {num_gpus: 0.5}
  - name: bpr
    checkpoint: experiments/ml-1m/serialized/BPR_example.pth
    unknown_user: popular
```
