# WarpRec MCP Server

Model Context Protocol (MCP) server for WarpRec model inference. Designed
for integration with LLM agents that can discover and invoke MCP tools.

## Setup

1. Ensure model checkpoints and dataset files are in the directories specified
   in `serving/serving_config.yml`.

2. Install dependencies (using specialized environment.yml):
   ```bash
   conda install --file serving/mcp/environment.yml
   ```

3. Start the server:
   ```bash
   python serving/mcp/mcp_server.py
   ```

   The server starts on the host and port defined in the configuration
   (defaults to `0.0.0.0:8081`) using HTTP transport.

## Available Tools

### `list_models`

Returns all loaded model-dataset pairs and their recommendation types.

**Parameters:** None.

**Returns:** A dictionary mapping model keys to types:
```json
{
  "SASRec_movielens": "sequential",
  "BPR_movielens": "collaborative"
}
```

### `recommend`

Get recommendations from any loaded model-dataset pair.

**Parameters:**

| Parameter        | Type         | Required | Description                                   |
|------------------|------------- |----------|-----------------------------------------------|
| `model_key`      | `str`        | Yes      | Model-dataset identifier (e.g., `SASRec_movielens`). |
| `top_k`          | `int`        | No       | Number of recommendations (default: 10).      |
| `item_sequence`  | `list[int]`  | Sequential models | Ordered item names.                  |
| `user_index`     | `int`        | Collaborative models | External user identifier.          |
| `context`        | `list[int]`  | Contextual models | Context feature values.              |

**Returns:** Ordered list of recommended items.

## Usage Example

An LLM agent interacting with this MCP server would:

1. Call `list_models` to discover available model keys and types.
2. Based on the user's request, select the appropriate `model_key`.
3. Call `recommend` with the `model_key` and the relevant parameters.

For a sequential model like `SASRec_movielens`, the agent provides:
```json
{
  "model_key": "SASRec_movielens",
  "item_sequence": [5 28],
  "top_k": 5
}
```

For a collaborative model like `BPR_movielens`, the agent provides:
```json
{
  "model_key": "BPR_movielens",
  "user_index": 42,
  "top_k": 5
}
```

## Health Check

A health check endpoint is available at `GET /health`, returning plain text `OK`.

## Adding New Models

Add entries to the `endpoints` list in `serving/serving_config.yml` and restart
the server. New models are automatically exposed through the `recommend` tool
without any code changes. See the [main serving README](../README.md) for details.
