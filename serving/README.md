# WarpRec Serving

Inference serving layer for WarpRec recommendation models. Provides two independent
applications -- a REST API and an MCP server -- that share a common configuration,
model management, and inference engine.

## Directory Structure

```
serving/
├── serving_config.yml      # Central configuration file
├── common/                 # Shared code (config, model loading, inference)
├── restAPI/                # FastAPI REST application
└── mcp/                    # MCP server application
```

## Configuration

All settings are defined in `serving_config.yml`. The main sections are:

| Section     | Purpose                                                     |
|-------------|-------------------------------------------------------------|
| `server`    | Host, ports, and API key for the REST and MCP servers.      |
| `paths`     | Directories for model checkpoints and dataset files.        |
| `datasets`  | List of dataset and their configurations.                   |
| `endpoints` | List of model-dataset pairs to load at startup.             |

### Adding a new endpoint

1. Place the checkpoint file at `{checkpoints_dir}/{model}_{dataset}.pth`.
2. Place the dataset file at `{datasets_dir}/{item_mapping}`.
3. Add an entry to the `endpoints` list in `serving_config.yml`:

```yaml
endpoints:
  - model: "YourModel"
    dataset: "your_dataset"
    type: "sequential"  # or "collaborative" / "contextual"
```

4. Restart the server. The new model is automatically available via its key
   `YourModel_your_dataset`.

### Environment variable overrides

| Variable             | Overrides              |
|----------------------|------------------------|
| `SERVER_HOST`        | `server.host`          |
| `SERVER_API_KEY`     | `server.api_key`       |
| `SERVING_CONFIG_PATH`| Path to the YAML file  |

## Dependencies

All dependencies are declared in the respective directories as separate `environment.yml`.

## Quick Start

```bash
# Start the REST API
python serving/restAPI/server.py

# Start the MCP server
python serving/mcp/mcp_server.py
```

See the individual application READMEs for detailed usage instructions:
- [REST API](restAPI/README.md)
- [MCP Server](mcp/README.md)
