# WarpRec REST API

FastAPI-based REST server for WarpRec model inference.

## Setup

1. Ensure model checkpoints and dataset files are in the directories specified
   in `serving/serving_config.yml`.

2. Install dependencies (using specialized environment.yml):
   ```bash
   conda install --file serving/restAPI/environment.yml
   ```

3. Start the server:
   ```bash
   python serving/restAPI/server.py
   ```

   The server starts on the host and port defined in the configuration
   (defaults to `0.0.0.0:8080`).

## Endpoints

### Health Check

```
GET /health
```

Returns `{"message": "healthy", "timestamp": "..."}`.

### List Available Models

```
GET /api/warprec/v1/models
```

Returns a JSON object mapping model keys to their types:

```json
{
  "SASRec_movielens": "sequential",
  "BPR_movielens": "collaborative"
}
```

### Get Recommendations

```
POST /api/warprec/v1/recommend/{model_key}
```

**Path parameter:**
- `model_key` -- the model-dataset identifier (e.g., `SASRec_movielens`).

**Request body fields (provide those relevant to the model type):**

| Field            | Type         | Required for        | Description                        |
|------------------|------------- |---------------------|------------------------------------|
| `top_k`          | `int`        | All (default: 10)   | Number of recommendations.         |
| `item_sequence`  | `list[int]`  | Sequential          | Ordered item names.                |
| `user_index`     | `int`        | Collaborative       | External user identifier.          |
| `context`        | `list[int]`  | Contextual          | Context feature values.            |

**Response:**

```json
{
  "model_key": "SASRec_movielens",
  "model_type": "sequential",
  "recommendations": ["Toy Story", "Jurassic Park", "Star Wars"]
}
```

## Examples

### Sequential recommendation

```bash
curl -X POST http://localhost:8080/api/warprec/v1/recommend/SASRec_movielens \
  -H "Content-Type: application/json" \
  -d '{
    "item_sequence": [5 28],
    "top_k": 5
  }'
```

### Collaborative recommendation

```bash
curl -X POST http://localhost:8080/api/warprec/v1/recommend/BPR_movielens \
  -H "Content-Type: application/json" \
  -d '{
    "user_index": 42,
    "top_k": 5
  }'
```

## Authentication

Set `server.api_key` in `serving_config.yml` or the `SERVER_API_KEY` environment
variable. When configured, all requests must include the `X-API-Key` header.
If the key is left empty, authentication is disabled.

## API Documentation

Interactive Swagger UI is available at `http://localhost:8080/docs` when the
server is running.
