import hmac
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from ray.serve.handle import DeploymentHandle

from warprec.serving.schemas import (
    ContextDescription,
    LookupRequest,
    LookupResponse,
    ModelInfo,
    PopularRequest,
    PopularResponse,
    RecommendRequest,
    RecommendResponse,
    ScoreRequest,
    ScoreResponse,
    SearchResponse,
)


def build_api(
    models: Dict[str, DeploymentHandle],
    api_key: Optional[str] = None,
    mcp: bool = False,
    endpoints: Optional[List[Dict[str, Any]]] = None,
) -> FastAPI:
    """The HTTP interface of the serving application.

    It is built inside the gateway replica, once the handles to the model
    deployments exist, which is why it is a function rather than a module-level
    app.

    Args:
        models (Dict[str, DeploymentHandle]): The model deployments by endpoint name.
        api_key (Optional[str]): The key every request but /healthz must carry.
        mcp (bool): Whether to mount the MCP tools under /mcp.
        endpoints (Optional[List[Dict[str, Any]]]): Each endpoint's name,
            description and item noun, for the instructions MCP clients receive.

    Returns:
        FastAPI: The application.
    """
    mcp_app = None
    if mcp:
        # Imported here: fastmcp is an optional extra of its own.
        from warprec.serving.mcp import build_mcp  # pylint: disable=import-outside-toplevel

        mcp_app = build_mcp(
            models,
            endpoints or [{"name": name, "item_noun": "item"} for name in models],
        )

    app = FastAPI(
        title="WarpRec",
        lifespan=mcp_app.lifespan if mcp_app is not None else None,
    )

    if api_key:

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            # Only the health route itself is public: matched exactly, with or
            # without the route prefix the app is mounted under.
            root = request.scope.get("root_path", "").rstrip("/")
            if request.url.path not in ("/healthz", f"{root}/healthz"):
                sent = request.headers.get("x-api-key", "")
                if not hmac.compare_digest(sent.encode(), api_key.encode()):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing or invalid X-API-Key header."},
                    )
            return await call_next(request)

    def handle(name: str) -> DeploymentHandle:
        if name not in models:
            raise HTTPException(
                404, f"No model named '{name}'. Available: {', '.join(models)}."
            )
        return models[name]

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=List[ModelInfo])
    async def list_models() -> List[ModelInfo]:
        return [
            ModelInfo(name=name, **await model.describe.remote())
            for name, model in models.items()
        ]

    @app.get("/v1/models/{name}", response_model=ModelInfo)
    async def describe_model(name: str) -> ModelInfo:
        return ModelInfo(name=name, **await handle(name).describe.remote())

    @app.post(
        "/v1/models/{name}/recommend",
        response_model=RecommendResponse,
        # An item's name is left out, not sent as null, without a catalogue.
        response_model_exclude_none=True,
    )
    async def recommend(name: str, body: RecommendRequest) -> RecommendResponse:
        result = await handle(name).recommend.remote(body.model_dump(exclude_none=True))
        return RecommendResponse(model=name, **_unwrap(result))

    @app.post(
        "/v1/models/{name}/score",
        response_model=ScoreResponse,
        # An item's name is left out, not sent as null, without a catalogue.
        response_model_exclude_none=True,
    )
    async def score(name: str, body: ScoreRequest) -> ScoreResponse:
        result = await handle(name).score.remote(body.model_dump(exclude_none=True))
        return ScoreResponse(model=name, **_unwrap(result))

    @app.get("/v1/models/{name}/context", response_model=ContextDescription)
    async def describe_context(name: str) -> ContextDescription:
        result = await handle(name).describe_context.remote()
        return ContextDescription(model=name, **_unwrap(result))

    @app.get(
        "/v1/models/{name}/items",
        response_model=SearchResponse,
        response_model_exclude_none=True,
    )
    async def search_items(name: str, q: str, limit: int = 10) -> SearchResponse:
        result = await handle(name).search_items.remote({"query": q, "limit": limit})
        return SearchResponse(model=name, **_unwrap(result))

    @app.post(
        "/v1/models/{name}/items/lookup",
        response_model=LookupResponse,
        response_model_exclude_none=True,
    )
    async def lookup_items(name: str, body: LookupRequest) -> LookupResponse:
        result = await handle(name).get_items.remote(body.model_dump())
        return LookupResponse(model=name, **_unwrap(result))

    @app.post(
        "/v1/models/{name}/popular",
        response_model=PopularResponse,
        response_model_exclude_none=True,
    )
    async def popular_items(name: str, body: PopularRequest) -> PopularResponse:
        result = await handle(name).popular.remote(body.model_dump(exclude_none=True))
        return PopularResponse(model=name, **_unwrap(result))

    if mcp_app is not None:
        app.mount("/mcp", mcp_app)
    return app


def _unwrap(result: Dict[str, Any]) -> Dict[str, Any]:
    """Turn an error returned by a model deployment into an HTTP error.

    Args:
        result (Dict[str, Any]): What the deployment returned.

    Returns:
        Dict[str, Any]: The result, when it is not an error.

    Raises:
        HTTPException: With the status the deployment reported.
    """
    if "error" in result:
        raise HTTPException(result["error"]["status"], result["error"]["detail"])
    return result
