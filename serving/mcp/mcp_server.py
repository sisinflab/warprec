"""MCP server entry point for the WarpRec inference service.

Provides generalized recommendation tools via the Model Context Protocol.
Instead of one tool per model-dataset combination, a single ``recommend`` tool
accepts a ``model_key`` parameter to select the target model. A companion
``list_models`` tool allows LLM agents to discover available model keys and
their types at runtime.
"""

import sys
from pathlib import Path

# Ensure the repository root is on the Python path so that both
# the ``warprec`` package and the ``serving`` package are importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from serving.common import InferenceService, ModelManager, ServingConfig

# Load configuration and initialize shared services
config = ServingConfig.from_yaml()
model_manager = ModelManager(config)
model_manager.load_all()
inference_service = InferenceService(model_manager)

mcp = FastMCP("WarpRec MCP Server")


@mcp.tool(
    description=(
        "Recommend items using a WarpRec model. "
        "Provide the 'model_key' to select which model-dataset pair to use "
        "(e.g., 'SASRec_movielens' for sequential, 'BPR_movielens' for collaborative). "
        "Use the 'list_models' tool first to discover available model keys and their types. "
        "For sequential models, provide 'item_sequence' as a list of integer item IDs. "
        "For collaborative models, provide 'user_index' as an integer. "
        "For contextual models, provide both 'user_index' and 'context'."
    )
)
def recommend(
    model_key: str,
    top_k: int = 10,
    item_sequence: list[int] | None = None,
    user_index: int | None = None,
    context: list[int] | None = None,
) -> list:
    """Get recommendations from a specified model-dataset pair.

    Args:
        model_key: Identifier selecting the model and dataset (e.g., "SASRec_movielens").
        top_k: Number of recommendations to return.
        item_sequence: External item IDs for sequential models.
        user_index: User identifier for collaborative or contextual models.
        context: Context feature values for contextual models.

    Returns:
        Ordered list of recommended items.
    """
    return inference_service.recommend(
        model_key=model_key,
        top_k=top_k,
        item_sequence=item_sequence,
        user_index=user_index,
        context=context,
    )


@mcp.tool(
    description="List all available WarpRec model-dataset pairs and their recommendation types."
)
def list_models() -> dict[str, str]:
    """Return the available model keys and their recommender types.

    Returns:
        Dictionary mapping model keys to their types
        (e.g., ``{"SASRec_movielens": "sequential", "BPR_movielens": "collaborative"}``).
    """
    return model_manager.get_available_endpoints()


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Health check endpoint returning a plain OK response."""
    return PlainTextResponse("OK")


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host=config.server.host,
        port=config.server.mcp_port,
    )
