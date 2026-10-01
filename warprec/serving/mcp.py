from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from ray.serve.handle import DeploymentHandle


def build_mcp(models: Dict[str, DeploymentHandle]) -> Any:
    """The MCP tools through which an agent asks the served models.

    Args:
        models (Dict[str, DeploymentHandle]): The model deployments by endpoint name.

    Returns:
        Any: An ASGI application, with its lifespan, to mount under /mcp.
    """
    mcp = FastMCP("WarpRec")

    @mcp.tool
    async def list_models() -> List[Dict[str, Any]]:
        """List the recommendation models on this server and what each accepts.

        A 'sequential' model can answer from a history of items; a 'general'
        one needs a user_id. When needs_user is true, a history also needs a
        user_id.
        """
        return [
            {"name": name, **await model.describe.remote()}
            for name, model in models.items()
        ]

    @mcp.tool
    async def recommend(
        model: str,
        user_id: Optional[Union[int, str]] = None,
        history: Optional[List[Union[int, str]]] = None,
        k: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Recommend items with one of the models from list_models.

        Send a user_id, or for a sequential model a history of items the user
        interacted with, oldest first. Items may be given by id or, when the
        model has an item catalogue, by exact name. A context-aware model also
        needs a context: the value of each field list_models shows for it,
        chosen among the known values listed there.
        """
        if model not in models:
            raise ToolError(
                f"No model named '{model}'. Available: {', '.join(models)}."
            )
        request = {
            key: value
            for key, value in {
                "user_id": user_id,
                "history": history,
                "k": k,
                "context": context,
            }.items()
            if value is not None
        }
        result = await models[model].recommend.remote(request)
        if "error" in result:
            raise ToolError(result["error"]["detail"])
        return result

    return mcp.http_app(path="/")
