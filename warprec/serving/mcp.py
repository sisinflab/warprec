import json
from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from ray.serve.handle import DeploymentHandle

Item = Union[int, str]


def instructions(endpoints: List[Dict[str, Any]]) -> str:
    """What the server tells an MCP client when it connects.

    Clients pass these instructions to the model before any tool is called, so
    they say what the server is for, what each endpoint serves and which tool
    answers which question.

    Args:
        endpoints (List[Dict[str, Any]]): Each endpoint's name and, when
            configured, description and item noun.

    Returns:
        str: The instructions.
    """
    lines = [
        "This server recommends items with models trained by WarpRec. "
        "Each model is an endpoint you name in every tool call:",
    ]
    for endpoint in endpoints:
        description = (
            endpoint.get("description") or f"recommends {endpoint['item_noun']}s"
        )
        lines.append(f"- {endpoint['name']}: {description}")
    lines += [
        "",
        "How to use it:",
        "- describe_model tells what a model needs (a user_id, a history of items, "
        "a context), gives a request that works, and what it was trained on.",
        "- search_items and get_items answer whether a model knows an item, by name; "
        "an empty search means the item is not in its training data.",
        "- describe_context lists the context a context-aware model accepts, with "
        "the known values and an example.",
        "- recommend, popular_items and score_items produce recommendations; "
        "filter restricts them by item attributes, explain adds evidence from the "
        "training data (not the model's reasoning).",
        "Item names must match the catalogue: search for them first.",
    ]
    return "\n".join(lines)


def build_mcp(
    models: Dict[str, DeploymentHandle], endpoints: List[Dict[str, Any]]
) -> Any:
    """The MCP tools, resources and prompts through which an agent asks the models.

    Args:
        models (Dict[str, DeploymentHandle]): The model deployments by endpoint name.
        endpoints (List[Dict[str, Any]]): Each endpoint's name, description and
            item noun, for the instructions.

    Returns:
        Any: An ASGI application, with its lifespan, to mount under /mcp.
    """
    mcp = FastMCP("WarpRec", instructions=instructions(endpoints))

    def handle(model: str) -> DeploymentHandle:
        if model not in models:
            raise ToolError(
                f"No model named '{model}'. Available: {', '.join(models)}."
            )
        return models[model]

    def unwrap(result: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in result:
            raise ToolError(result["error"]["detail"])
        return result

    def request(**fields: Any) -> Dict[str, Any]:
        return {key: value for key, value in fields.items() if value is not None}

    async def summaries() -> List[Dict[str, Any]]:
        listed = []
        for name, model in models.items():
            card = await model.describe.remote()
            listed.append(
                {
                    "name": name,
                    "model": card["model"],
                    "kind": card["kind"],
                    "description": card["description"],
                    "item_noun": card["item_noun"],
                    "n_items": card["n_items"],
                    "catalogue": card["catalogue"]["names"],
                    "context": list(card["context"]) if card["context"] else None,
                    "needs_user": card["needs_user"],
                }
            )
        return listed

    @mcp.tool
    async def list_models() -> List[Dict[str, Any]]:
        """List the models on this server: what each serves and what it needs.

        For the full description of one model, with an example request, call
        describe_model.
        """
        return await summaries()

    @mcp.tool
    async def describe_model(model: str) -> Dict[str, Any]:
        """Describe one model: how to ask it, with an example request that
        works, the items it knows, what it was trained on and how well it
        scored, and the context it accepts."""
        return await handle(model).describe.remote()

    @mcp.tool
    async def describe_context(model: str) -> Dict[str, Any]:
        """List the context a context-aware model accepts: each field, its
        known values (most frequent first), its meaning, and an example
        context to send with recommend."""
        return unwrap(await handle(model).describe_context.remote())

    @mcp.tool
    async def search_items(model: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search the items a model knows by part of their name, ignoring case.

        An empty 'matches' means the model was not trained on such an item;
        'suggestions' then lists the closest names. Each item comes with its
        attributes and how many training interactions it had.
        """
        return unwrap(
            await handle(model).search_items.remote({"query": query, "limit": limit})
        )

    @mcp.tool
    async def get_items(model: str, items: List[Item]) -> Dict[str, Any]:
        """Look items up by id or exact name: their name, attributes and
        training interactions. Items not found are listed with the closest names."""
        return unwrap(await handle(model).get_items.remote({"items": items}))

    @mcp.tool
    async def popular_items(
        model: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Union[str, List[str]]]] = None,  # pylint: disable=redefined-builtin
        exclude: Optional[List[Item]] = None,
    ) -> Dict[str, Any]:
        """The items with the most training interactions - what is popular,
        for anyone. filter keeps only items with given attributes, such as
        {"genres": "Comedy"}."""
        return unwrap(
            await handle(model).popular.remote(
                request(k=k, filter=filter, exclude=exclude)
            )
        )

    @mcp.tool
    async def recommend(
        model: str,
        user_id: Optional[Item] = None,
        history: Optional[List[Item]] = None,
        k: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        exclude: Optional[List[Item]] = None,
        filter: Optional[Dict[str, Union[str, List[str]]]] = None,  # pylint: disable=redefined-builtin
        explain: bool = False,
    ) -> Dict[str, Any]:
        """Recommend items with one of the models from list_models.

        Send a user_id, or for a sequential model a history of items the user
        interacted with, oldest first; items may be ids or exact names (use
        search_items to find them). A context-aware model also needs a context
        (see describe_context). filter keeps only items with given attributes;
        exclude leaves items out; explain adds, for each item, the user's own
        items that training users most often consumed with it - evidence from
        the data, not the model's reasoning.
        """
        return unwrap(
            await handle(model).recommend.remote(
                request(
                    user_id=user_id,
                    history=history,
                    k=k,
                    context=context,
                    exclude=exclude,
                    filter=filter,
                    explain=explain or None,
                )
            )
        )

    @mcp.tool
    async def score_items(
        model: str,
        items: List[Item],
        user_id: Optional[Item] = None,
        history: Optional[List[Item]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score given items for a user or a session, in the order sent - to
        compare or re-rank candidates, such as "which of these would I like most?"."""
        return unwrap(
            await handle(model).score.remote(
                request(items=items, user_id=user_id, history=history, context=context)
            )
        )

    @mcp.resource("warprec://models")
    async def model_list() -> str:
        """The models on this server, as listed by list_models."""
        return json.dumps(await summaries())

    @mcp.resource("warprec://models/{name}")
    async def model_card(name: str) -> str:
        """The card of one model, as describe_model returns it."""
        return json.dumps(await handle(name).describe.remote(), default=str)

    @mcp.prompt
    def recommend_for_me(model: str = "") -> str:
        """Guide a conversation that ends in personal recommendations."""
        target = f"the '{model}' model" if model else "a model from list_models"
        return (
            f"Help me get recommendations from {target}. First call describe_model "
            "to learn what it needs. Ask me which items I liked, confirm each one "
            "with search_items (tell me if it is not in the catalogue), and ask for "
            "a context if the model needs one (describe_context lists the choices). "
            "Then call recommend, with explain set to true, and present the results "
            "with their names and why they were suggested."
        )

    @mcp.prompt
    def explore_catalogue(model: str, topic: str = "") -> str:
        """Guide an exploration of what a model knows."""
        about = f" about '{topic}'" if topic else ""
        return (
            f"Explore what the '{model}' model knows{about}. Call describe_model "
            "for its catalogue and training, search_items to find items by name, "
            "and popular_items - with a filter on the attributes it lists - to see "
            "what is most common. Summarise what you find."
        )

    return mcp.http_app(path="/")
