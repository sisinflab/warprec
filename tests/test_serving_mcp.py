"""An agent reaches the served models through MCP tools on the same server.

Skipped unless the 'mcp' extra is installed.
"""

import asyncio
import json
import socket

import pytest

fastmcp = pytest.importorskip("fastmcp")
ray = pytest.importorskip("ray")
from fastmcp import Client  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from ray import serve  # noqa: E402

from warprec.data.dataset import Dataset  # noqa: E402
from warprec.serving.app import APP_NAME, build_application  # noqa: E402
from warprec.utils.config.serving_configuration import API_KEY_ENV, ServingConfiguration  # noqa: E402

from conftest import make_model, save_servable, write_catalogue  # noqa: E402


@pytest.fixture(scope="module")
def mcp_url(tmp_path_factory: pytest.TempPathFactory, dataset: Dataset):
    root = tmp_path_factory.mktemp("mcp")
    path = save_servable(root / "sasrec.pth", make_model("SASRec", dataset), dataset)
    fm = save_servable(root / "fm.pth", make_model("FM", dataset), dataset)
    movies = save_servable(root / "movies.pth", make_model("BPR", dataset), dataset)
    items = write_catalogue(root / "items.dat", dataset)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = ServingConfiguration.model_validate(
        {
            "server": {"port": port, "mcp": True, "api_key": "secret"},
            "endpoints": [
                {"name": "sasrec", "checkpoint": str(path)},
                {
                    "name": "fm",
                    "checkpoint": str(fm),
                    "description": "Apps, depending on where and when they are used",
                    "item_noun": "app",
                    "context_descriptions": {"daytime": "the time of day"},
                },
                {
                    "name": "movies",
                    "checkpoint": str(movies),
                    "description": "Films from the test dataset",
                    "item_noun": "movie",
                    "item_metadata": {
                        "path": str(items),
                        "sep": "::",
                        "header": False,
                        "columns": {"genres": {"column": 2, "separator": "|"}},
                    },
                },
            ],
        }
    )
    mp = pytest.MonkeyPatch()
    mp.delenv(API_KEY_ENV, raising=False)
    # One CPU per model replica; the gateway needs none.
    ray.init(num_cpus=3, include_dashboard=False, ignore_reinit_error=True)
    serve.start(http_options={"host": "127.0.0.1", "port": port})
    serve.run(build_application(config), name=APP_NAME, route_prefix="/")
    yield f"http://127.0.0.1:{port}/mcp/"
    serve.shutdown()
    ray.shutdown()
    mp.undo()


def call(url: str, tool: str, arguments: dict, key: str = "secret"):
    async def run():
        from fastmcp.client.transports import StreamableHttpTransport

        transport = StreamableHttpTransport(url, headers={"X-API-Key": key})
        async with Client(transport) as client:
            return (await client.call_tool(tool, arguments)).data

    return asyncio.run(run())


def test_the_models_are_listed_as_tools_see_them(mcp_url: str):
    models = {model["name"]: model for model in call(mcp_url, "list_models", {})}
    assert models["sasrec"]["kind"] == "sequential"
    assert models["fm"]["context"] == ["daytime", "weather"]


def test_recommend_passes_the_context(mcp_url: str, dataset: Dataset):
    users, _ = dataset.get_inverse_mappings()
    answer = call(
        mcp_url,
        "recommend",
        {
            "model": "fm",
            "user_id": users[0],
            "k": 4,
            "context": {"daytime": "morning", "weather": "sunny"},
        },
    )
    assert len(answer["items"]) == 4


def test_recommend_answers_a_session(mcp_url: str, dataset: Dataset):
    _, items = dataset.get_inverse_mappings()
    answer = call(
        mcp_url,
        "recommend",
        {"model": "sasrec", "history": [items[1], items[2]], "k": 3},
    )
    assert len(answer["items"]) == 3


def test_errors_reach_the_agent_as_tool_errors(mcp_url: str):
    with pytest.raises(ToolError, match="nope"):
        call(mcp_url, "recommend", {"model": "nope", "user_id": 1})


def test_mcp_is_behind_the_api_key(mcp_url: str):
    with pytest.raises(Exception):
        call(mcp_url, "list_models", {}, key="wrong")


def session(url: str, steps, key: str = "secret"):
    """Run a few calls on one MCP connection and return what they give back."""

    async def run():
        from fastmcp.client.transports import StreamableHttpTransport

        transport = StreamableHttpTransport(url, headers={"X-API-Key": key})
        async with Client(transport) as client:
            return await steps(client)

    return asyncio.run(run())


def test_the_server_explains_itself_on_connect(mcp_url: str):
    """Clients read these instructions before calling anything."""

    async def steps(client):
        return client.initialize_result.instructions

    instructions = session(mcp_url, steps)
    assert "movies: Films from the test dataset" in instructions
    assert "search_items" in instructions and "describe_context" in instructions


def test_every_capability_is_a_tool(mcp_url: str):
    async def steps(client):
        return {tool.name for tool in await client.list_tools()}

    assert session(mcp_url, steps) == {
        "list_models",
        "describe_model",
        "describe_context",
        "search_items",
        "get_items",
        "popular_items",
        "recommend",
        "score_items",
    }


def test_an_agent_can_tell_whether_a_title_is_known(mcp_url: str):
    """'Are you trained on Kung Fu Panda?' - no; 'Toy Story?' - yes, two of them."""
    assert (
        call(mcp_url, "search_items", {"model": "movies", "query": "Kung Fu Panda"})[
            "matches"
        ]
        == []
    )
    found = call(mcp_url, "search_items", {"model": "movies", "query": "toy story"})
    assert {m["name"] for m in found["matches"]} == {
        "Toy Story (1995)",
        "Toy Story 2 (1999)",
    }
    looked = call(mcp_url, "get_items", {"model": "movies", "items": ["heat (1995)"]})
    assert looked["items"][0]["attributes"]["genres"] == ["Action", "Crime"]


def test_an_agent_can_tell_what_context_to_give(mcp_url: str, dataset: Dataset):
    """'What context can I provide?' - the fields, their values and an example."""
    described = call(mcp_url, "describe_context", {"model": "fm"})
    assert described["fields"]["daytime"]["description"] == "the time of day"
    assert set(described["fields"]["daytime"]["values"]) == {"morning", "evening"}
    users, _ = dataset.get_inverse_mappings()
    answer = call(
        mcp_url,
        "recommend",
        {"model": "fm", "user_id": users[0], "context": described["example"]},
    )
    assert answer["items"]


def test_the_card_and_list_describe_the_models(mcp_url: str):
    listed = {model["name"]: model for model in call(mcp_url, "list_models", {})}
    assert listed["movies"]["description"] == "Films from the test dataset"
    assert listed["movies"]["catalogue"] is True and listed["fm"]["context"] == [
        "daytime",
        "weather",
    ]
    card = call(mcp_url, "describe_model", {"model": "movies"})
    assert card["how_to_ask"] and card["training"]["n_interactions"] > 0


def test_popular_filtered_and_explained_recommendations(mcp_url: str, dataset: Dataset):
    popular = call(
        mcp_url,
        "popular_items",
        {"model": "movies", "k": 3, "filter": {"genres": "Comedy"}},
    )
    assert all("Comedy" in item["attributes"]["genres"] for item in popular["items"])
    users, _ = dataset.get_inverse_mappings()
    answer = call(
        mcp_url,
        "recommend",
        {
            "model": "movies",
            "user_id": users[0],
            "k": 3,
            "filter": {"genres": "Action"},
            "explain": True,
        },
    )
    assert all("because" in item for item in answer["items"])
    scored = call(
        mcp_url,
        "score_items",
        {
            "model": "movies",
            "user_id": users[0],
            "items": ["Heat (1995)", "Toy Story (1995)"],
        },
    )
    assert [s["name"] for s in scored["scores"]] == ["Heat (1995)", "Toy Story (1995)"]


def test_model_cards_are_resources(mcp_url: str):
    async def steps(client):
        resources = [str(r.uri) for r in await client.list_resources()]
        templates = [t.uriTemplate for t in await client.list_resource_templates()]
        card = await client.read_resource("warprec://models/movies")
        return resources, templates, json.loads(card[0].text)

    resources, templates, card = session(mcp_url, steps)
    assert "warprec://models" in resources and "warprec://models/{name}" in templates
    assert card["description"] == "Films from the test dataset"


def test_prompts_guide_an_agent(mcp_url: str):
    async def steps(client):
        names = {prompt.name for prompt in await client.list_prompts()}
        guide = await client.get_prompt("recommend_for_me", {"model": "movies"})
        return names, guide.messages[0].content.text

    names, text = session(mcp_url, steps)
    assert names == {"recommend_for_me", "explore_catalogue"}
    assert "movies" in text and "search_items" in text
