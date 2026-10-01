"""An agent reaches the served models through MCP tools on the same server.

Skipped unless the 'mcp' extra is installed.
"""

import asyncio
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

from conftest import make_model, save_servable  # noqa: E402


@pytest.fixture(scope="module")
def mcp_url(tmp_path_factory: pytest.TempPathFactory, dataset: Dataset):
    root = tmp_path_factory.mktemp("mcp")
    path = save_servable(root / "sasrec.pth", make_model("SASRec", dataset), dataset)
    fm = save_servable(root / "fm.pth", make_model("FM", dataset), dataset)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = ServingConfiguration.model_validate(
        {
            "server": {"port": port, "mcp": True, "api_key": "secret"},
            "endpoints": [
                {"name": "sasrec", "checkpoint": str(path)},
                {"name": "fm", "checkpoint": str(fm)},
            ],
        }
    )
    mp = pytest.MonkeyPatch()
    mp.delenv(API_KEY_ENV, raising=False)
    # One CPU per model replica plus one for the gateway, or the gateway is never
    # scheduled and serve.run waits forever.
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
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
    assert models["fm"]["context"]["daytime"]["values"] == ["evening", "morning"]


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
