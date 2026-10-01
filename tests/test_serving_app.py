"""The Ray Serve application answers over HTTP as the served models do in-process.

One application is started for the whole module on a free port, with two
models behind it. Every answer is checked against ServableModel called
directly, so the HTTP, batching and Ray layers can only pass by adding nothing.
"""

import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
ray = pytest.importorskip("ray")
from ray import serve  # noqa: E402

from warprec.data.dataset import Dataset  # noqa: E402
from warprec.serving.app import APP_NAME, build_application  # noqa: E402
from warprec.serving.servable import ServableModel, ServingPolicy  # noqa: E402
from warprec.utils.config.serving_configuration import API_KEY_ENV, ServingConfiguration  # noqa: E402

from conftest import make_model, save_servable  # noqa: E402

KEY = {"X-API-Key": "secret"}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory, dataset: Dataset):
    root = tmp_path_factory.mktemp("serving")
    return {
        "bpr": save_servable(root / "bpr.pth", make_model("BPR", dataset), dataset),
        "sasrec": save_servable(
            root / "sasrec.pth", make_model("SASRec", dataset), dataset
        ),
        "fm": save_servable(root / "fm.pth", make_model("FM", dataset), dataset),
    }


@pytest.fixture(scope="module")
def url(checkpoints):
    port = free_port()
    config = ServingConfiguration.model_validate(
        {
            # Mounted under a prefix, so that every route here is also checked
            # away from the root.
            "server": {"port": port, "api_key": "secret", "route_prefix": "/api"},
            "endpoints": [
                {
                    "name": "bpr",
                    "checkpoint": str(checkpoints["bpr"]),
                    "unknown_user": "popular",
                },
                {
                    "name": "sasrec",
                    "checkpoint": str(checkpoints["sasrec"]),
                    "batching": {"max_batch_size": 8, "batch_wait_timeout_s": 0.05},
                },
                {"name": "fm", "checkpoint": str(checkpoints["fm"])},
            ],
        }
    )
    mp = pytest.MonkeyPatch()
    mp.delenv(API_KEY_ENV, raising=False)
    # Exactly one CPU per model replica: the gateway only forwards requests
    # and must not need one of its own, or a small machine never starts it.
    ray.init(num_cpus=3, include_dashboard=False, ignore_reinit_error=True)
    serve.start(http_options={"host": "127.0.0.1", "port": port})
    serve.run(build_application(config), name=APP_NAME, route_prefix="/api")
    yield f"http://127.0.0.1:{port}/api"
    serve.shutdown()
    ray.shutdown()
    mp.undo()


def in_process(path: Path, **policy) -> ServableModel:
    return ServableModel.from_checkpoint(path, policy=ServingPolicy(**policy))


def test_health_needs_no_key(url: str):
    assert requests.get(f"{url}/healthz", timeout=10).json() == {"status": "ok"}


def test_only_the_health_route_itself_skips_the_key(url: str):
    """A path that merely ends in /healthz is not the health check."""
    response = requests.get(f"{url}/v1/models/healthz", timeout=10)
    assert response.status_code == 401
    assert "bpr" not in response.text, "no endpoint name leaks without the key"


def test_the_gateway_reserves_no_cpu():
    from warprec.serving.deployments import Gateway

    assert Gateway.ray_actor_options["num_cpus"] == 0


def test_everything_else_needs_the_key(url: str):
    assert requests.get(f"{url}/v1/models", timeout=10).status_code == 401
    wrong = requests.get(f"{url}/v1/models", headers={"X-API-Key": "nope"}, timeout=10)
    assert wrong.status_code == 401


def test_models_are_listed(url: str):
    models = requests.get(f"{url}/v1/models", headers=KEY, timeout=10).json()
    assert {(m["name"], m["kind"]) for m in models} == {
        ("bpr", "general"),
        ("sasrec", "sequential"),
        ("fm", "general"),
    }
    (fm,) = [m for m in models if m["name"] == "fm"]
    assert set(fm["context"]) == {"daytime", "weather"}
    detail = requests.get(f"{url}/v1/models/sasrec", headers=KEY, timeout=10).json()
    assert detail["model"] == "SASRec"


def test_recommend_matches_the_model_in_process(
    url: str, checkpoints, dataset: Dataset
):
    users, _ = dataset.get_inverse_mappings()
    local = in_process(checkpoints["bpr"], unknown_user="popular")
    expected = local.recommend([local.resolve(user_id=users[4], k=5)])[0]
    body = requests.post(
        f"{url}/v1/models/bpr/recommend",
        json={"user_id": users[4], "k": 5},
        headers=KEY,
        timeout=10,
    ).json()
    assert body["model"] == "bpr" and body["fallback"] is False
    # Without an item catalogue there is no name to give, so the field is absent.
    assert all("name" not in entry for entry in body["items"])
    assert [e["item_id"] for e in body["items"]] == [e["item_id"] for e in expected]


def test_an_unknown_user_falls_back_where_configured(url: str):
    fallback = requests.post(
        f"{url}/v1/models/bpr/recommend",
        json={"user_id": "nobody"},
        headers=KEY,
        timeout=10,
    )
    assert fallback.status_code == 200 and fallback.json()["fallback"] is True
    missing = requests.post(
        f"{url}/v1/models/sasrec/recommend",
        json={"user_id": "nobody"},
        headers=KEY,
        timeout=10,
    )
    assert missing.status_code == 404


@pytest.mark.parametrize(
    "path, body, status",
    [
        ("/v1/models/nope/recommend", {"user_id": 1}, 404),
        ("/v1/models/bpr/recommend", {"user_id": 1, "k": 1000}, 422),
        ("/v1/models/bpr/recommend", {"user_id": 1, "unexpected": True}, 422),
        ("/v1/models/bpr/recommend", {"history": [1, 2]}, 422),
        ("/v1/models/bpr/score", {"user_id": 1, "items": []}, 422),
    ],
)
def test_bad_requests_get_the_right_status(
    url: str, path: str, body: dict, status: int
):
    assert (
        requests.post(f"{url}{path}", json=body, headers=KEY, timeout=10).status_code
        == status
    )


def test_score_keeps_the_request_order(url: str, dataset: Dataset):
    users, items = dataset.get_inverse_mappings()
    wanted = [items[3], items[1]]
    body = requests.post(
        f"{url}/v1/models/bpr/score",
        json={"user_id": users[0], "items": wanted},
        headers=KEY,
        timeout=10,
    ).json()
    assert [e["item_id"] for e in body["scores"]] == wanted


def test_a_context_aware_model_takes_the_context(
    url: str, checkpoints, dataset: Dataset
):
    users, _ = dataset.get_inverse_mappings()
    context = {"daytime": "evening", "weather": "rainy"}
    local = in_process(checkpoints["fm"])
    expected = local.recommend([local.resolve(user_id=users[3], k=5, context=context)])[
        0
    ]
    body = requests.post(
        f"{url}/v1/models/fm/recommend",
        json={"user_id": users[3], "k": 5, "context": context},
        headers=KEY,
        timeout=10,
    ).json()
    assert [e["item_id"] for e in body["items"]] == [e["item_id"] for e in expected]

    missing = requests.post(
        f"{url}/v1/models/fm/recommend",
        json={"user_id": users[3]},
        headers=KEY,
        timeout=10,
    )
    assert missing.status_code == 422 and "context" in missing.json()["detail"]
    unknown = requests.post(
        f"{url}/v1/models/fm/recommend",
        json={
            "user_id": users[3],
            "context": {"daytime": "brunch", "weather": "rainy"},
        },
        headers=KEY,
        timeout=10,
    )
    assert unknown.status_code == 422 and "Known values" in unknown.json()["detail"]


def test_concurrent_requests_each_get_their_own_answer(
    url: str, checkpoints, dataset: Dataset
):
    users, _ = dataset.get_inverse_mappings()
    local = in_process(checkpoints["sasrec"])
    requests_by_user = [(users[u], 1 + u % 5) for u in range(24)]
    expected = [
        [e["item_id"] for e in local.recommend([local.resolve(user_id=user, k=k)])[0]]
        for user, k in requests_by_user
    ]

    def ask(pair):
        user, k = pair
        body = requests.post(
            f"{url}/v1/models/sasrec/recommend",
            json={"user_id": user, "k": k},
            headers=KEY,
            timeout=30,
        ).json()
        return [e["item_id"] for e in body["items"]]

    with ThreadPoolExecutor(max_workers=24) as pool:
        answers = list(pool.map(ask, requests_by_user))
    assert answers == expected


# Driving the deployment class outside Ray Serve leaves a live batch queue on
# the class itself, which then cannot be shipped to a replica. The probe runs in
# a process of its own so that nothing leaks into the deployments other tests
# start.
SCORE_BATCH_PROBE = """
import asyncio, json, sys
from types import SimpleNamespace
from ray import serve
from warprec.serving.deployments import ModelServer

checkpoint, users, items = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
context = SimpleNamespace(_deployment_config=SimpleNamespace(max_ongoing_requests=8))
serve.get_replica_context = lambda: context  # read by serve.batch only to warn
server = ModelServer.func_or_class(
    {"name": "bpr", "checkpoint": checkpoint,
     "batching": {"max_batch_size": 8, "batch_wait_timeout_s": 0.2}}
)
sizes = []
score_batch = server._model.score_batch
def spy(queries):
    sizes.append(len(queries))
    return score_batch(queries)
server._model.score_batch = spy

async def burst():
    return await asyncio.gather(*[
        server.score({"user_id": users[u], "items": [items[0], items[u]]}) for u in range(6)
    ])

answers = asyncio.run(burst())
print(json.dumps({"sizes": sizes,
                  "ids": [[e["item_id"] for e in a["scores"]] for a in answers]}))
"""


def test_concurrent_score_requests_share_a_forward_pass(checkpoints, dataset: Dataset):
    """Scoring is batched like recommending, so a burst costs one forward pass."""
    import json
    import subprocess
    import sys

    users, items = dataset.get_inverse_mappings()
    user_labels = [int(users[u]) for u in range(6)]
    item_labels = [int(items[i]) for i in range(6)]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            SCORE_BATCH_PROBE,
            str(checkpoints["bpr"]),
            json.dumps(user_labels),
            json.dumps(item_labels),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["sizes"] == [6]
    assert result["ids"] == [[item_labels[0], item_labels[u]] for u in range(6)]
