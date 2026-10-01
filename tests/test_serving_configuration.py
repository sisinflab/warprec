"""The serving configuration rejects what would only fail later, inside Ray.

A mistake caught here is a clear message at startup; the same mistake caught
by a replica is a crash loop in a log the user has to go looking for.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from warprec.utils.config.serving_configuration import (
    API_KEY_ENV,
    ServingConfiguration,
    load_serving_configuration,
)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pth"
    path.write_bytes(b"")
    return path


def config_with(checkpoint: Path, **endpoint: Any) -> Dict[str, Any]:
    return {"endpoints": [{"name": "bpr", "checkpoint": str(checkpoint), **endpoint}]}


def test_defaults_are_filled_in(checkpoint: Path):
    config = ServingConfiguration.model_validate(config_with(checkpoint))
    endpoint = config.endpoints[0]
    assert (config.server.host, config.server.port, config.server.mcp) == (
        "127.0.0.1",
        8000,
        False,
    )
    assert (endpoint.default_k, endpoint.max_k, endpoint.unknown_user) == (
        10,
        100,
        "error",
    )
    assert endpoint.mask_seen is True and endpoint.device == "cpu"


def test_it_loads_from_yaml(tmp_path: Path, checkpoint: Path):
    path = tmp_path / "serve.yml"
    path.write_text(
        f"server:\n  port: 9000\nendpoints:\n  - name: bpr\n    checkpoint: {checkpoint}\n"
    )
    assert load_serving_configuration(path).server.port == 9000


@pytest.mark.parametrize(
    "endpoint, message",
    [
        ({"default_k": 50, "max_k": 10}, "default_k"),
        ({"name": "gateway"}, "reserved"),
        ({"name": "has space"}, "name"),
        ({"device": "gpu"}, "device"),
        ({"unknown_user": "random"}, "unknown_user"),
        ({"typo_key": 1}, "typo_key"),
    ],
)
def test_invalid_endpoints_are_rejected(
    checkpoint: Path, endpoint: Dict[str, Any], message: str
):
    with pytest.raises(ValidationError, match=message):
        ServingConfiguration.model_validate(config_with(checkpoint, **endpoint))


def test_a_missing_checkpoint_is_rejected(tmp_path: Path):
    with pytest.raises(ValidationError, match="does not exist"):
        ServingConfiguration.model_validate(config_with(tmp_path / "missing.pth"))


def test_endpoint_names_are_unique(checkpoint: Path):
    endpoint = {"name": "bpr", "checkpoint": str(checkpoint)}
    with pytest.raises(ValidationError, match="unique"):
        ServingConfiguration.model_validate({"endpoints": [endpoint, endpoint]})


def test_at_least_one_endpoint_is_required():
    with pytest.raises(ValidationError, match="endpoints"):
        ServingConfiguration.model_validate({"endpoints": []})


def test_replicas_and_autoscaling_are_exclusive(checkpoint: Path):
    deployment = {"num_replicas": 2, "autoscaling_config": {"max_replicas": 4}}
    with pytest.raises(ValidationError, match="num_replicas"):
        ServingConfiguration.model_validate(
            config_with(checkpoint, deployment=deployment)
        )


def test_a_cuda_endpoint_asks_ray_for_a_gpu(checkpoint: Path):
    config = ServingConfiguration.model_validate(config_with(checkpoint, device="cuda"))
    assert (
        config.endpoints[0].deployment_options()["ray_actor_options"]["num_gpus"] == 1
    )

    shared = config_with(
        checkpoint,
        device="cuda:0",
        deployment={"ray_actor_options": {"num_gpus": 0.25}},
    )
    options = (
        ServingConfiguration.model_validate(shared).endpoints[0].deployment_options()
    )
    assert options["ray_actor_options"]["num_gpus"] == 0.25


def test_a_cpu_endpoint_passes_only_what_was_set(checkpoint: Path):
    config = ServingConfiguration.model_validate(config_with(checkpoint))
    assert config.endpoints[0].deployment_options() == {"num_replicas": 1}


def test_the_environment_overrides_the_api_key(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
):
    config = ServingConfiguration.model_validate(
        {**config_with(checkpoint), "server": {"api_key": "file"}}
    )
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert config.server.effective_api_key() == "file"
    monkeypatch.setenv(API_KEY_ENV, "env")
    assert config.server.effective_api_key() == "env"


def test_the_portable_form_has_absolute_paths_and_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    Path("model.pth").write_bytes(b"")
    Path("items.csv").write_text("id,name\n1,a\n")
    config = ServingConfiguration.model_validate(
        {
            "server": {"api_key": "secret"},
            "endpoints": [
                {
                    "name": "bpr",
                    "checkpoint": "model.pth",
                    "item_metadata": {"path": "items.csv"},
                }
            ],
        }
    )
    portable = config.portable()
    assert portable.server.api_key is None
    assert Path(portable.endpoints[0].checkpoint).is_absolute()
    assert Path(portable.endpoints[0].item_metadata.path).is_absolute()
    assert config.server.api_key == "secret", "the original is left untouched"
