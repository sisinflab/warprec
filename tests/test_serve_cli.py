"""warprec.serve starts the serving application or writes it out for a cluster."""

import importlib.util
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

from warprec.data.dataset import Dataset
from warprec.serve import main

from conftest import make_model, save_servable


def write_config(tmp_path: Path, **server) -> Path:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"")
    path = tmp_path / "serve.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "server": server,
                "endpoints": [{"name": "bpr", "checkpoint": "model.pth"}],
            }
        )
    )
    return path


def test_a_missing_config_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing.yml"):
        main(["-c", str(tmp_path / "missing.yml")])


def test_export_writes_a_ray_serve_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("ray.serve")
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path, port=9100, api_key="secret")
    main(["-c", "serve.yml", "--export", "serve_app.yaml"])

    document = yaml.safe_load((tmp_path / "serve_app.yaml").read_text())
    assert document["http_options"]["port"] == 9100
    (application,) = document["applications"]
    assert application["import_path"] == "warprec.serving.app:app_builder"
    config = application["args"]["config"]
    assert Path(config["endpoints"][0]["checkpoint"]).is_absolute()
    assert config["server"]["api_key"] is None


def test_the_exported_config_builds_an_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("ray.serve")
    from ray.serve import Application

    from warprec.serving.app import app_builder

    monkeypatch.chdir(tmp_path)
    write_config(tmp_path)
    main(["-c", "serve.yml", "--export", "serve_app.yaml"])
    (application,) = yaml.safe_load((tmp_path / "serve_app.yaml").read_text())[
        "applications"
    ]
    assert isinstance(app_builder(application["args"]), Application)


def test_the_module_runs_as_a_command():
    done = subprocess.run(
        [sys.executable, "-m", "warprec.serve", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0 and "--export" in done.stdout


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_the_server_stops_cleanly_on_a_signal(
    tmp_path: Path, dataset: Dataset, stop_signal: int
):
    """Ctrl-C at a terminal and SIGTERM from a container both end it with code 0."""
    pytest.importorskip("ray.serve")
    checkpoint = save_servable(
        tmp_path / "bpr.pth", make_model("BPR", dataset), dataset
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = tmp_path / "serve.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "server": {"port": port},
                "endpoints": [{"name": "bpr", "checkpoint": str(checkpoint)}],
            }
        )
    )
    output = tmp_path / "serve.log"
    server = subprocess.Popen(
        [sys.executable, "-m", "warprec.serve", "-c", str(config)],
        # A shell running this in the background would ignore SIGINT, as it
        # does for any background job; a terminal user's Ctrl-C does not.
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        stdout=output.open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
                break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail(
                "the server never answered /healthz:\n" + output.read_text()[-2000:]
            )

        server.send_signal(stop_signal)
        assert server.wait(timeout=60) == 0, output.read_text()[-2000:]
    finally:
        if server.poll() is None:
            server.kill()
    assert "Serving stopped" in output.read_text()


@pytest.mark.skipif(
    importlib.util.find_spec("fastmcp") is not None,
    reason="checks the message shown when fastmcp is missing",
)
def test_mcp_without_its_extra_is_refused_before_ray_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Otherwise the gateway fails inside Ray and the deploy stalls without a word."""
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path, mcp=True)
    with pytest.raises(SystemExit, match=r"warprec\[mcp\]"):
        main(["-c", "serve.yml"])


class Recorder:
    """Stands in for a Ray deployment and records what it is bound with."""

    def __init__(self):
        self.bound = []

    def options(self, **_options):
        return self

    def bind(self, *args):
        self.bound.append(args)
        return args


def test_replicas_are_given_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """On a joined cluster a replica runs where the cluster started, not here."""
    pytest.importorskip("ray.serve")
    from warprec.serving import app
    from warprec.utils.config.serving_configuration import ServingConfiguration

    monkeypatch.chdir(tmp_path)
    Path("model.pth").write_bytes(b"")
    servers, gateway = Recorder(), Recorder()
    monkeypatch.setattr(app, "ModelServer", servers)
    monkeypatch.setattr(app, "Gateway", gateway)

    config = ServingConfiguration.model_validate(
        {
            "server": {"api_key": "secret"},
            "endpoints": [{"name": "bpr", "checkpoint": "model.pth"}],
        }
    )
    app.build_application(config)

    ((endpoint,),) = servers.bound
    assert Path(endpoint["checkpoint"]).is_absolute()
    assert gateway.bound[0][1]["api_key"] == "secret"


@pytest.mark.parametrize(
    "ray_address, expected",
    [
        (None, ["serve.shutdown", "ray.shutdown"]),
        ("auto", ["serve.delete:warprec", "ray.shutdown"]),
    ],
)
def test_stopping_removes_only_its_own_application_on_a_shared_cluster(
    monkeypatch: pytest.MonkeyPatch, ray_address, expected
):
    """serve.shutdown would delete every team's applications on a joined cluster."""
    pytest.importorskip("ray.serve")
    from warprec.serving import app

    calls = []
    monkeypatch.setattr(app.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(app.ray, "shutdown", lambda: calls.append("ray.shutdown"))
    monkeypatch.setattr(app.serve, "shutdown", lambda: calls.append("serve.shutdown"))
    monkeypatch.setattr(
        app.serve, "delete", lambda name: calls.append(f"serve.delete:{name}")
    )

    app.stop_serving(ray_address)
    assert calls == expected
