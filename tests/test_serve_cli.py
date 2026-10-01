"""warprec.serve starts the serving application or writes it out for a cluster."""

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
