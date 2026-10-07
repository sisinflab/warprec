"""Behavioural tests for the runtime environment the pipelines give Ray.

Every Ray worker imports WarpRec and the user's modules from what the driver
ships it, so what is shipped cannot depend on the directory WarpRec was
launched from: an installed WarpRec is not under the working directory at all.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

import warprec
from warprec.pipelines import common


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Stand in for ray.init and keep what it was called with.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces ray.init.

    Returns:
        Dict[str, Any]: The keyword arguments of the call, once it is made.
    """
    calls: Dict[str, Any] = {}
    monkeypatch.setattr(common.ray, "init", lambda **kwargs: calls.update(kwargs))
    return calls


def configuration(custom_modules: list) -> Any:
    """The part of a training configuration initialise_ray reads.

    Args:
        custom_modules (list): The configured custom modules.

    Returns:
        Any: An object shaped like the configuration.
    """
    return SimpleNamespace(
        general=SimpleNamespace(ray_address="auto", custom_modules=custom_modules)
    )


def test_warprec_is_shipped_from_wherever_it_is_installed(
    captured: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Launched outside the repository, there is no ./warprec to ship."""
    monkeypatch.chdir(tmp_path)

    common.initialise_ray(configuration([]))

    shipped = captured["runtime_env"]["py_modules"]
    assert warprec in shipped or str(Path(warprec.__file__).parent) in shipped
    assert "warprec" not in shipped


def test_the_user_modules_are_shipped_and_left_as_configured(
    captured: Dict[str, Any],
):
    """Shipping WarpRec must not add it to the user's own module list."""
    custom = ["user_code/my_model.py"]
    config = configuration(custom)

    common.initialise_ray(config)

    assert "user_code/my_model.py" in captured["runtime_env"]["py_modules"]
    assert config.general.custom_modules == ["user_code/my_model.py"]
