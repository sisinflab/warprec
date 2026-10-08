"""Where the CPU budget of a pipeline process comes from.

The design, eval and estimate pipelines run in the driver, without Ray. Asking
Ray for the granted resources there would start a local Ray instance just to
learn that nothing was granted, so Ray is asked only when it is already up.
"""

from typing import Any, Dict, Iterator

import pytest
import ray

from warprec.utils import helpers
from warprec.utils.helpers import resolve_available_cpus


class _Context:
    """Stand in for Ray's runtime context inside a task."""

    def __init__(self, assigned: Dict[str, float]):
        self.assigned = assigned

    def get_assigned_resources(self) -> Dict[str, float]:
        """The resources Ray granted the task.

        Returns:
            Dict[str, float]: The granted resources.
        """
        return self.assigned


@pytest.fixture
def no_ray() -> Iterator[None]:
    """Run with Ray down, and bring it down again if the test started it.

    Yields:
        None: Nothing; the fixture only guards the Ray state.
    """
    if ray.is_initialized():
        pytest.skip("Ray is already running in this process")
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_without_ray_no_ray_instance_is_started(no_ray: Any):
    assert resolve_available_cpus(3) == 3
    assert not ray.is_initialized()


def test_without_ray_and_budget_the_node_cores_are_used(no_ray: Any, monkeypatch):
    monkeypatch.setattr(helpers.os, "cpu_count", lambda: 7)
    assert resolve_available_cpus() == 7
    assert not ray.is_initialized()


def test_inside_a_ray_task_the_granted_cpus_win(monkeypatch):
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _Context({"CPU": 4.0}))
    assert resolve_available_cpus(2) == 4


def test_a_driver_with_ray_up_but_nothing_granted_uses_the_budget(monkeypatch):
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _Context({}))
    assert resolve_available_cpus(2) == 2
