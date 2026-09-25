"""How the precision and gradient-clipping settings reach the trainer.

Both are arguments a Lightning trainer is built with, and four places in the
framework build one. The risk is not that a single site is wrong but that the
four drift apart, so what is pinned here is the shared translation from
configuration to trainer arguments, plus the guards that stop a setting being
accepted where it would do nothing.
"""

from typing import Any

import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.recommenders.trainer.runtime import (
    PRECISIONS,
    lightning_runtime,
    resolve_precision,
)
from warprec.utils.registry import params_registry


def optimization(**overrides: Any) -> Any:
    """An optimization block with the given settings.

    Args:
        **overrides (Any): The settings to apply.

    Returns:
        Any: The optimization configuration.
    """
    params = params_registry.get(
        "BPR",
        embedding_size=8,
        reg_weight=1e-5,
        batch_size=16,
        epochs=1,
        learning_rate=0.001,
        optimization=overrides,
    )
    return params.optimization


def test_the_default_changes_nothing():
    """An existing configuration has to train exactly as it did before."""
    runtime = lightning_runtime(optimization(), "gpu")

    assert runtime["precision"] == "32-true"
    assert runtime["gradient_clip_val"] is None
    assert runtime["gradient_clip_algorithm"] is None


def test_mixed_precision_is_refused_off_the_gpu():
    """On a CPU it costs the autocast overhead and buys nothing."""
    for wanted in ("16-mixed", "bf16-mixed"):
        assert resolve_precision(wanted, "cpu") == "32-true"


@pytest.mark.parametrize("accelerator", ["gpu", "cuda", "cuda:0"])
def test_the_gpu_is_recognised_however_it_is_spelled(accelerator: str):
    """Some call sites pass Lightning's name, others the device string."""
    # Whatever this resolves to, it must not be the CPU fallback.
    assert resolve_precision("16-mixed", accelerator) in ("16-mixed", "bf16-mixed")


def test_a_full_precision_request_is_left_alone_anywhere():
    """The default must never be second-guessed."""
    for accelerator in ("cpu", "gpu", "cuda"):
        assert resolve_precision("32-true", accelerator) == "32-true"
        assert resolve_precision(None, accelerator) == "32-true"


def test_clipping_is_passed_through_with_its_algorithm():
    """Both halves of the setting have to arrive together."""
    runtime = lightning_runtime(
        optimization(gradient_clip=1.5, gradient_clip_algorithm="value"), "gpu"
    )

    assert runtime["gradient_clip_val"] == pytest.approx(1.5)
    assert runtime["gradient_clip_algorithm"] == "value"


def test_clipping_defaults_to_the_norm():
    """Clipping by norm is the usual meaning of a single bound."""
    runtime = lightning_runtime(optimization(gradient_clip=2.0), "gpu")

    assert runtime["gradient_clip_algorithm"] == "norm"


@pytest.mark.parametrize("value", [0, 0.0, -1.0])
def test_a_non_positive_bound_means_no_clipping(value: float):
    """It is how a configuration says 'not at all'.

    Passing it through would ask Lightning to flatten every gradient to zero,
    which trains nothing and looks like a broken model rather than a typo.
    """
    runtime = lightning_runtime(optimization(gradient_clip=value), "gpu")

    assert runtime["gradient_clip_val"] is None
    assert runtime["gradient_clip_algorithm"] is None


def test_an_unknown_precision_is_refused_by_the_schema():
    """A typo must fail while the configuration is being read, not later."""
    with pytest.raises(Exception):
        optimization(precision="fp8")


def test_every_offered_precision_is_one_lightning_knows():
    """The names are handed straight to Lightning, so they have to be its own."""
    from typing import get_args

    from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT_STR

    known = set(get_args(_PRECISION_INPUT_STR))
    assert set(PRECISIONS) <= known, (
        f"offered precisions Lightning does not know: {set(PRECISIONS) - known}"
    )


def test_clipping_actually_bounds_the_gradient():
    """The setting has to do the thing it is named after.

    Lightning applies it inside its own optimisation step, so this checks the
    underlying operation the trainer is configured to perform rather than
    standing up a trainer: a loss large enough to blow well past the bound must
    come back inside it.
    """
    layer = torch.nn.Linear(4, 1)
    (layer(torch.ones(1, 4)) * 1e6).sum().backward()

    before = torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1e12)
    assert float(before) > 1.0, "the fixture did not produce a large gradient"

    after = torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
    assert float(after) > 1.0  # the norm reported is the one before clipping

    clipped = torch.cat([p.grad.flatten() for p in layer.parameters()])
    assert float(clipped.norm()) == pytest.approx(1.0, abs=1e-5)
