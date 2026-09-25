from typing import Any, Dict, Optional

import torch

from warprec.utils.logger import logger

# What Lightning understands. Only the mixed modes are of practical interest
# here: a pure half-precision run keeps the weights in half too, which the
# embedding tables these models are built from do not tolerate well.
PRECISIONS = ("32-true", "16-mixed", "bf16-mixed", "64-true")

# Mixed precision is a CUDA story. On CPU the autocast path exists but buys
# nothing, and bf16 needs an instruction set most CPUs running these
# experiments do not have.
_MIXED = ("16-mixed", "bf16-mixed")


def _is_gpu(accelerator: str) -> bool:
    """Whether the trainer will run on a GPU.

    The construction sites do not agree on a spelling: some pass Lightning's
    'gpu', others pass the device string 'cuda' straight through.

    Args:
        accelerator (str): The accelerator as the caller spells it.

    Returns:
        bool: True when the run is on a GPU.
    """
    return accelerator == "gpu" or accelerator.startswith("cuda")


def resolve_precision(precision: Optional[str], accelerator: str) -> str:
    """Decide what precision the trainer should actually run at.

    Args:
        precision (Optional[str]): What the configuration asked for.
        accelerator (str): The Lightning accelerator, 'gpu' or 'cpu'.

    Returns:
        str: The precision to hand to the trainer.
    """
    wanted = precision or "32-true"

    if wanted in _MIXED and not _is_gpu(accelerator):
        logger.attention(
            f"Mixed precision '{wanted}' was requested on the {accelerator}, "
            "where it does not speed anything up. Falling back to '32-true'."
        )
        return "32-true"

    if wanted == "bf16-mixed" and _is_gpu(accelerator):
        # Turing and older report False here, and silently running bf16 there
        # is slower than fp32 rather than faster.
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.attention(
                "This GPU does not support bfloat16. Falling back to '16-mixed'."
            )
            return "16-mixed"

    return wanted


def lightning_runtime(optimization: Any, accelerator: str) -> Dict[str, Any]:
    """The precision and gradient-clipping arguments a trainer is built with.

    Four places in the framework construct a Lightning trainer, and each one
    needs these arguments to mean the same thing. They are worked out here so
    that a fifth cannot quietly disagree with the other four.

    Args:
        optimization (Any): The model's optimization configuration.
        accelerator (str): The Lightning accelerator, 'gpu' or 'cpu'.

    Returns:
        Dict[str, Any]: The keyword arguments for the trainer.
    """
    clip = getattr(optimization, "gradient_clip", None)

    # A clip of zero is how a configuration says "not at all", rather than
    # asking for every gradient to be flattened.
    if clip is not None and clip <= 0:
        clip = None

    return {
        "precision": resolve_precision(
            getattr(optimization, "precision", None), accelerator
        ),
        "gradient_clip_val": clip,
        "gradient_clip_algorithm": (
            getattr(optimization, "gradient_clip_algorithm", None) or "norm"
            if clip is not None
            else None
        ),
    }
