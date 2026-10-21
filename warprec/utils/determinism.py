import os

import torch

# cuBLAS is only deterministic with a fixed workspace, and reads this variable
# when its first handle is created, so it must be set before any GPU matmul
_CUBLAS_WORKSPACE_CONFIG = "CUBLAS_WORKSPACE_CONFIG"
_CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"


def apply_determinism(enabled: bool) -> None:
    """Makes the GPU kernels of this process deterministic, when asked to.

    Seeding alone does not make a GPU run reproducible: scatter and index
    kernels accumulate with atomics, in an order that changes from run to run.
    Once enabled, PyTorch uses the deterministic implementation of every
    operation that has one, and warns, naming it, about every operation that
    has none, which then runs as before. cuDNN is already kept deterministic
    by 'Recommender.set_seed'.

    It has to run in every process that trains or evaluates, before the first
    GPU operation of that process. When disabled, nothing is changed, so a
    process keeps whatever the user or an earlier call set.

    Args:
        enabled (bool): Whether deterministic algorithms are required, as
            set by 'general.gpu_deterministic'.
    """
    if not enabled:
        return
    # A workspace configuration the user exported is kept: ':16:8' is
    # deterministic too, with less memory and a lower speed
    os.environ.setdefault(_CUBLAS_WORKSPACE_CONFIG, _CUBLAS_DETERMINISTIC_WORKSPACE)
    torch.use_deterministic_algorithms(True, warn_only=True)
