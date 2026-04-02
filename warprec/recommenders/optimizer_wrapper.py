# pylint: disable=unused-argument, too-few-public-methods
from typing import Any, Iterable
from abc import ABC, abstractmethod

from torch.optim import (
    Adam,
    AdamW,
    SGD,
    RMSprop,
    Adagrad,
)
from warprec.utils.registry import optimizer_registry


class BaseOptimizerWrapper(ABC):
    """Common interface for all optimizer wrappers."""

    @abstractmethod
    def __init__(self, params: Iterable, lr: float, **kwargs: Any):
        pass


@optimizer_registry.register("Adam")
class AdamWrapper(Adam, BaseOptimizerWrapper):
    """Wrapper for the Adam optimizer"""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        **kwargs: Any,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


@optimizer_registry.register("AdamW")
class AdamWWrapper(AdamW, BaseOptimizerWrapper):
    """Wrapper for the AdamW optimizer"""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        **kwargs: Any,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


@optimizer_registry.register("SGD")
class SGDWrapper(SGD, BaseOptimizerWrapper):
    """Wrapper for the SGD optimizer"""

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        **kwargs: Any,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )


@optimizer_registry.register("RMSprop")
class RMSpropWrapper(RMSprop, BaseOptimizerWrapper):
    """Wrapper for the RMSprop optimizer"""

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        **kwargs: Any,
    ):
        super().__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )


@optimizer_registry.register("Adagrad")
class AdagradWrapper(Adagrad, BaseOptimizerWrapper):
    """Wrapper for the Adagrad optimizer"""

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        **kwargs: Any,
    ):
        super().__init__(
            params,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )
