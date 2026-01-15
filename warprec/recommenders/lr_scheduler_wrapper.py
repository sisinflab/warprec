# pylint: disable=unused-argument, too-few-public-methods
from typing import Any
from abc import ABC, abstractmethod

from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    PolynomialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)
from warprec.utils.registry import lr_scheduler_registry


class BaseLearningRateSchedulerWrapper(ABC):
    """Common interface for all learning rate scheduler wrappers."""

    @abstractmethod
    def __init__(self, optimizer, **kwargs: Any):
        pass


@lr_scheduler_registry.register("StepLR")
class StepLRWrapper(StepLR, BaseLearningRateSchedulerWrapper):
    """Wrapper for the StepLR scheduler."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, **kwargs: Any):
        super().__init__(
            optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch
        )


@lr_scheduler_registry.register("MultiStepLR")
class MultiStepLRWrapper(MultiStepLR, BaseLearningRateSchedulerWrapper):
    """Wrapper for the MultiStepLR scheduler."""

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, **kwargs: Any):
        super().__init__(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=last_epoch,
        )


@lr_scheduler_registry.register("ConstantLR")
class ConstantLRWrapper(ConstantLR, BaseLearningRateSchedulerWrapper):
    """Wrapper for the ConstantLR scheduler."""

    def __init__(
        self, optimizer, factor=1 / 3, total_iters=5, last_epoch=-1, **kwargs: Any
    ):
        super().__init__(
            optimizer=optimizer,
            factor=factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
        )


@lr_scheduler_registry.register("LinearLR")
class LinearLRWrapper(LinearLR, BaseLearningRateSchedulerWrapper):
    """Wrapper for the LinearLR scheduler."""

    def __init__(
        self,
        optimizer,
        start_factor=1 / 3,
        end_factor=1,
        total_iters=5,
        last_epoch=-1,
        **kwargs: Any,
    ):
        super().__init__(
            optimizer=optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
        )


@lr_scheduler_registry.register("ExponentialLR")
class ExponentialLRWrapper(ExponentialLR, BaseLearningRateSchedulerWrapper):
    """Wrapper for the ExponentialLR scheduler."""

    def __init__(self, optimizer, gamma, last_epoch=-1, **kwargs: Any):
        super().__init__(optimizer=optimizer, gamma=gamma, last_epoch=last_epoch)


@lr_scheduler_registry.register("PolynomialLR")
class PolynomialLRWrapper(PolynomialLR, BaseLearningRateSchedulerWrapper):
    """Wrapper for the PolynomialLR scheduler."""

    def __init__(self, optimizer, total_iters=5, power=1, last_epoch=-1, **kwargs: Any):
        super().__init__(
            optimizer=optimizer,
            total_iters=total_iters,
            power=power,
            last_epoch=last_epoch,
        )


@lr_scheduler_registry.register("CosineAnnealingLR")
class CosineAnnealingLRWrapper(CosineAnnealingLR, BaseLearningRateSchedulerWrapper):
    """Wrapper for the CosineAnnealingLR scheduler."""

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, **kwargs: Any):
        super().__init__(
            optimizer=optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
        )


@lr_scheduler_registry.register("ReduceLROnPlateau")
class ReduceLROnPlateauWrapper(ReduceLROnPlateau, BaseLearningRateSchedulerWrapper):
    """Wrapper for the ReduceLROnPlateau scheduler."""

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        **kwargs: Any,
    ):
        super().__init__(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )


@lr_scheduler_registry.register("CosineAnnealingWarmRestarts")
class CosineAnnealingWarmRestartsWrapper(
    CosineAnnealingWarmRestarts, BaseLearningRateSchedulerWrapper
):
    """Wrapper for the CosineAnnealingWarmRestarts scheduler."""

    def __init__(
        self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, **kwargs: Any
    ):
        super().__init__(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
