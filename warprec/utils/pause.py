import os
import signal
import threading
from types import FrameType
from typing import Any, Optional

from warprec.utils.logger import logger

# Ray Tune reads this variable when 'Tuner.fit' installs its signal handlers.
_TUNE_DISABLE_SIGINT = "TUNE_DISABLE_SIGINT_HANDLER"

# The controller currently installed, if any. Ray Tune returns from an
# interrupted sweep normally rather than raising, and the trial states it
# reports afterwards are indistinguishable from a completed sweep, so the
# trainer needs a reliable way to ask whether a pause was requested.
_ACTIVE_CONTROLLER: Optional["PauseController"] = None


def is_pause_requested() -> bool:
    """Returns whether a pause was requested on the active controller.

    Returns:
        bool: True when a pause controller is installed and a pause has been
            requested. False when no controller is installed.
    """
    if _ACTIVE_CONTROLLER is None:
        return False
    return _ACTIVE_CONTROLLER.pause_requested


class RunPaused(Exception):
    """Raised to unwind a pipeline when a pause has been requested."""


class PauseController:
    """Turns SIGINT and SIGTERM into a graceful pause request.

    Ray Tune normally installs its own SIGINT handler for the duration of
    'Tuner.fit' and restores the previous one after the first signal, which
    would leave WarpRec unable to tell that a pause was ever requested. Instead,
    this controller keeps ownership of SIGINT by setting
    'TUNE_DISABLE_SIGINT_HANDLER', and asks Ray Tune to shut down gracefully by
    raising SIGUSR1, which Ray Tune always handles while a sweep is running.
    A sweep therefore still checkpoints its experiment state one last time,
    while WarpRec keeps a reliable record that the run was paused.

    SIGTERM, which Ray Tune does not handle at all, is re-raised as SIGINT and
    follows the same path.

    Args:
        enabled (bool): Whether to install the handlers. When False the
            controller is inert and signals behave as they normally would.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._pause_requested = threading.Event()
        self._original_sigint: Any = None
        self._original_sigterm: Any = None
        self._original_sigusr1: Any = None
        self._original_tune_env: Optional[str] = None

    @property
    def pause_requested(self) -> bool:
        """Returns whether a pause has been requested.

        Returns:
            bool: True when a pause was requested.
        """
        return self._pause_requested.is_set()

    def request_pause(self) -> None:
        """Records a pause request without going through a signal."""
        self._pause_requested.set()

    def check(self) -> None:
        """Raises when a pause has been requested.

        Raises:
            RunPaused: If a pause has been requested.
        """
        if self.pause_requested:
            raise RunPaused()

    def _handle_sigint(self, signum: int, frame: Optional[FrameType]) -> None:
        """Handles SIGINT by requesting a pause, or aborting on a second signal.

        Args:
            signum (int): The received signal number.
            frame (Optional[FrameType]): The interrupted stack frame.

        Raises:
            KeyboardInterrupt: If a pause had already been requested.
        """
        if self.pause_requested:
            logger.negative("Second interrupt received. Aborting without saving.")
            raise KeyboardInterrupt()

        self._pause_requested.set()
        logger.attention(
            "Pause requested. WarpRec will stop at the next safe point and save "
            "the run state. Interrupt again to abort immediately."
        )

        # Ask Ray Tune to wind down gracefully. While a sweep is running this
        # reaches Ray Tune's own handler, which checkpoints the experiment state
        # one last time; otherwise it reaches the placeholder installed below.
        if hasattr(signal, "SIGUSR1"):
            signal.raise_signal(signal.SIGUSR1)

    def _handle_sigusr1(self, signum: int, frame: Optional[FrameType]) -> None:
        """Absorbs SIGUSR1 when no Ray Tune sweep is running.

        The default action for SIGUSR1 is to terminate the process, so a
        placeholder handler must stay installed for the signal to be safe to
        raise at any point of the run.

        Args:
            signum (int): The received signal number.
            frame (Optional[FrameType]): The interrupted stack frame.
        """
        self._pause_requested.set()

    def _handle_sigterm(self, signum: int, frame: Optional[FrameType]) -> None:
        """Handles SIGTERM by re-raising it as SIGINT.

        Args:
            signum (int): The received signal number.
            frame (Optional[FrameType]): The interrupted stack frame.
        """
        signal.raise_signal(signal.SIGINT)

    def __enter__(self) -> "PauseController":
        """Installs the signal handlers and claims SIGINT from Ray Tune.

        Returns:
            PauseController: This controller.
        """
        global _ACTIVE_CONTROLLER  # pylint: disable=global-statement

        if not self._enabled:
            return self

        if threading.current_thread() is not threading.main_thread():
            logger.attention(
                "The pause controller is not running on the main thread. Signal "
                "handling is disabled for this run."
            )
            self._enabled = False
            return self

        self._original_tune_env = os.environ.get(_TUNE_DISABLE_SIGINT)
        os.environ[_TUNE_DISABLE_SIGINT] = "1"

        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        if hasattr(signal, "SIGUSR1"):
            self._original_sigusr1 = signal.getsignal(signal.SIGUSR1)
            signal.signal(signal.SIGUSR1, self._handle_sigusr1)

        _ACTIVE_CONTROLLER = self
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Restores the previous signal handlers and environment.

        Nothing is returned, so exceptions are never suppressed: a pause must
        not hide a failure raised inside the pipeline.

        Args:
            exc_type (Any): The exception type, if any.
            exc (Any): The exception instance, if any.
            tb (Any): The traceback, if any.
        """
        global _ACTIVE_CONTROLLER  # pylint: disable=global-statement

        if not self._enabled:
            return

        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        if hasattr(signal, "SIGUSR1") and self._original_sigusr1 is not None:
            signal.signal(signal.SIGUSR1, self._original_sigusr1)

        if self._original_tune_env is None:
            os.environ.pop(_TUNE_DISABLE_SIGINT, None)
        else:
            os.environ[_TUNE_DISABLE_SIGINT] = self._original_tune_env

        _ACTIVE_CONTROLLER = None
