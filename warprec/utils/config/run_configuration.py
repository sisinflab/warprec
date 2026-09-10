import re
from typing import Dict, Optional

from pydantic import BaseModel, field_validator

from warprec.utils.enums import ErroredTrialPolicy, ResumeMode

# A run name becomes a path component locally and a blob key on Azure,
# so it is restricted to characters that are safe in both.
_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WarpRecRunConfig(BaseModel):
    """Definition of the run configuration part of the configuration file.

    This section controls how a run is identified and whether it can be paused
    and resumed. It is honoured by the 'train' and 'swarm' pipelines.

    Attributes:
        name (Optional[str]): The identifier of the run. When None, the name is
            derived from the dataset name and a fingerprint of the configuration.
        resume (ResumeMode): The resume policy of the run.
            - auto: Resume when compatible saved state exists, otherwise start fresh.
            - force: Resume, and fail if no compatible saved state exists.
            - never: Always start fresh, discarding saved state for this run name.
        errored_trials (ErroredTrialPolicy): How to treat Ray Tune trials that
            errored before the pause.
            - skip: Leave them as they are.
            - resume: Continue each from its last checkpoint.
            - restart: Rerun each from scratch.
        pause_on_signal (bool): Whether WarpRec installs handlers that turn
            SIGINT and SIGTERM into a graceful pause. Defaults to True.

    Note:
        Only 'name' is optional. The other three fields always hold a value, so
        that the pipelines can pass them on without having to second-guess a
        None that carries no meaning.
    """

    name: Optional[str] = None
    resume: ResumeMode = ResumeMode.AUTO
    errored_trials: ErroredTrialPolicy = ErroredTrialPolicy.SKIP
    pause_on_signal: bool = True

    @field_validator("name")
    @classmethod
    def check_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate name.

        Args:
            v (Optional[str]): The run name to validate.

        Returns:
            Optional[str]: The validated run name.

        Raises:
            ValueError: If the run name contains unsupported characters.
        """
        if v is None:
            return None
        if not _SAFE_RUN_NAME.match(v):
            raise ValueError(
                f"Run name '{v}' is not valid. It must start with a letter or a "
                "digit and contain only letters, digits, '.', '_' and '-'."
            )
        return v

    def restore_flags(self) -> Dict[str, bool]:
        """Translates the errored trial policy into Ray Tune restore flags.

        Ray exposes 'resume_errored' and 'restart_errored' as two mutually
        exclusive booleans. This method guarantees only valid combinations.

        Returns:
            Dict[str, bool]: The keyword arguments to pass to 'Tuner.restore'.
        """
        return {
            "resume_errored": self.errored_trials == ErroredTrialPolicy.RESUME,
            "restart_errored": self.errored_trials == ErroredTrialPolicy.RESTART,
        }
