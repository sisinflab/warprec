from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from warprec.utils.registry import reranker_registry


class RerankConfig(BaseModel):
    """Definition of the re-ranking configuration part of the configuration file.

    Re-ranking changes *what is recommended*, not how it is measured, so it lives
    at the top level rather than under evaluation: the list a run reports and the
    list it writes out are the same list, and both pass through here.

    Attributes:
        name (Optional[str]): The registered re-ranker, or None to rank by score
            alone. Defaults to None.
        pool (Optional[int]): How many of the top candidates are reconsidered. The
            objectives are quadratic in this number, so it is a budget rather than
            a preference. Defaults to 100.
        params (Dict[str, Any]): The parameters passed to the re-ranker.
    """

    name: Optional[str] = None
    pool: Optional[int] = 100
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]):
        """Validate the re-ranker name against the registry."""
        if v is None:
            return None
        if v.upper() not in reranker_registry.list_registered():
            raise ValueError(
                f"Re-ranker {v} not found in registry. Registered: "
                f"{reranker_registry.list_registered()}."
            )
        return v

    @field_validator("pool")
    @classmethod
    def validate_pool(cls, v: int):
        """Validate the candidate pool."""
        if v < 1:
            raise ValueError(f"A re-ranking pool must hold at least one item, got {v}.")
        return v

    def enabled(self) -> bool:
        """Whether a re-ranker was configured.

        Returns:
            bool: True when a re-ranker should be built.
        """
        return self.name is not None
