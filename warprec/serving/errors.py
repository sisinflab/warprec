from typing import Any, Dict


class ServingError(Exception):
    """A request the model cannot answer, with the HTTP status that says why.

    It crosses the Ray boundary as a plain dictionary rather than as a raised
    exception, so that the gateway maps it to a status code without depending
    on how Ray wraps exceptions raised inside a replica.

    Args:
        status (int): The HTTP status of the failure.
        detail (str): The message for the caller.
    """

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        """The error as the gateway expects it.

        Returns:
            Dict[str, Any]: The status and the detail.
        """
        return {"status": self.status, "detail": self.detail}
