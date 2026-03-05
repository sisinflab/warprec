"""API key authentication dependency for the REST API."""

from fastapi import HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(
    request: Request,
    api_key: str = Security(api_key_header),
) -> str:
    """Validate the API key from the request header.

    Reads the expected key from the application's ``ServingConfig`` stored in
    ``request.app.state.config``. If the configured key is empty, authentication
    is effectively disabled.

    Args:
        request (Request): The incoming HTTP request (used to access app state).
        api_key (str): Value of the ``X-API-Key`` header.

    Returns:
        str: The validated API key string.

    Raises:
        HTTPException: 401 if the header is missing, 403 if the key is invalid.
    """
    expected_key = request.app.state.config.server.api_key

    # If no API key is configured, skip validation
    if not expected_key:
        return api_key or ""

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key header not provided",
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )

    return api_key
