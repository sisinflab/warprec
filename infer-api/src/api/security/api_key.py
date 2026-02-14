from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from src.config import BaseConfig


# Get API key from environment variables
config = BaseConfig()

# Define API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Dependency function to validate API key
async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key header not provided",
        )

    if api_key != config.SERVER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    
    return api_key
