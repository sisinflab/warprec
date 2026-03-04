"""Main router for the WarpRec REST API.

Provides a health check endpoint and includes the versioned recommendation
routes.
"""

from fastapi import APIRouter

from serving.common.models import ApiResponse

from .v1 import recommend_router

router = APIRouter()


@router.get(
    "/health",
    status_code=200,
    response_model=ApiResponse,
    include_in_schema=False,
)
def health() -> ApiResponse:
    """Return a simple health check response."""
    return ApiResponse(message="healthy")


# Mount the v1 recommendation endpoints
router.include_router(recommend_router)
