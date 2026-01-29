from fastapi import APIRouter

from .model.base_model import ApiResponse

from .endpoints.v1 import warprec_router


# Create a main router
router = APIRouter()

# Health check endpoint
@router.get(
    "/health", 
    status_code=200, 
    response_model=ApiResponse, 
    include_in_schema=False
)
def health():
    return ApiResponse(
        message="healthy"
    )

# Include individual routers
router.include_router(router=warprec_router)