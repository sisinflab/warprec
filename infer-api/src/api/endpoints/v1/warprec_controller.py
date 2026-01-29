from fastapi import APIRouter

from .sequential import sequential_router


# Main Warprec router that integrates all specialized controllers
router = APIRouter(prefix="/api/warprec/v1")

# Include specialized routers
router.include_router(sequential_router)
    