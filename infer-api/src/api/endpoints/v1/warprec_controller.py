from fastapi import APIRouter

from .collaborative import collaborative_router
from .contextual import contextual_router
from .sequential import sequential_router


# Main Warprec router that integrates all specialized controllers
router = APIRouter(prefix="/api/warprec/v1")

# Include specialized routers
router.include_router(sequential_router)
router.include_router(contextual_router)
router.include_router(collaborative_router)