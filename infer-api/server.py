import sys
from pathlib import Path
import uvicorn

from fastapi import FastAPI

# Add warprec to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BaseConfig
from src.api import router

# Create FastAPI app
app = FastAPI(
    title="Warprec Inference API",
    description="API for Warprec model inference",
    version="1.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": 0},
)

# Load configuration
config = BaseConfig()

# Include the main router
app.include_router(router=router)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info",
    )