# pylint: disable = C0413
"""FastAPI entry point for the WarpRec REST inference server.

Loads the serving configuration, initializes the model manager and inference
service, and starts the FastAPI application with Uvicorn.
"""

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# Ensure the repository root is on the Python path so that both
# the ``warprec`` package and the ``serving`` package are importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from serving.common import InferenceService, ModelManager, ServingConfig
from serving.restAPI.routes import router

# Load configuration and initialize shared services
config = ServingConfig.from_yaml()
model_manager = ModelManager(config)
model_manager.load_all()
inference_service = InferenceService(model_manager)

# Create FastAPI application
app = FastAPI(
    title="WarpRec Serving API",
    description="REST API for WarpRec model inference",
    version="1.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": 0},
)

# Store shared services in app state so route handlers can access them
app.state.config = config
app.state.model_manager = model_manager
app.state.inference_service = inference_service

# Include the main router
app.include_router(router=router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.rest_port,
        log_level="info",
    )
