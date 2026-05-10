import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import settings
from src.models import get_best_model
from src.routes import health_router, predict_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model, metadata = get_best_model(settings.mlflow_experiment_name)
    app.state.model = model
    app.state.model_metadata = metadata
    logger.info(
        "Loaded model from experiment '%s': %s",
        settings.mlflow_experiment_name,
        metadata,
    )
    yield


app = FastAPI(
    title="Prediction API",
    description=(
        "REST API for bioprocess model inference. Given time-series and scalar "
        "input data describing a bioprocess experiment, the model returns a "
        "predicted final titer value."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(health_router)
app.include_router(predict_router)
