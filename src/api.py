from fastapi import FastAPI

from src.routes import health_router, predict_router

app = FastAPI(
    title="Prediction API",
    description=(
        "REST API for bioprocess model inference. Given time-series and scalar "
        "input data describing a bioprocess experiment, the model returns a "
        "predicted final titer value."
    ),
    version="1.0.0",
)

# Register routers
app.include_router(health_router)
app.include_router(predict_router)
