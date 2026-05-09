from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


@router.get(
    "/health",
    response_model=HealthResponse,
    operation_id="getHealth",
    summary="Health check",
    description="Returns the health status of the service.",
    responses={200: {"description": "Service is healthy"}},
)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok")
