from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

router = APIRouter(tags=["predict"])

_ALLOWED_PREFIXES = ("Z:", "W:", "X:")


class PredictRequest(BaseModel):
    timestamps: list[float] = Field(..., description="Time points for the experiment")
    values: dict[str, list[float]] = Field(
        ...,
        description=(
            "Variable name -> array of numbers. Z: scalar (length 1), "
            "W: input time series, X: observed time series."
        ),
    )

    @model_validator(mode="after")
    def _check_shape(self) -> "PredictRequest":
        n = len(self.timestamps)
        for key, arr in self.values.items():
            if not key.startswith(_ALLOWED_PREFIXES):
                raise ValueError(
                    f"Variable '{key}' must start with one of {_ALLOWED_PREFIXES}"
                )
            if key.startswith("Z:") and len(arr) != 1:
                raise ValueError(
                    f"Scalar parameter '{key}' must have exactly one value, got {len(arr)}"
                )
            if key.startswith(("W:", "X:")) and len(arr) != n:
                raise ValueError(
                    f"Time-series '{key}' length {len(arr)} does not match "
                    f"timestamps length {n}"
                )
        return self


class PredictResponse(BaseModel):
    titer: float


@router.post(
    "/predict",
    response_model=PredictResponse,
    operation_id="postPredict",
    summary="Run inference",
    description=(
        "Accepts bioprocess input data and returns a predicted final titer. "
        "The payload contains timestamps and a dictionary of variables. "
        "Variable keys use a prefix convention: Z: for scalar parameters, "
        "W: for time-series inputs, and X: for time-series observations."
    ),
    responses={
        200: {"description": "Prediction completed successfully"},
        400: {"description": "Invalid request payload"},
        500: {"description": "Internal server error"},
    },
)
def post_predict(payload: PredictRequest) -> PredictResponse:
    try:
        return PredictResponse(titer=0.0)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
