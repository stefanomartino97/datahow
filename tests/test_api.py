from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data import build_feature_matrix


class StubModel:
    """Mimics the sklearn pipeline interface used by the predict route."""

    def __init__(self, feature_names: list[str], log_titer: float = 7.0):
        self.feature_names_in_ = np.array(feature_names)
        self._log_titer = log_titer

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._log_titer)


@pytest.fixture(scope="module")
def exp_df() -> pd.DataFrame:
    """Synthetic single-experiment dataframe shaped like the training data.

    Same column-prefix convention as the real CSV (Z: scalars filled at day 0,
    W:/X: time-series filled at every timestep), so build_feature_matrix and
    the /predict request schema both treat it the same way.
    """
    n = 5
    times = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Exp": ["Exp 1"] * n,
            "Time[day]": times,
            "Z:Temp": [37.0] + [np.nan] * (n - 1),
            "Z:Duration": [float(n - 1)] + [np.nan] * (n - 1),
            "W:FeedGlc": np.linspace(0.0, 5.0, n),
            "X:VCD": np.linspace(1.0, 10.0, n),
            "X:Glc": np.linspace(5.0, 1.0, n),
        }
    )


@pytest.fixture(scope="module")
def stub_model(exp_df: pd.DataFrame) -> StubModel:
    feature_names = list(build_feature_matrix(exp_df).columns)
    return StubModel(feature_names)


@pytest.fixture(scope="module")
def client(stub_model: StubModel):
    metadata = {"run_id": "test-run", "model_name": "stub", "rmse_mean": 0.0}
    with patch("src.api.get_best_model", return_value=(stub_model, metadata)):
        from src.api import app

        with TestClient(app) as test_client:
            yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success(client: TestClient, exp_df: pd.DataFrame) -> None:
    z_cols = [c for c in exp_df.columns if c.startswith("Z:")]
    w_cols = [c for c in exp_df.columns if c.startswith("W:")]
    x_cols = [c for c in exp_df.columns if c.startswith("X:")]

    day0 = exp_df.loc[exp_df["Time[day]"] == 0].iloc[0]
    values = {col: [float(day0[col])] for col in z_cols}
    for col in w_cols + x_cols:
        values[col] = exp_df[col].astype(float).tolist()

    payload = {
        "timestamps": exp_df["Time[day]"].astype(float).tolist(),
        "values": values,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"titer"}
    assert body["titer"] == pytest.approx(float(np.exp(7.0)))


def test_predict_wrong_schema(client: TestClient) -> None:
    """W: array length must match timestamps length — mismatch should 422."""
    payload = {
        "timestamps": [0.0, 1.0, 2.0],
        "values": {
            "Z:Temp": [37.0],
            "W:Glucose": [5.0, 6.0],  # wrong length
        },
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
