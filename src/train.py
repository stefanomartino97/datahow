"""Training pipeline for mAb titer prediction.

Trains the candidate models from the modeling notebook on log Y:Titer,
evaluates them with repeated nested 5-fold CV, and logs everything to MLflow.
The lowest-RMSE run is tagged 'best' so it can be loaded later via
`get_best_model`.

Run:
    python -m src.train
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "datahow_interview_train_data.csv"
TRAIN_TARGETS_PATH = DATA_DIR / "datahow_interview_train_targets.csv"

EXPERIMENT_NAME = "titer_prediction"
N_SEEDS = 5
INNER_CV_SEED = 0


# ---------- Data ----------


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each experiment's time-series into one row of features.

    Mirrors the modeling notebook's feature construction exactly: 13 Z setpoints
    as-is, sum/mean of W aggregates, and final/max/mean/AUC plus growth-phase
    summaries (peak_day, growth_rate, decline_rate) of each X variable.
    """
    z_cols = [c for c in df.columns if c.startswith("Z:")]
    w_cols = [c for c in df.columns if c.startswith("W:")]
    x_cols = [c for c in df.columns if c.startswith("X:")]

    rows = []
    for exp, exp_df in df.groupby("Exp"):
        exp_df = exp_df.sort_values("Time[day]")
        time = exp_df["Time[day]"].values
        day0 = exp_df.loc[exp_df["Time[day]"] == 0].iloc[0]

        row: dict[str, object] = {"Exp": exp}
        for col in z_cols:
            row[col] = day0[col]
        for col in w_cols:
            row[f"{col}_sum"] = exp_df[col].sum()
            row[f"{col}_mean"] = exp_df[col].mean()
        for col in x_cols:
            series = exp_df[col].values
            row[f"{col}_final"] = series[-1]
            row[f"{col}_max"] = series.max()
            row[f"{col}_mean"] = series.mean()
            row[f"{col}_auc"] = np.trapezoid(series, time)
            peak_idx = int(np.argmax(series))
            peak_day = time[peak_idx]
            last_day = time[-1]
            row[f"{col}_peak_day"] = peak_day
            row[f"{col}_growth_rate"] = (
                (series[peak_idx] - series[0]) / peak_day if peak_day > 0 else 0.0
            )
            row[f"{col}_decline_rate"] = (
                (series[-1] - series[peak_idx]) / (last_day - peak_day)
                if last_day > peak_day
                else 0.0
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("Exp")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the train CSVs and return (X_features, y = log Y:Titer)."""
    train_df = pd.read_csv(TRAIN_DATA_PATH).drop(columns="RowID")
    targets_df = pd.read_csv(TRAIN_TARGETS_PATH).drop(columns="RowID")
    df = pd.merge(train_df, targets_df, how="outer", on=["Exp", "Time[day]"])

    titer_by_exp = df.dropna(subset=["Y:Titer"]).set_index("Exp")["Y:Titer"]
    X = build_feature_matrix(df)
    y = np.log(titer_by_exp.loc[X.index]).rename("log_titer")
    return X, y


# ---------- Candidate models ----------


def make_ridge_pipe() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge())])


def make_pls_pipe() -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("model", PLSRegression(scale=False))]
    )


def make_xgb_pipe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(random_state=42, n_jobs=1)),
        ]
    )


def make_lgbm_pipe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LGBMRegressor(random_state=42, n_jobs=1, verbose=-1)),
        ]
    )


CANDIDATES: dict[str, dict] = {
    "ridge": {
        "factory": make_ridge_pipe,
        "param_grid": {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
        "n_iter": None,
    },
    "pls": {
        "factory": make_pls_pipe,
        "param_grid": {"model__n_components": list(range(1, 16))},
        "n_iter": None,
    },
    "xgboost": {
        "factory": make_xgb_pipe,
        "param_grid": {
            "model__n_estimators": [100, 300, 500],
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": loguniform(0.01, 100),
            "model__reg_alpha": loguniform(0.0001, 10),
        },
        "n_iter": 40,
    },
    "lightgbm": {
        "factory": make_lgbm_pipe,
        "param_grid": {
            "model__n_estimators": [100, 300, 500],
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__num_leaves": randint(8, 64),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_samples": randint(3, 20),
            "model__reg_lambda": loguniform(0.01, 100),
            "model__reg_alpha": loguniform(0.0001, 10),
        },
        "n_iter": 40,
    },
}


# ---------- Training ----------


def _make_search(
    factory: Callable[[], Pipeline],
    param_grid: dict,
    inner_cv: KFold,
    n_iter: int | None,
    seed: int,
) -> GridSearchCV | RandomizedSearchCV:
    if n_iter is None:
        return GridSearchCV(
            factory(),
            param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
    return RandomizedSearchCV(
        factory(),
        param_grid,
        n_iter=n_iter,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=seed,
    )


def repeated_nested_cv(
    factory: Callable[[], Pipeline],
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_seeds: int = N_SEEDS,
    n_iter: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeated nested 5-fold CV. Returns RMSE and R² across n_seeds × 5 folds."""
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=INNER_CV_SEED)
    rmse_all, r2_all = [], []
    for seed in range(n_seeds):
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        search = _make_search(factory, param_grid, inner_cv, n_iter, seed)
        res = cross_validate(
            search,
            X,
            y,
            cv=outer_cv,
            scoring=["neg_root_mean_squared_error", "r2"],
        )
        rmse_all.append(-res["test_neg_root_mean_squared_error"])
        r2_all.append(res["test_r2"])
    return np.concatenate(rmse_all), np.concatenate(r2_all)


def fit_final(
    factory: Callable[[], Pipeline],
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int | None = None,
) -> tuple[Pipeline, dict]:
    """Fit the final model on all training data; HPs chosen by 5-fold inner CV."""
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=INNER_CV_SEED)
    search = _make_search(factory, param_grid, inner_cv, n_iter, seed=42)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def train_and_log(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Train one candidate end-to-end (CV + final fit) and log to MLflow."""
    spec = CANDIDATES[name]

    with mlflow.start_run(run_name=name) as run:
        rmse, r2 = repeated_nested_cv(
            spec["factory"], spec["param_grid"], X, y, n_iter=spec["n_iter"]
        )
        final_pipe, best_params = fit_final(
            spec["factory"], spec["param_grid"], X, y, n_iter=spec["n_iter"]
        )

        mlflow.log_param("model", name)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", len(y))
        mlflow.log_param("n_seeds", N_SEEDS)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        mlflow.log_metric("rmse_mean", float(rmse.mean()))
        mlflow.log_metric("rmse_std", float(rmse.std()))
        mlflow.log_metric("r2_mean", float(r2.mean()))
        mlflow.log_metric("r2_std", float(r2.std()))

        mlflow.sklearn.log_model(
            final_pipe,
            artifact_path="model",
            input_example=X.iloc[:2],
        )
        mlflow.log_dict({"feature_columns": list(X.columns)}, "feature_columns.json")

        return {
            "run_id": run.info.run_id,
            "name": name,
            "rmse_mean": float(rmse.mean()),
            "rmse_std": float(rmse.std()),
            "r2_mean": float(r2.mean()),
        }


def train_all() -> list[dict]:
    """Train every candidate, log to MLflow, tag the lowest-RMSE run as 'best'."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    X, y = load_data()
    print(f"Feature matrix: {X.shape}; target: {y.shape}")

    results = []
    for name in CANDIDATES:
        print(f"\n=== {name} ===")
        result = train_and_log(name, X, y)
        print(f"  RMSE = {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
        print(f"  R²   = {result['r2_mean']:.4f}")
        results.append(result)

    best = min(results, key=lambda r: r["rmse_mean"])
    print(f"\nBest: {best['name']} (RMSE = {best['rmse_mean']:.4f})")
    client = mlflow.tracking.MlflowClient()
    client.set_tag(best["run_id"], "best", "true")
    return results


# ---------- Inference ----------


def get_best_model(
    experiment_name: str = EXPERIMENT_NAME,
) -> tuple[Pipeline, dict]:
    """Load the lowest-RMSE model from `experiment_name` along with its metadata.

    Returns:
        (model, metadata) where metadata carries the run id, model name, and CV
        metrics. `model.predict(X)` returns predictions on log Y:Titer; exponentiate
        for titer-space predictions.

    Raises:
        ValueError: if the experiment doesn't exist or has no runs.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse_mean ASC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs in MLflow experiment '{experiment_name}'")

    best = runs[0]
    model = mlflow.sklearn.load_model(f"runs:/{best.info.run_id}/model")
    metadata = {
        "run_id": best.info.run_id,
        "model_name": best.data.params.get("model"),
        "rmse_mean": best.data.metrics.get("rmse_mean"),
        "rmse_std": best.data.metrics.get("rmse_std"),
        "r2_mean": best.data.metrics.get("r2_mean"),
    }
    return model, metadata


if __name__ == "__main__":
    train_all()
