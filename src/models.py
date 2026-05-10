from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.config import settings
import mlflow


def get_best_model(experiment_name: str) -> tuple[Pipeline, dict]:
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


# ==================================================
# Models configurations
# ==================================================
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
            ("model", XGBRegressor(random_state=settings.model_random_seed, n_jobs=1)),
        ]
    )


def make_lgbm_pipe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LGBMRegressor(
                    random_state=settings.model_random_seed, n_jobs=1, verbose=-1
                ),
            ),
        ]
    )


model_candidates: dict[str, dict] = {
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
