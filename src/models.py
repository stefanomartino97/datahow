from typing import Literal

from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.config import settings
import mlflow


def get_best_model(
    experiment_name: str,
    stage: Literal["cv_best", "production"] = "cv_best",
) -> tuple[Pipeline, dict]:
    """Load a tagged model from `experiment_name` along with its metadata.

    Two stages are supported:

    * "cv_best" returns the candidate with the lowest nested-CV RMSE —
      the run tagged stage=cv_best by :func:`src.train.train_all`. Used
      for comparing candidates and as the seed for the full-data refit.
    * "production" returns the model refit on train+test combined — the
      run tagged stage=production by
      :func:`src.train.retrain_best_on_full_data`. Used by the serving API.

    `model.predict(X)` returns predictions on log Y:Titer; exponentiate for
    titer-space predictions.

    Args:
        experiment_name: MLflow experiment to search.
        stage: Which tagged run to load.

    Returns:
        (model, metadata). For "cv_best" the metadata carries CV
        metrics; for "production" it also carries source_run_id
        pointing back to the CV run the refit was seeded from.

    Raises:
        ValueError: if the experiment doesn't exist or no run carries the
            requested stage tag.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found")

    order_by = (
        ["metrics.rmse_mean ASC"]
        if stage == "cv_best"
        else ["attributes.start_time DESC"]
    )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.stage = '{stage}'",
        order_by=order_by,
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No run tagged stage='{stage}' in MLflow experiment '{experiment_name}'"
        )

    best = runs[0]
    model = mlflow.sklearn.load_model(f"runs:/{best.info.run_id}/model")
    metadata = {
        "run_id": best.info.run_id,
        "stage": stage,
        "model_name": best.data.params.get("model"),
        "rmse_mean": best.data.metrics.get("rmse_mean"),
        "rmse_std": best.data.metrics.get("rmse_std"),
        "r2_mean": best.data.metrics.get("r2_mean"),
        "source_run_id": best.data.params.get("source_run_id"),
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
