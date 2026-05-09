from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.config import settings


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
