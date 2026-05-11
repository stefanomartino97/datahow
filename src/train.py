import argparse
import logging
import warnings
from collections.abc import Callable

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from src.data import load_raw_data, preprocess_raw_data
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from src.config import settings
from src.models import get_best_model
from src.models import model_candidates

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")
logging.getLogger("mlflow.utils.uv_utils").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)


def _make_search(
    factory: Callable[[], Pipeline],
    param_grid: dict,
    inner_cv: KFold,
    n_iter: int | None,
    seed: int,
) -> GridSearchCV | RandomizedSearchCV:
    """Build a CV hyperparameter searcher for a candidate pipeline.

    Uses :class:`GridSearchCV` when 'n_iter' is 'None', otherwise
    :class:`RandomizedSearchCV` with 'n_iter' sampled configurations.

    Args:
        factory: Zero-arg callable returning a fresh 'sklearn' 'Pipeline'.
        param_grid: Hyperparameter search space (lists for grid, lists or
            scipy.stats distributions for randomized).
        inner_cv: :class:`KFold` splitter used to score each configuration.
        n_iter: Number of randomized samples, or 'None' for grid search.
        seed: Random seed for :class:`RandomizedSearchCV` (ignored for grid).

    Returns:
        A configured, unfitted :class:`GridSearchCV` or
        :class:`RandomizedSearchCV`.
    """
    if n_iter is None:
        return GridSearchCV(
            factory(),
            param_grid,
            cv=inner_cv,
            scoring=settings.sklearn_loss_metric,
            n_jobs=-1,
        )

    return RandomizedSearchCV(
        factory(),
        param_grid,
        n_iter=n_iter,
        cv=inner_cv,
        scoring=settings.sklearn_loss_metric,
        n_jobs=-1,
        random_state=seed,
    )


def repeated_nested_cv(
    factory: Callable[[], Pipeline],
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_seeds: int = settings.n_seeds,
    n_iter: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate model performance with repeated nested 5-fold CV.

    Runs outer 5-fold CV 'n_seeds' times (different shuffle each repeat).
    On every outer split, an inner 5-fold CV picks the hyperparameters; the
    chosen model is scored on the held-out outer fold. Nesting the HP search
    keeps the reported metrics free of selection bias.

    Args:
        factory: Zero-arg callable returning a fresh 'sklearn' 'Pipeline'.
        param_grid: Hyperparameter search space.
        X: Feature matrix.
        y: Target series aligned to 'X'.
        n_seeds: Number of outer-CV repeats. Defaults to 'settings.n_seeds'.
        n_iter: 'None' for grid search, otherwise number of randomized
            configurations.

    Returns:
        '(rmse, r2)' arrays of length 'n_seeds x 5' with the per-fold
        scores. Aggregate with '.mean()' / '.std()'.
    """
    inner_cv = KFold(
        n_splits=settings.n_kfold_splits,
        shuffle=True,
        random_state=settings.inner_cv_seed,
    )
    rmse_all, r2_all = [], []
    for seed in range(n_seeds):
        outer_cv = KFold(
            n_splits=settings.n_kfold_splits, shuffle=True, random_state=seed
        )
        search = _make_search(factory, param_grid, inner_cv, n_iter, seed)
        res = cross_validate(
            search,
            X,
            y,
            cv=outer_cv,
            scoring=[settings.sklearn_loss_metric, "r2"],
        )
        rmse_all.append(-res[f"test_{settings.sklearn_loss_metric}"])
        r2_all.append(res["test_r2"])

    return np.concatenate(rmse_all), np.concatenate(r2_all)


def fit_final(
    factory: Callable[[], Pipeline],
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int | None = None,
) -> tuple[Pipeline, dict]:
    """Fit the final model on all training data with HPs picked by inner CV.

    Runs a single 5-fold CV hyperparameter search and returns the best
    estimator refit on the full '(X, y)'. Call once per candidate after
    :func:`repeated_nested_cv` has produced the unbiased performance estimate.

    Args:
        factory: Zero-arg callable returning a fresh 'sklearn' 'Pipeline'.
        param_grid: Hyperparameter search space.
        X: Feature matrix to fit on.
        y: Target series aligned to 'X'.
        n_iter: 'None' for grid search, otherwise number of randomized
            configurations.

    Returns:
        '(best_estimator, best_params)': the refit pipeline and the chosen
        hyperparameters (keys use the 'model__<param>' form).
    """
    inner_cv = KFold(
        n_splits=settings.n_kfold_splits,
        shuffle=True,
        random_state=settings.inner_cv_seed,
    )
    search = _make_search(
        factory=factory,
        param_grid=param_grid,
        inner_cv=inner_cv,
        n_iter=n_iter,
        seed=settings.model_random_seed,
    )
    search.fit(X, y)

    return search.best_estimator_, search.best_params_


def _log_pipeline_artifacts(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Log the shared MLflow artifacts inside the active run.

    Records the model/n_features/n_samples params, the fitted
    sklearn pipeline (serialized with skops, with the trusted-types list
    needed by lightgbm/xgboost boosters), and feature_columns.json.

    Args:
        model_name: Value for the model param.
        pipeline: Fitted sklearn pipeline to log under the artifact name
            "model".
        X: Feature matrix used to derive n_features, the input example,
            and the column list.
        y: Target series used to derive n_samples.
    """
    mlflow.log_param("model", model_name)
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_samples", len(y))
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        input_example=X.iloc[:2],
        serialization_format="skops",
        skops_trusted_types=[
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMRegressor",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor",
        ],
    )
    mlflow.log_dict({"feature_columns": list(X.columns)}, "feature_columns.json")


def train_and_log_candidate_model(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Train one candidate model end-to-end and log it to MLflow.

    Looks up the spec in :data:`src.models.model_candidates`, runs
    :func:`repeated_nested_cv` for unbiased RMSE/R² and :func:`fit_final` for
    the deployable pipeline, and records both inside a single MLflow run.
    Logged: 'model'/'n_features'/'n_samples'/'n_seeds' and
    'best_<param>' params; 'rmse_mean'/'rmse_std'/'r2_mean'/'r2_std'
    metrics; the fitted pipeline plus 'feature_columns.json'.

    Args:
        name: Key into :data:`src.models.model_candidates`.
        X: Feature matrix from :func:`src.data.preprocess_raw_data`.
        y: Target series aligned to 'X' (log-titer).

    Returns:
        Summary dict with 'run_id', 'name', 'rmse_mean', 'rmse_std',
        'r2_mean' - consumed by :func:`train_all` to pick the best model.
    """
    spec = model_candidates[name]

    with mlflow.start_run(run_name=name) as run:
        rmse, r2 = repeated_nested_cv(
            factory=spec["factory"],
            param_grid=spec["param_grid"],
            X=X,
            y=y,
            n_iter=spec["n_iter"],
        )

        final_pipe, best_params = fit_final(
            factory=spec["factory"],
            param_grid=spec["param_grid"],
            X=X,
            y=y,
            n_iter=spec["n_iter"],
        )

        _log_pipeline_artifacts(model_name=name, pipeline=final_pipe, X=X, y=y)
        mlflow.log_param("n_seeds", settings.n_seeds)

        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        mlflow.log_metric("rmse_mean", float(rmse.mean()))
        mlflow.log_metric("rmse_std", float(rmse.std()))
        mlflow.log_metric("r2_mean", float(r2.mean()))
        mlflow.log_metric("r2_std", float(r2.std()))

        return {
            "run_id": run.info.run_id,
            "name": name,
            "rmse_mean": float(rmse.mean()),
            "rmse_std": float(rmse.std()),
            "r2_mean": float(r2.mean()),
        }


def train_all() -> list[dict]:
    """Train every candidate model and tag the best run in MLflow.

    Loads and preprocesses the data once, trains each candidate in
    :data:`src.models.model_candidates` (one MLflow run per candidate), and
    tags the lowest-RMSE run with 'best=true' so inference can find it.

    Returns:
        List of per-candidate summary dicts (in training order), as returned
        by :func:`train_and_log_candidate_model`.
    """
    mlflow.set_experiment(settings.mlflow_experiment_name)
    raw_data_df = load_raw_data()
    X, y = preprocess_raw_data(raw_data_df)

    print(f"Feature matrix: {X.shape}; target: {y.shape}")

    results = []
    for name in model_candidates:
        print(f"\n==================== {name} ==================== ")
        result = train_and_log_candidate_model(name, X, y)
        print(f"  RMSE = {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
        print(f"  R²   = {result['r2_mean']:.4f}")
        results.append(result)

    best = min(results, key=lambda r: r["rmse_mean"])
    print(f"\nBest: {best['name']} (RMSE = {best['rmse_mean']:.4f})")
    client = mlflow.tracking.MlflowClient()
    client.set_tag(best["run_id"], "stage", "cv_best")

    return results


def retrain_best_on_full_data() -> tuple[Pipeline, dict]:
    """Refit the CV-best model on train+test and log it as the production run.

    Loads the run tagged stage="cv_best" (its pipeline carries the CV-
    selected hyperparameters), refits the pipeline on the concatenation of
    the train and test splits returned by :func:`src.data.load_raw_data`
    with split="train_test", then logs the result as a new MLflow run
    tagged stage="production". Any previous production run is demoted
    (tag removed) so :func:`src.models.get_best_model` with
    stage="production" always resolves to this refit.
    The hyperparameters are preserved; only the fit changes.
    The production run stores source_run_id as a param pointing back to
    the CV-best run it was seeded from.
    Returns:
        (refit_pipeline, metadata): the pipeline refit on the full
        dataset, paired with the metadata dict for the new production run.
    """
    mlflow.set_experiment(settings.mlflow_experiment_name)
    print("Loading CV-best model...")
    model, source_metadata = get_best_model(
        settings.mlflow_experiment_name, stage="cv_best"
    )
    print(
        f"  Loaded {source_metadata['model_name']} "
        f"(source run_id={source_metadata['run_id']})"
    )
    raw_data_df = load_raw_data(split="train_test")
    X, y = preprocess_raw_data(raw_data_df)
    print(f"Refitting on full data: X={X.shape}, y={y.shape}")
    model.fit(X, y)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    # Delete any prior production run so the API's lifespan always loads this refit
    for prev in client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'production'",
    ):
        client.delete_tag(prev.info.run_id, "stage")

    run_name = f"{source_metadata['model_name']}_production"
    with mlflow.start_run(run_name=run_name) as run:
        _log_pipeline_artifacts(
            model_name=source_metadata["model_name"],
            pipeline=model,
            X=X,
            y=y,
        )
        mlflow.log_param("source_run_id", source_metadata["run_id"])
        mlflow.set_tag("stage", "production")
        production_run_id = run.info.run_id

    print(f"Logged production run: {production_run_id}")

    metadata = {
        "run_id": production_run_id,
        "stage": "production",
        "model_name": source_metadata["model_name"],
        "source_run_id": source_metadata["run_id"],
    }
    return model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models or refit the CV-best.")
    parser.add_argument(
        "mode",
        choices=["train_all", "retrain_best"],
        help="train_all: CV over all candidates; retrain_best: refit CV-best on full data.",
    )
    args = parser.parse_args()

    if args.mode == "train_all":
        train_all()
    else:
        retrain_best_on_full_data()
