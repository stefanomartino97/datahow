import mlflow
from sklearn.pipeline import Pipeline


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
