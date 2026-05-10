from pydantic_settings import BaseSettings

from pathlib import Path

SRC_FOLDER = Path(__file__).parent.resolve()
PROJECT_FOLDER = SRC_FOLDER.parent
DATA_FOLDER = PROJECT_FOLDER / "data"
TRAIN_DATA_PATH = DATA_FOLDER / "datahow_interview_train_data.csv"
TRAIN_TARGETS_PATH = DATA_FOLDER / "datahow_interview_train_targets.csv"
TEST_DATA_PATH = DATA_FOLDER / "datahow_interview_test_data.csv"
TEST_TARGETS_PATH = DATA_FOLDER / "datahow_interview_test_targets.csv"


class Settings(BaseSettings):
    # ==================================================
    # Training params
    # ==================================================
    mlflow_experiment_name: str = "titer_prediction"
    n_seeds: int = 5
    inner_cv_seed: int = 0
    model_random_seed: int = 42
    sklearn_loss_metric: str = "neg_root_mean_squared_error"
    n_kfold_splits: int = 5


settings = Settings()
