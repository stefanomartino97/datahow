from src.config import DATA_FOLDER
import pandas as pd


def read_train_df() -> pd.DataFrame:
    """Load and merge the training features and targets into a single DataFrame.

    Reads the variables and targets CSVs from ``DATA_FOLDER``, drops the
    ``RowID`` column from each, and outer-joins them on ``Exp`` and
    ``Time[day]`` so rows present in only one file are preserved.
    """
    df_vars = pd.read_csv(DATA_FOLDER / "datahow_interview_train_data.csv")
    df_vars = df_vars.drop(columns="RowID")

    targets_df = pd.read_csv(DATA_FOLDER / "datahow_interview_train_targets.csv")
    targets_df = targets_df.drop(columns="RowID")

    train_df = pd.merge(df_vars, targets_df, how="outer", on=["Exp", "Time[day]"])

    return train_df
