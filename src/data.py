from typing import Literal

import pandas as pd
import numpy as np
from src.config import (
    TEST_DATA_PATH,
    TEST_TARGETS_PATH,
    TRAIN_DATA_PATH,
    TRAIN_TARGETS_PATH,
)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each experiment's time-series into a single feature row.

    For each experiment (one row of the output, indexed by 'Exp'), the following
    features are built from the raw long-form dataframe:

    Setpoints ('Z:*'):
        - 'Z:<col>': the day-0 value of the setpoint, taken as-is. Setpoints
          are constant per experiment, so the day-0 reading represents the
          whole run.

    Feed/action aggregates ('W:*'):
        - 'W:<col>_sum': total amount accumulated over the experiment
          (e.g. cumulative feed volume).
        - 'W:<col>_mean': average per-timepoint value over the experiment.

    Process variables ('X:*'):
        - 'X:<col>_final': value at the last recorded timepoint.
        - 'X:<col>_max': maximum value observed during the experiment.
        - 'X:<col>_mean': time-unweighted mean over all timepoints.
        - 'X:<col>_auc': area under the curve over 'Time[day]', computed
          via the trapezoidal rule (a time-weighted exposure summary).
        - 'X:<col>_peak_day': day on which the variable reached its maximum.
        - 'X:<col>_growth_rate': average rate of change from day 0 to the
          peak, '(peak - start) / peak_day'; '0' if the peak is at day 0.
        - 'X:<col>_decline_rate': average rate of change from the peak to
          the final timepoint, '(final - peak) / (last_day - peak_day)';
          '0' if the peak coincides with the last day.

    Args:
        df: Long-form dataframe with one row per ('Exp', 'Time[day]')
            pair and columns prefixed 'Z:', 'W:', 'X:' for the three
            variable groups.

    Returns:
        Wide-form dataframe with one row per experiment, indexed by 'Exp',
        whose columns are the features described above.
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

        # ====================
        # Z cols
        # ====================
        for col in z_cols:
            row[col] = day0[col]

        # ====================
        # W cols
        # ====================
        for col in w_cols:
            row[f"{col}_sum"] = exp_df[col].sum()
            row[f"{col}_mean"] = exp_df[col].mean()

        # ====================
        # X cols
        # ====================
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


def load_raw_data(
    split: Literal["train", "test", "train_test"] = "train",
) -> pd.DataFrame:
    """Load and merge the features and targets for a given split.

    Reads the variables CSV and the targets CSV for the requested split,
    drops the 'RowID' column from each, and outer-joins them on 'Exp'
    and 'Time[day]' so rows present in only one file are preserved (e.g.
    titer measurements taken at timepoints not in the variables file).

    When 'split="train_test"', the train and test long-form frames are
    loaded independently and concatenated row-wise. 'Exp' values are
    disjoint across splits, so no key collisions occur.

    Args:
        split: Which split to load. '"train"' reads
            'TRAIN_DATA_PATH'/'TRAIN_TARGETS_PATH', '"test"' reads
            'TEST_DATA_PATH'/'TEST_TARGETS_PATH' (the test targets file
            is a template with placeholder values), and '"train_test"'
            returns the concatenation of the two.

    Returns:
        Long-form dataframe with one row per ('Exp', 'Time[day]') pair,
        carrying the 'Z:'/'W:'/'X:' variable columns alongside the
        'Y:Titer' target column. Cells without a corresponding measurement
        in the source CSV are 'NaN'.
    """
    paths = {
        "train": (TRAIN_DATA_PATH, TRAIN_TARGETS_PATH),
        "test": (TEST_DATA_PATH, TEST_TARGETS_PATH),
    }

    if split == "train_test":
        return pd.concat(
            [load_raw_data("train"), load_raw_data("test")], ignore_index=True
        )

    data_path, targets_path = paths[split]
    data_df = pd.read_csv(data_path).drop(columns="RowID")
    targets_df = pd.read_csv(targets_path).drop(columns="RowID")

    return pd.merge(data_df, targets_df, how="outer", on=["Exp", "Time[day]"])


def preprocess_raw_data(raw_data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the modeling feature matrix and log-transformed target.

    Collapses the long-form merged dataframe into one feature row per
    experiment via :func:`_build_feature_matrix`, and pairs it with the
    final-titer target on the natural-log scale. 'log Y:Titer' is used so
    the regressors model a roughly symmetric, additive-error target rather
    than the heavy-tailed raw titer.

    Args:
        raw_data_df: Long-form dataframe as returned by :func:`load_raw_data`,
            containing the 'Z:'/'W:'/'X:' variable columns and the
            'Y:Titer' target column.

    Returns:
        A tuple '(X, y)' where:
            - 'X': feature matrix (one row per experiment, indexed by
              'Exp') as produced by :func:`_build_feature_matrix`.
            - 'y': Series named '"log_titer"', aligned to 'X''s index,
              containing 'log(Y:Titer)' for each experiment.
    """

    titer_by_exp = raw_data_df.dropna(subset=["Y:Titer"]).set_index("Exp")["Y:Titer"]
    X = build_feature_matrix(raw_data_df)
    y = np.log(titer_by_exp.loc[X.index]).rename("log_titer")

    return X, y


if __name__ == "__main__":
    raw_data_df = load_raw_data(split="train")
    print(raw_data_df.shape)

    X, y = preprocess_raw_data(raw_data_df)
    print(X.shape)
    print(y.shape)
