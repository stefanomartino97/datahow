import pandas as pd
from src.utils import get_setpoints_cols


def check_setpoints_validation(df: pd.DataFrame) -> bool:
    """Check that ``Z:`` setpoints are recorded once per experiment, on day 0.

    Setpoints describe the experimental design and are time-invariant, so the
    expectation is that they appear only on day 0 of each experiment and are
    NaN on every subsequent day.

    Args:
        df: Long-format dataframe with one row per (experiment, day). Must
            contain an ``Exp`` column, a ``Time[day]`` column, and the ``Z:``
            setpoint columns.

    Returns:
        True if the check passes, False if it fails (setpoints recorded on a
        day other than 0, or some experiment is missing its day-0 setpoints).
    """
    num_experiments = df["Exp"].nunique()
    setpoints_cols = get_setpoints_cols(df)

    rows_with_setpoints = df.dropna(subset=setpoints_cols, how="all")
    days_with_setpoints = rows_with_setpoints["Time[day]"].unique()

    if list(days_with_setpoints) != [0]:
        return False

    if len(rows_with_setpoints) != num_experiments:
        return False

    return True


def check_titer_is_only_set_for_last_day(train_df: pd.DataFrame) -> bool:
    """Check that ``Y:Titer`` is recorded once per experiment, on its last day.

    The target is the final mAb concentration, expected exactly once per
    experiment on the day equal to ``Z:ExpDuration``.

    Args:
        train_df: Long-format dataframe with one row per (experiment, day).
            Must contain ``Exp``, ``Time[day]``, ``Z:ExpDuration``, and
            ``Y:Titer`` columns.

    Returns:
        True if the check passes, False if it fails (number of titer
        measurements differs from the number of experiments, or any titer
        measurement is not on ``Z:ExpDuration``).
    """
    num_experiments = train_df["Exp"].nunique()
    duration_per_exp = train_df.query("`Time[day]` == 0")[["Exp", "Z:ExpDuration"]]

    titer_rows = train_df.dropna(subset="Y:Titer")[
        ["Exp", "Time[day]", "Y:Titer"]
    ].merge(duration_per_exp, on="Exp")

    if len(titer_rows) != num_experiments:
        return False

    if not (titer_rows["Time[day]"] == titer_rows["Z:ExpDuration"]).all():
        return False

    return True


def check_missing_data(df: pd.DataFrame) -> bool:
    """Check that no NaN appears outside the structurally sparse columns.

    NaN is structurally expected in the ``Z:`` setpoints (recorded only on
    day 0) and in ``Y:Titer`` (recorded only on the final day). Every other
    column should be populated for every (experiment, day) pair.

    Args:
        df: Long-format dataframe with one row per (experiment, day),
            containing the ``Z:`` setpoint columns and ``Y:Titer``.

    Returns:
        True if the check passes, False if it fails (any column outside the
        expected-sparse set contains NaN).
    """
    setpoints_cols = get_setpoints_cols(df)
    expected_sparse_cols = setpoints_cols + ["Y:Titer"]
    dense_cols = df.columns.difference(expected_sparse_cols)
    nans_per_col = df[dense_cols].isna().sum()

    return bool((nans_per_col == 0).all())


def validate_data(df: pd.DataFrame) -> list[str]:
    """Run all dataset validations and collect error messages from failed checks.

    Calls each individual ``check_*`` validator. For every check that returns
    False, appends a descriptive error message to the returned list. The
    pipeline can stop downstream processing whenever the returned list is
    non-empty.

    Args:
        df: Long-format dataframe with one row per (experiment, day),
            containing the ``Z:`` setpoint columns, the ``Y:Titer`` target,
            and the ``W:`` / ``X:`` daily columns.

    Returns:
        A list of error messages, one per failed check. Empty if all
        validations pass.
    """
    errors: list[str] = []
    if not check_setpoints_validation(df):
        errors.append(
            "Setpoints are not all on day 0, or some experiments are missing setpoints."
        )

    if not check_titer_is_only_set_for_last_day(df):
        errors.append(
            "Y:Titer is not recorded exactly once per experiment on its last day."
        )

    if not check_missing_data(df):
        errors.append("Unexpected NaN found in non-setpoint, non-target columns.")

    return errors
