import pandas as pd


def get_setpoints_cols(df: pd.DataFrame) -> list[str]:
    """Return the names of columns in ``df`` that represent setpoints.

    Setpoint columns are identified by the ``"Z:"`` prefix convention.

    Args:
        df: DataFrame whose columns will be inspected.

    Returns:
        A list of column names from ``df`` that start with ``"Z:"``.
    """
    setpoints_cols = [col for col in df.columns if col.startswith("Z:")]
    return setpoints_cols
