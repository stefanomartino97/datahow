import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def valid_df() -> pd.DataFrame:
    """Build a synthetic, validation-passing dataframe."""
    durations = {1: 2, 2: 3}

    rows = []
    for exp, duration in durations.items():
        for day in range(duration + 1):
            rows.append(
                {
                    "Exp": exp,
                    "Time[day]": day,
                    "Z:ExpDuration": duration if day == 0 else np.nan,
                    "Z:Temp": 37.0 if day == 0 else np.nan,
                    "W:Glucose": 5.0 + day,
                    "X:VCD": 1.0 + day,
                    "Y:Titer": (10.0 * exp) if day == duration else np.nan,
                }
            )
    return pd.DataFrame(rows)
