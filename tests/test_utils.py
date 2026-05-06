import pandas as pd

from src.utils import get_setpoints_cols


class TestGetSetpointsCols:
    def test_returns_only_z_prefixed_columns(self):
        df = pd.DataFrame(
            {
                "Z:temp": [1, 2],
                "Z:pressure": [3, 4],
                "feature_a": [5, 6],
                "target": [7, 8],
            }
        )
        assert get_setpoints_cols(df) == ["Z:temp", "Z:pressure"]

    def test_preserves_column_order(self):
        df = pd.DataFrame(columns=["a", "Z:b", "c", "Z:a", "Z:c"])
        assert get_setpoints_cols(df) == ["Z:b", "Z:a", "Z:c"]

    def test_returns_empty_list_when_no_setpoints(self):
        df = pd.DataFrame({"feature_a": [1], "feature_b": [2]})
        assert get_setpoints_cols(df) == []

    def test_returns_empty_list_for_empty_dataframe(self):
        df = pd.DataFrame()
        assert get_setpoints_cols(df) == []

    def test_does_not_match_substring_or_lowercase_prefix(self):
        df = pd.DataFrame(columns=["z:lower", "XZ:mid", "Z", "Z:ok"])
        assert get_setpoints_cols(df) == ["Z:ok"]

    def test_returns_all_when_every_column_is_setpoint(self):
        df = pd.DataFrame(columns=["Z:a", "Z:b"])
        assert get_setpoints_cols(df) == ["Z:a", "Z:b"]
