import numpy as np

from src.validation import (
    check_missing_data,
    check_setpoints_validation,
    check_titer_is_only_set_for_last_day,
    validate_data,
)


class TestCheckSetpointsValidation:
    def test_passes_on_valid_data(self, valid_df):
        assert check_setpoints_validation(valid_df) is True

    def test_fails_when_setpoint_recorded_on_non_zero_day(self, valid_df):
        df = valid_df.copy()
        df.loc[(df["Exp"] == 1) & (df["Time[day]"] == 1), "Z:Temp"] = 38.0
        assert not check_setpoints_validation(df)

    def test_fails_when_experiment_missing_day_zero_setpoints(self, valid_df):
        df = valid_df.copy()
        mask = (df["Exp"] == 1) & (df["Time[day]"] == 0)
        df.loc[mask, ["Z:ExpDuration", "Z:Temp"]] = np.nan
        assert not check_setpoints_validation(df)


class TestCheckTiterIsOnlySetForLastDay:
    def test_passes_on_valid_data(self, valid_df):
        assert check_titer_is_only_set_for_last_day(valid_df)

    def test_fails_when_titer_recorded_on_non_final_day(self, valid_df):
        df = valid_df.copy()
        df.loc[(df["Exp"] == 1) & (df["Time[day]"] == 1), "Y:Titer"] = 5.0
        assert not check_titer_is_only_set_for_last_day(df)

    def test_fails_when_experiment_missing_titer(self, valid_df):
        df = valid_df.copy()
        df.loc[df["Exp"] == 1, "Y:Titer"] = np.nan
        assert not check_titer_is_only_set_for_last_day(df)

    def test_fails_when_titer_present_but_not_on_expduration(self, valid_df):
        df = valid_df.copy()
        exp1_last = (df["Exp"] == 1) & (df["Time[day]"] == 2)
        exp1_earlier = (df["Exp"] == 1) & (df["Time[day]"] == 1)
        df.loc[exp1_last, "Y:Titer"] = np.nan
        df.loc[exp1_earlier, "Y:Titer"] = 10.0
        assert not check_titer_is_only_set_for_last_day(df)


class TestCheckMissingData:
    def test_passes_on_valid_data(self, valid_df):
        assert check_missing_data(valid_df)

    def test_fails_when_dense_column_has_nan(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "W:Glucose"] = np.nan
        assert not check_missing_data(df)

    def test_passes_when_only_setpoints_and_titer_have_nan(self, valid_df):
        df = valid_df.copy()
        assert df["Z:Temp"].isna().any()
        assert df["Y:Titer"].isna().any()
        assert check_missing_data(df)

    def test_fails_when_time_column_has_nan(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "Time[day]"] = np.nan
        assert not check_missing_data(df)


class TestValidateData:
    def test_returns_empty_list_on_valid_data(self, valid_df):
        assert validate_data(valid_df) == []

    def test_reports_setpoint_failure(self, valid_df):
        df = valid_df.copy()
        df.loc[(df["Exp"] == 1) & (df["Time[day]"] == 1), "Z:Temp"] = 38.0
        errors = validate_data(df)
        assert len(errors) == 1
        assert "Setpoints" in errors[0]

    def test_reports_titer_failure(self, valid_df):
        df = valid_df.copy()
        df.loc[df["Exp"] == 1, "Y:Titer"] = np.nan
        errors = validate_data(df)
        assert len(errors) == 1
        assert "Y:Titer" in errors[0]

    def test_reports_missing_data_failure(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "W:Glucose"] = np.nan
        errors = validate_data(df)
        assert len(errors) == 1
        assert "NaN" in errors[0]

    def test_reports_multiple_failures(self, valid_df):
        df = valid_df.copy()
        df.loc[(df["Exp"] == 1) & (df["Time[day]"] == 1), "Z:Temp"] = 38.0
        df.loc[df["Exp"] == 2, "Y:Titer"] = np.nan
        df.loc[0, "W:Glucose"] = np.nan
        assert len(validate_data(df)) == 3
