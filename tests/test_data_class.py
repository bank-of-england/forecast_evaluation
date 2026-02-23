from unittest.mock import patch

import pandas as pd
import pytest

from forecast_evaluation.data.ForecastData import (
    ForecastData,
    _check_duplicates,
    _validate_records,
)
from forecast_evaluation.data.sample_data import create_sample_forecasts, create_sample_outturns
from forecast_evaluation.data.schema import FORECAST_REQUIRED_COLUMNS, OUTTURN_REQUIRED_COLUMNS


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def sample_outturns() -> pd.DataFrame:
    return create_sample_outturns()


@pytest.fixture
def sample_forecasts() -> pd.DataFrame:
    return create_sample_forecasts()


@pytest.fixture
def forecastdata_multi_ids() -> ForecastData:
    outturns_df = create_sample_outturns()
    forecast_df_1 = create_sample_forecasts()
    forecast_df_1["new label"] = "ABC"
    forecast_df_2 = create_sample_forecasts()
    forecast_df_2["new label"] = "EFG"

    # Add own forecasts
    forecast_data = ForecastData(outturns_data=outturns_df)
    forecast_data.add_forecasts(forecast_df_1, extra_ids=["new label"])
    forecast_data.add_forecasts(forecast_df_2, extra_ids=["new label"])

    return forecast_data


@pytest.fixture
def fer_df():
    """Mock FER DataFrame with mixed variables."""
    return pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=5, freq="QE"),
            "variable": ["gdpkp", "invalid_var", "gdpkp", "invalid_var", "gdpkp"],
            "vintage_date": "2022-03-31",
            "source": ["fer"] * 5,
            "frequency": ["Q"] * 5,
            "forecast_horizon": list(range(5)),
            "value": list(range(200, 205)),
        }
    )


# -----------------------
# Constructor Tests
# -----------------------
def test_init_empty():
    fd = ForecastData()
    assert fd._raw_forecasts.empty
    assert fd.forecasts.empty
    assert fd.outturns is not None
    assert isinstance(fd.forecast_required_columns, list)
    assert isinstance(fd.outturn_required_columns, list)


def test_init_with_user_data(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    assert not fd._raw_forecasts.empty
    assert not fd.forecasts.empty


def test_init_with_fer_load():
    with (
        patch(
            "forecast_evaluation.data.loader.load_fer_forecasts",
            return_value=pd.DataFrame(columns=FORECAST_REQUIRED_COLUMNS),
        ),
        patch(
            "forecast_evaluation.data.loader.load_fer_outturns",
            return_value=pd.DataFrame(columns=OUTTURN_REQUIRED_COLUMNS),
        ),
    ):
        fd = ForecastData(load_fer=True)
        assert isinstance(fd._raw_forecasts, pd.DataFrame)


# -----------------------
# Validation Tests
# -----------------------
def test_init_missing_column_raises(sample_outturns, sample_forecasts):
    bad_df = sample_forecasts.drop(columns=["value"])
    with pytest.raises(ValueError, match="Attempting to add data but the following columns are missing"):
        ForecastData(outturns_data=sample_outturns, forecasts_data=bad_df)


# -----------------------
# Adding Data Tests
# -----------------------
def test_add_data(sample_outturns, sample_forecasts):
    fd = ForecastData()
    fd.add_outturns(sample_outturns)
    fd.add_forecasts(sample_forecasts)

    assert len(fd._raw_outturns) == len(sample_outturns)
    assert not fd.outturns.empty
    assert len(fd._raw_forecasts) == len(sample_forecasts)
    assert not fd.forecasts.empty


def test_add_forecasts_duplicate_raises(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    with pytest.raises(ValueError, match="Duplicate records found with different values."):
        sample_forecasts["value"] = sample_forecasts["value"] + 1.0
        fd.add_forecasts(sample_forecasts)


def test_add_forecasts_without_outturns(sample_forecasts):
    fd = ForecastData()
    with pytest.raises(ValueError, match=r"Outturns must be added before forecasts."):
        fd.add_forecasts(sample_forecasts)


def test_add_forecasts_without_specific_outturns(sample_forecasts):
    # Amend sample_forecasts to have a variable not in outturns
    sample_forecasts_amended = sample_forecasts.copy()
    sample_forecasts_amended["variable"] = "non_existent_variable"
    with (
        patch(
            "forecast_evaluation.data.loader.load_fer_forecasts",
            return_value=pd.DataFrame(columns=FORECAST_REQUIRED_COLUMNS),
        ),
        patch(
            "forecast_evaluation.data.loader.load_fer_outturns",
            return_value=pd.DataFrame(columns=OUTTURN_REQUIRED_COLUMNS),
        ),
    ):
        fd = ForecastData(load_fer=True)
        with pytest.warns(UserWarning):
            fd.add_forecasts(sample_forecasts_amended)


def _make_minimal_forecasts_df(*, include_metric: bool, metric_value: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-06-30", "2022-09-30", "2022-12-31"]),
            "variable": ["gdpkp"] * 3,
            "vintage_date": pd.to_datetime(["2022-03-31"] * 3),
            "source": ["user"] * 3,
            "frequency": ["Q"] * 3,
            "forecast_horizon": [0, 1, 2],
            "value": [1.0, 2.0, 3.0],
        }
    )

    if include_metric:
        if metric_value is None:
            raise ValueError("metric_value must be provided when include_metric=True")
        df["metric"] = metric_value

    return df


def test_add_forecasts_metric_column_present_kwarg_omitted(sample_outturns):
    """If the user supplies a metric column, add_forecasts() should use it."""
    fd = ForecastData(outturns_data=sample_outturns)
    df = _make_minimal_forecasts_df(include_metric=True, metric_value="pop")

    fd.add_forecasts(df)

    assert set(fd._raw_forecasts["metric"].unique()) == {"pop"}
    assert set(fd.forecasts["metric"].unique()) == {"pop"}


def test_add_forecasts_metric_kwarg_used_when_column_missing(sample_outturns):
    """If the user does not supply a metric column, metric= should populate it."""
    fd = ForecastData(outturns_data=sample_outturns)
    df = _make_minimal_forecasts_df(include_metric=False)

    fd.add_forecasts(df, metric="yoy")

    assert set(fd._raw_forecasts["metric"].unique()) == {"yoy"}
    assert set(fd.forecasts["metric"].unique()) == {"yoy"}


def test_add_forecasts_metric_kwarg_ignored_when_column_present(sample_outturns):
    """If the user supplies a metric column, metric= should not override it."""
    fd = ForecastData(outturns_data=sample_outturns)
    df = _make_minimal_forecasts_df(include_metric=True, metric_value="yoy")

    fd.add_forecasts(df, metric="pop")

    assert set(fd._raw_forecasts["metric"].unique()) == {"yoy"}
    assert set(fd.forecasts["metric"].unique()) == {"yoy"}


def test_add_fer_data():
    with (
        patch("forecast_evaluation.data.loader.load_fer_forecasts", return_value=create_sample_forecasts()),
        patch("forecast_evaluation.data.loader.load_fer_outturns", return_value=create_sample_outturns()),
    ):
        fd = ForecastData()
        fd.add_fer_data()
        assert isinstance(fd._raw_forecasts, pd.DataFrame)
        assert not fd._raw_forecasts.empty
        assert not fd.outturns.empty
        assert not fd._main_table.empty


def test_add_forecasts_invalid_metric_in_column_raises(sample_outturns):
    fd = ForecastData(outturns_data=sample_outturns)
    df = _make_minimal_forecasts_df(include_metric=True, metric_value="not_a_metric")

    with pytest.raises(ValueError, match=r"Invalid metric values found"):
        fd.add_forecasts(df)


def test_add_forecasts_invalid_metric_kwarg_raises_when_column_missing(sample_outturns):
    fd = ForecastData(outturns_data=sample_outturns)
    df = _make_minimal_forecasts_df(include_metric=False)

    with pytest.raises(ValueError, match=r"Invalid metric values found"):
        fd.add_forecasts(df, metric="not_a_metric")


def test_add_forecasts_mixed_frequencies_raises(sample_outturns):
    """Test that adding forecasts with mixed frequencies raises a ValueError."""
    fd = ForecastData(outturns_data=sample_outturns)

    # Create a dataframe with mixed frequencies
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-03-31", "2022-06-30", "2022-04-30", "2022-05-31"]),
            "variable": ["gdpkp"] * 4,
            "vintage_date": pd.to_datetime(["2022-03-31"] * 4),
            "source": ["user"] * 4,
            "frequency": ["Q", "Q", "M", "M"],  # Mixed quarterly and monthly
            "forecast_horizon": [0, 1, 0, 1],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match=r"Forecasts being added contain multiple frequencies"):
        fd.add_forecasts(df)


def test_add_forecasts_different_frequency_from_existing_raises(sample_outturns):
    """Test that adding forecasts with a different frequency than existing data raises a ValueError."""
    fd = ForecastData(outturns_data=sample_outturns)

    # Add quarterly forecasts first
    df_quarterly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-03-31", "2022-06-30"]),
            "variable": ["gdpkp"] * 2,
            "vintage_date": pd.to_datetime(["2022-03-31"] * 2),
            "source": ["user"] * 2,
            "frequency": ["Q"] * 2,
            "forecast_horizon": [0, 1],
            "value": [1.0, 2.0],
        }
    )
    fd.add_forecasts(df_quarterly)

    # Try to add monthly forecasts
    df_monthly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-04-30", "2022-05-31"]),
            "variable": ["gdpkp"] * 2,
            "vintage_date": pd.to_datetime(["2022-03-31"] * 2),
            "source": ["user"] * 2,
            "frequency": ["M"] * 2,
            "forecast_horizon": [0, 1],
            "value": [3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match=r"New forecasts have frequency 'M' but existing data has frequency 'Q'"):
        fd.add_forecasts(df_monthly)


# -----------------------
# Filtering Tests
# -----------------------
def test_filter_and_clear(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    original_len = len(fd.forecasts)

    fd.filter(start_date="2030-01-01", end_date="2030-12-31")

    assert fd.forecasts.size == 0

    fd.clear_filter()
    assert len(fd.forecasts) == original_len


def test_filter_by_variable():
    with (
        patch("forecast_evaluation.data.loader.load_fer_forecasts", return_value=create_sample_forecasts()),
        patch("forecast_evaluation.data.loader.load_fer_outturns", return_value=create_sample_outturns()),
    ):
        fd = ForecastData(load_fer=True)
        fd.filter(variables=["gdpkp"])
        assert all(fd.forecasts["variable"] == "gdpkp")


def test_filter_with_custom_function(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)

    def custom(df):
        return df[df["forecast_horizon"] < 5]

    fd.filter(custom_filter=custom)
    assert fd.forecasts["forecast_horizon"].max() < 5


def test_filtering_single_source_when_multi_ids(forecastdata_multi_ids):
    forecastdata_multi_ids.filter(sources="ABC")

    assert forecastdata_multi_ids.forecasts["new_label"].unique() == "ABC"
    assert forecastdata_multi_ids._main_table["new_label"].unique() == "ABC"


def test_filtering_single_source_when_single_id(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)

    fd.filter(sources="mpr2")

    assert fd.forecasts["source"].unique() == "mpr2"
    assert fd._main_table["source"].unique() == "mpr2"


def test_filtering_single_source_with_plus_when_multi_ids(forecastdata_multi_ids):
    forecastdata_multi_ids.filter(sources="mpr2 + ABC")

    assert forecastdata_multi_ids.forecasts["new_label"].unique() == "ABC"
    assert forecastdata_multi_ids._main_table["new_label"].unique() == "ABC"


def test_filtering_multi_sources_when_multi_ids(forecastdata_multi_ids):
    forecastdata_multi_ids.filter(sources=["ABC", "EFG"])

    assert all(forecastdata_multi_ids.forecasts["new_label"].unique() == ["ABC", "EFG"])
    assert all(forecastdata_multi_ids._main_table["new_label"].unique() == ["ABC", "EFG"])


def test_filtering_multi_source_mixing_plus_when_multi_ids(forecastdata_multi_ids):
    forecastdata_multi_ids.filter(sources=["mpr2 + ABC", "EFG"])

    assert all(forecastdata_multi_ids.forecasts["new_label"].unique() == ["ABC", "EFG"])
    assert all(forecastdata_multi_ids._main_table["new_label"].unique() == ["ABC", "EFG"])


# -----------------------
# Copy and Representation
# -----------------------
def test_copy_is_deep(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    fd_copy = fd.copy()
    fd._raw_forecasts.iloc[0, fd._raw_forecasts.columns.get_loc("value")] = 999
    assert fd_copy._raw_forecasts.iloc[0]["value"] != 999


def test_repr_returns_string(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    assert isinstance(repr(fd), str)


# -----------------------
# Edge Cases
# -----------------------
def test_add_empty_dataframe_does_not_break(sample_outturns):
    fd = ForecastData(outturns_data=sample_outturns)
    empty_df = pd.DataFrame(columns=fd.forecast_required_columns)
    fd.add_forecasts(empty_df)
    assert fd._raw_forecasts.empty


# -----------------------
# Property Tests for Required Columns
# -----------------------
def test_forecast_required_columns_returns_list():
    """Test that forecast_required_columns property returns a list."""
    fd = ForecastData()
    required_cols = fd.forecast_required_columns
    assert isinstance(required_cols, list)


def test_forecast_required_columns_not_empty():
    """Test that forecast_required_columns property returns a non-empty list."""
    fd = ForecastData()
    required_cols = fd.forecast_required_columns
    assert len(required_cols) > 0


def test_forecast_required_columns_all_strings():
    """Test that all elements in forecast_required_columns are strings."""
    fd = ForecastData()
    required_cols = fd.forecast_required_columns
    assert all(isinstance(col, str) for col in required_cols)


def test_forecast_required_columns_equals_constant():
    """Test that forecast_required_columns property returns FORECAST_REQUIRED_COLUMNS."""
    fd = ForecastData()
    required_cols = fd.forecast_required_columns
    assert required_cols == FORECAST_REQUIRED_COLUMNS


def test_forecast_required_columns_contains_expected_columns():
    """Test that forecast_required_columns contains expected essential columns."""
    fd = ForecastData()
    required_cols = fd.forecast_required_columns
    expected_columns = ["date", "variable", "vintage_date", "source", "frequency", "forecast_horizon", "value"]
    for col in expected_columns:
        assert col in required_cols, f"Expected column '{col}' not in forecast_required_columns"


def test_outturn_required_columns_returns_list():
    """Test that outturn_required_columns property returns a list."""
    fd = ForecastData()
    required_cols = fd.outturn_required_columns
    assert isinstance(required_cols, list)


def test_outturn_required_columns_not_empty():
    """Test that outturn_required_columns property returns a non-empty list."""
    fd = ForecastData()
    required_cols = fd.outturn_required_columns
    assert len(required_cols) > 0


def test_outturn_required_columns_all_strings():
    """Test that all elements in outturn_required_columns are strings."""
    fd = ForecastData()
    required_cols = fd.outturn_required_columns
    assert all(isinstance(col, str) for col in required_cols)


def test_outturn_required_columns_equals_constant():
    """Test that outturn_required_columns property returns OUTTURN_REQUIRED_COLUMNS."""
    fd = ForecastData()
    required_cols = fd.outturn_required_columns
    assert required_cols == OUTTURN_REQUIRED_COLUMNS


def test_forecast_and_outturn_columns_different():
    """Test that forecast and outturn required columns are different."""
    fd = ForecastData()
    forecast_cols = set(fd.forecast_required_columns)
    outturn_cols = set(fd.outturn_required_columns)
    assert forecast_cols != outturn_cols, "Forecast and outturn required columns should be different"


# -----------------------
# Private Helper Tests
# -----------------------
def test_validate_records_success(sample_forecasts):
    validated = _validate_records(sample_forecasts, forecast=True)
    assert isinstance(validated, pd.DataFrame)
    assert set(FORECAST_REQUIRED_COLUMNS).issubset(validated.columns)


def test_validate_records_missing_columns_raises():
    bad_df = pd.DataFrame({"date": ["2022-01-01"]})
    with pytest.raises(ValueError, match="Attempting to add data but the following columns are missing"):
        _validate_records(bad_df, forecast=True)


def test_check_duplicates_detects(sample_forecasts):
    sample_forecasts_1 = sample_forecasts.copy()
    sample_forecasts_1["value"] = sample_forecasts["value"] + 1.0
    with pytest.raises(ValueError, match="Duplicate records found with different values."):
        _check_duplicates(sample_forecasts, sample_forecasts_1)


# -----------------------
# Merge Tests
# -----------------------
def test_merge_non_overlapping_data(sample_outturns):
    """Test merging two ForecastData instances with non-overlapping forecast data."""
    forecast_df_1 = create_sample_forecasts()
    forecast_df_1["source"] = "source1"

    forecast_df_2 = create_sample_forecasts()
    forecast_df_2["source"] = "source2"

    fd1 = ForecastData(outturns_data=sample_outturns, forecasts_data=forecast_df_1)
    fd2 = ForecastData(outturns_data=sample_outturns, forecasts_data=forecast_df_2)
    fd1_original = fd1.copy()

    fd1.merge(fd2)

    assert len(fd1.forecasts) == len(fd1_original.forecasts) + len(fd2.forecasts)
    assert set(fd1.forecasts["source"].unique()) == {"source1", "source2"}


def test_merge_with_empty_instance(sample_outturns, sample_forecasts):
    """Test merging with an empty ForecastData instance."""
    fd1 = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    fd2 = ForecastData()

    merged = fd1.copy()
    merged.merge(fd2)

    assert len(merged.forecasts) == len(fd1.forecasts)
    assert len(merged.outturns) == len(fd1.outturns)


def test_merge_empty_with_populated(sample_outturns, sample_forecasts):
    """Test merging an empty instance with a populated one."""
    fd1 = ForecastData()
    fd2 = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)

    merged = fd1.copy()
    merged.merge(fd2)

    assert len(merged.forecasts) == len(fd2.forecasts)
    assert len(merged.outturns) == len(fd2.outturns)


def test_merge_with_duplicate_forecasts(sample_outturns, sample_forecasts):
    """Test that merge handles duplicate forecasts correctly (removes duplicates)."""
    fd1 = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    fd2 = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)

    with pytest.warns(UserWarning, match="Removed .* duplicate records with identical values"):
        merged = fd1.copy()
        merged.merge(fd2)

    # Should have same length as original since duplicates are removed
    assert len(merged.forecasts) == len(fd1.forecasts)


def test_merge_with_different_extra_ids():
    """Test merging ForecastData instances with different extra_ids."""
    outturns_df = create_sample_outturns()

    forecast_df_1 = create_sample_forecasts()
    forecast_df_1["label1"] = "A"

    forecast_df_2 = create_sample_forecasts()
    forecast_df_2["label2"] = "B"

    fd1 = ForecastData(outturns_data=outturns_df, forecasts_data=forecast_df_1, extra_ids=["label1"])
    fd2 = ForecastData(outturns_data=outturns_df, forecasts_data=forecast_df_2, extra_ids=["label2"])

    merged = fd1.copy()
    merged.merge(fd2)

    # Both id columns should exist in merged data
    assert "label1" in merged.forecasts.columns
    assert "label2" in merged.forecasts.columns
    assert set(merged.id_columns) == {"source", "label1", "label2"}


def test_merge_with_different_outturns():
    """Test merging with different outturn datasets."""
    outturns_df_1 = create_sample_outturns()
    outturns_df_1 = outturns_df_1[outturns_df_1["variable"] == "gdpkp"]

    outturns_df_2 = create_sample_outturns()
    outturns_df_2["variable"] = "cpiy"

    forecast_df_1 = create_sample_forecasts()
    forecast_df_1 = forecast_df_1[forecast_df_1["variable"] == "gdpkp"]

    forecast_df_2 = create_sample_forecasts()
    forecast_df_2["variable"] = "cpiy"

    fd1 = ForecastData(outturns_data=outturns_df_1, forecasts_data=forecast_df_1)
    fd2 = ForecastData(outturns_data=outturns_df_2, forecasts_data=forecast_df_2)

    merged = fd1.copy()
    merged.merge(fd2)

    # Should have outturns for both variables
    assert "gdpkp" in merged.outturns["variable"].values
    assert "cpiy" in merged.outturns["variable"].values


# -----------------------
# Date Forcing Tests
# -----------------------
def test_quarterly_dates_forced_to_quarter_end(sample_outturns):
    """Test that quarterly dates are forced to end of quarter."""
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-15", "2023-04-10"]),
            "vintage_date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "variable": "gdpkp",
            "source": "BVAR",
            "frequency": "Q",
            "forecast_horizon": [0, 1],
            "value": [100, 101],
        }
    )

    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)

    expected_dates = pd.to_datetime(["2023-03-31", "2023-06-30"])
    actual_dates = fd._raw_forecasts["date"].sort_values().reset_index(drop=True)

    pd.testing.assert_series_equal(actual_dates, pd.Series(expected_dates, name="date"))


def test_monthly_dates_forced_to_month_end(sample_outturns):
    """Test that monthly dates are forced to end of month."""
    sample_outturns = sample_outturns.copy()
    sample_outturns["frequency"] = "M"

    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-15", "2023-12-10"]),
            "vintage_date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "variable": "gdpkp",
            "source": "BVAR",
            "frequency": "M",
            "forecast_horizon": [0, 1],
            "value": [100, 101],
        }
    )

    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)

    expected_dates = pd.to_datetime(["2023-01-31", "2023-12-31"])
    actual_dates = fd._raw_forecasts["date"].sort_values().reset_index(drop=True)

    pd.testing.assert_series_equal(actual_dates, pd.Series(expected_dates, name="date"))


# test when using outturns not in levels
def test_outturns_not_in_levels(sample_outturns):
    """If the user supplies a metric column, add_forecasts() should use it."""
    sample_outturns["metric"] = "pop"

    df = _make_minimal_forecasts_df(include_metric=True, metric_value="pop")

    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=df)

    assert len(fd.outturns) == len(sample_outturns)
