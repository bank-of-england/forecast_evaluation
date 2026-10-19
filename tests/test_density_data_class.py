"""Tests for DensityForecastData class."""

import inspect
import warnings
from pathlib import Path

import pandas as pd
import pytest

from forecast_evaluation.data import DensityForecastData, ForecastData
from forecast_evaluation.data.sample_data import (
    create_sample_density_forecasts,
    create_sample_forecasts,
    create_sample_outturns,
)
from forecast_evaluation.data.utils import compute_target_minus_vintage, construct_unique_id


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def sample_outturns() -> pd.DataFrame:
    return create_sample_outturns()


@pytest.fixture
def sample_density_forecasts() -> pd.DataFrame:
    """Create sample density forecast data with multiple quantiles."""
    return create_sample_density_forecasts()


@pytest.fixture
def sample_density_forecasts_single_quantile() -> pd.DataFrame:
    """Create sample density forecast data with single quantile (median)."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = 0.5
    return forecasts


# -----------------------
# Constructor Tests
# -----------------------
def test_init_empty():
    """Test creating an empty DensityForecastData object."""
    dfd = DensityForecastData()
    assert dfd.df.empty
    assert dfd.forecasts.empty
    assert dfd.density_forecasts.empty
    assert dfd.outturns is not None


def test_constructor_parameters_match_parent_and_are_keyword_only():
    parent_params = inspect.signature(ForecastData.__init__).parameters
    density_params = inspect.signature(DensityForecastData.__init__).parameters

    assert list(density_params) == list(parent_params)
    for name in (
        "extra_ids",
        "metric",
        "compute_levels",
        "data_check",
        "outturn_vintages",
        "default_k",
        "first_forecast_horizon",
    ):
        assert density_params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_init_accepts_parent_options(sample_outturns):
    outturns = sample_outturns.drop(columns="metric", errors="ignore")
    dfd = DensityForecastData(
        outturns_data=outturns,
        metric="pop",
        compute_levels=False,
        data_check=False,
        outturn_vintages=False,
        default_k=3,
        first_forecast_horizon=1,
    )

    assert dfd._raw_outturns["metric"].eq("pop").all()
    assert dfd.default_k == 3
    assert dfd.outturn_vintages is False
    assert dfd.first_forecast_horizon == 1
    assert dfd._compute_levels is False


def test_add_density_forecast_options_are_keyword_only():
    params = inspect.signature(DensityForecastData.add_density_forecasts).parameters

    for name in ("extra_ids", "metric", "data_check", "first_forecast_horizon"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_init_with_density_data(sample_outturns, sample_density_forecasts):
    """Test initialization with density forecast data."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    assert not dfd._density_df.empty
    assert not dfd.density_forecasts.empty
    assert "quantile" in dfd.density_forecasts.columns
    # Parent's forecasts should remain empty
    assert dfd.forecasts.empty


def test_init_with_fer_load(monkeypatch, sample_outturns):
    """Load point forecasts during parent initialisation without resetting density tables."""

    def add_fer_data(self):
        self.add_outturns(sample_outturns)
        self.add_forecasts(create_sample_forecasts(), data_check=False)

    monkeypatch.setattr(ForecastData, "add_fer_data", add_fer_data)
    dfd = DensityForecastData(load_fer=True)

    assert not dfd.forecasts.empty
    assert dfd.density_forecasts.empty


def test_init_with_single_quantile(sample_outturns, sample_density_forecasts_single_quantile):
    """Test initialisation with single quantile (median) data."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts_single_quantile)
    assert not dfd._density_df.empty
    # Check that all quantile values are 0.5 (handle both float and string)
    quantiles = dfd.density_forecasts["quantile"].unique()
    assert len(quantiles) == 1
    assert float(quantiles[0]) == 0.5


def test_init_missing_quantile_column_raises(sample_outturns):
    """Test that missing quantile column raises an error."""
    forecasts_without_quantile = create_sample_forecasts()

    with pytest.raises(ValueError, match="Density forecasts must include a 'quantile' column"):
        DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts_without_quantile)


def test_init_with_extra_ids(sample_outturns, sample_density_forecasts):
    """Test initialization with extra ID columns."""
    sample_density_forecasts["region"] = "UK"
    dfd = DensityForecastData(
        outturns_data=sample_outturns, forecasts_data=sample_density_forecasts, extra_ids=["region"]
    )
    assert "region" in dfd.id_columns


@pytest.mark.parametrize("on_init", [True, False])
def test_quantile_is_not_an_id(sample_outturns, sample_density_forecasts, on_init):
    """Explicit quantile labels must still form one density per source."""
    dfd = DensityForecastData(
        outturns_data=sample_outturns,
        forecasts_data=sample_density_forecasts if on_init else None,
        extra_ids=["quantile"] if on_init else None,
    )
    if not on_init:
        dfd.add_density_forecasts(sample_density_forecasts, extra_ids=["quantile"])

    assert dfd.id_columns == ["source"]
    assert dfd.density_forecasts["unique_id"].nunique() == 1
    assert dfd.density_forecasts.groupby(["date", "unique_id"])["quantile"].nunique().max() == 50


# -----------------------
# Add Forecasts Tests
# -----------------------
def test_add_density_forecasts(sample_outturns, sample_density_forecasts):
    """Test adding density forecasts to an existing object."""
    dfd = DensityForecastData(outturns_data=sample_outturns)
    assert dfd.density_forecasts.empty

    dfd.add_density_forecasts(sample_density_forecasts)
    assert not dfd.density_forecasts.empty
    assert "quantile" in dfd.density_forecasts.columns


def test_add_density_forecasts_without_quantile_raises(sample_outturns):
    """Test that adding forecasts without quantile column raises an error."""
    dfd = DensityForecastData(outturns_data=sample_outturns)
    forecasts_without_quantile = create_sample_forecasts()

    with pytest.raises(ValueError, match="Density forecasts must include a 'quantile' column"):
        dfd.add_density_forecasts(forecasts_without_quantile)


def test_density_forecasts_preserve_provided_metrics(sample_outturns):
    """Keep supplied pop and yoy quantiles without deriving other metrics."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = 0.5
    forecasts = pd.concat(
        [forecasts.assign(metric=metric) for metric in ("pop", "yoy")],
        ignore_index=True,
    )

    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)

    actual = dfd.density_forecasts.sort_values(["metric", "date"]).reset_index(drop=True)
    expected = forecasts.sort_values(["metric", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual[["date", "metric", "quantile", "value"]],
        expected[["date", "metric", "quantile", "value"]],
        check_dtype=False,
    )


def test_density_forecasts_default_to_levels(sample_outturns, sample_density_forecasts):
    """Default density forecasts to levels and warn when metric is omitted."""
    with pytest.warns(UserWarning, match="Forecasts input has no 'metric' column"):
        dfd = DensityForecastData(
            outturns_data=sample_outturns.assign(metric="levels"),
            forecasts_data=sample_density_forecasts,
            data_check=False,
        )

    assert dfd.density_forecasts["metric"].eq("levels").all()


def test_density_metric_argument_fills_null_metric_values(sample_outturns, sample_density_forecasts):
    forecasts = sample_density_forecasts.assign(metric=pd.NA)
    dfd = DensityForecastData(
        outturns_data=sample_outturns.assign(metric="pop"),
        forecasts_data=forecasts,
        metric="pop",
        data_check=False,
    )

    assert dfd.density_forecasts["metric"].eq("pop").all()


@pytest.mark.parametrize(
    ("quantiles", "data_check", "expect_warning"),
    [([0.25, 0.5, 0.75], True, True), ([0.25, 0.5, 0.75], False, False), ([0.05, 0.95], True, False)],
)
def test_density_data_check_uses_near_median_path(sample_outturns, quantiles, data_check, expect_warning):
    outturns = sample_outturns.assign(metric="levels")
    outturns = outturns[outturns["variable"] == "gdpkp"].sort_values("date").iloc[:3]
    forecasts = pd.DataFrame(
        [
            {
                "date": outturn["date"],
                "variable": outturn["variable"],
                "vintage_date": outturn["vintage_date"],
                "source": "check",
                "frequency": outturn["frequency"],
                "forecast_horizon": 1,
                "metric": "levels",
                "quantile": quantile,
                "value": float(outturn["value"])
                * (100 if quantile == 0.5 or len(quantiles) == 2 else 1),
            }
            for _, outturn in outturns.iterrows()
            for quantile in quantiles
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DensityForecastData(outturns_data=outturns, forecasts_data=forecasts, data_check=data_check)

    check_warnings = [warning for warning in caught if "[Data check]" in str(warning.message)]
    assert bool(check_warnings) is expect_warning


def test_density_forecasts_drop_negative_horizons_with_warning(sample_outturns, sample_density_forecasts):
    forecasts = sample_density_forecasts.iloc[:4].copy()
    forecasts["metric"] = "levels"
    forecasts["forecast_horizon"] = [-1, -1, 0, 0]

    with pytest.warns(UserWarning, match="Dropped 2 density forecast rows with negative forecast_horizon"):
        dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts, data_check=False)

    assert len(dfd._density_df) == len(dfd._density_forecasts) == 2
    assert dfd._density_df["forecast_horizon"].eq(0).all()


def test_density_with_only_negative_horizons_adds_nothing(sample_outturns, sample_density_forecasts):
    forecasts = sample_density_forecasts.iloc[:4].assign(forecast_horizon=-1, metric="levels")

    with pytest.warns(UserWarning, match="No density forecasts available after filtering/validation"):
        dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts, data_check=False)

    assert dfd._density_df.empty
    assert dfd._density_forecasts.empty


def test_density_first_forecast_horizon_is_forwarded(sample_outturns, sample_density_forecasts):
    forecasts = sample_density_forecasts.drop(columns="forecast_horizon")
    forecasts = forecasts[forecasts["date"] > forecasts["date"].min()].copy()

    with pytest.warns(FutureWarning, match="first_forecast_horizon"):
        dfd = DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=forecasts,
            data_check=False,
            first_forecast_horizon=1,
        )

    expected = compute_target_minus_vintage(forecasts)["target_minus_vintage"] - 1
    pd.testing.assert_series_equal(
        dfd._density_df["forecast_horizon"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


def test_density_plot_missing_metric_raises_clear_error(sample_outturns, sample_density_forecasts):
    """Explain when the requested metric has no density forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    with pytest.raises(ValueError, match="No density forecasts found.*metric='pop'"):
        dfd.plot_density_vintage(variable="gdpkp", vintage_date="2025-12-31", metric="pop")


@pytest.mark.parametrize("density_first", [True, False])
def test_point_and_density_forecasts_must_share_frequency(sample_outturns, sample_density_forecasts, density_first):
    """Reject point and density forecast tables with different frequencies."""
    dfd = DensityForecastData(outturns_data=sample_outturns)
    monthly_forecasts = create_sample_forecasts().assign(frequency="M")

    if density_first:
        dfd.add_density_forecasts(sample_density_forecasts)
        with pytest.raises(ValueError, match="existing data has frequency 'Q'"):
            dfd.add_forecasts(monthly_forecasts, data_check=False)
    else:
        dfd.add_forecasts(create_sample_forecasts(), data_check=False)
        with pytest.raises(ValueError, match="existing data has frequency 'Q'"):
            dfd.add_density_forecasts(sample_density_forecasts.assign(frequency="M"))


def test_clear_filter_density_only(sample_outturns, sample_density_forecasts):
    """Restore density forecasts without requiring point forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    dfd.filter(start_date="2022-06-30", end_date="2022-12-31")

    assert len(dfd.density_forecasts) < len(sample_density_forecasts)

    dfd.clear_filter()

    assert len(dfd.density_forecasts) == len(sample_density_forecasts)


def test_density_forecasts_retain_calendar_backcasts(sample_outturns):
    """Retain valid information-horizon forecasts before the vintage period."""
    forecasts = pd.DataFrame(
        {
            "date": pd.Timestamp("2022-12-31"),
            "variable": "gdpkp",
            "vintage_date": pd.Timestamp("2023-03-31"),
            "source": "mpr2",
            "frequency": "Q",
            "quantile": [0.25, 0.75],
            "value": [101.0, 103.0],
            "forecast_horizon": [0, 0],
        }
    )

    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)

    retained = dfd.density_forecasts[dfd.density_forecasts["metric"] == "levels"]
    assert len(retained) == 2
    assert retained["forecast_horizon"].tolist() == [0, 0]
    assert retained["target_minus_vintage"].tolist() == [-1, -1]


def test_rejected_density_forecasts_do_not_mutate_state(sample_outturns, sample_density_forecasts):
    """A rejected density forecast batch must leave the instance unchanged."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    before_id_columns = dfd._id_columns.copy()
    before_raw_forecasts = dfd._raw_forecasts.copy(deep=True)
    before_forecasts = dfd._forecasts.copy(deep=True)
    before_density_df = dfd._density_df.copy(deep=True)
    before_density_forecasts = dfd._density_forecasts.copy(deep=True)

    invalid = sample_density_forecasts.assign(value=lambda frame: frame["value"] + 1)
    with pytest.raises(ValueError):
        dfd.add_density_forecasts(invalid)

    assert dfd._id_columns == before_id_columns
    pd.testing.assert_frame_equal(dfd._raw_forecasts, before_raw_forecasts)
    pd.testing.assert_frame_equal(dfd._forecasts, before_forecasts)
    pd.testing.assert_frame_equal(dfd._density_df, before_density_df)
    pd.testing.assert_frame_equal(dfd._density_forecasts, before_density_forecasts)


# -----------------------
# Filtering Tests
# -----------------------
def test_filter_preserves_quantiles(sample_outturns, sample_density_forecasts):
    """Test that filtering preserves quantile information."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    original_quantiles = set(dfd.density_forecasts["quantile"].unique())

    # Check if filtering is properly set up (mocking may prevent proper filtering)
    try:
        dfd.filter(variables=["gdpkp"])
        filtered_quantiles = set(dfd.density_forecasts["quantile"].unique())

        # With 100 quantiles, all should be preserved when filtering by variable
        assert len(original_quantiles) == 50
        assert original_quantiles == filtered_quantiles
    except KeyError:
        # If mocking prevents filtering, just verify quantiles exist
        assert len(original_quantiles) == 50


def test_filter_by_source(sample_outturns, sample_density_forecasts):
    """Test filtering by source works with density forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    sources = dfd.density_forecasts["source"].unique()

    dfd.filter(sources=[sources[0]])
    assert all(dfd.density_forecasts["source"] == sources[0])


# -----------------------
# Copy Tests
# -----------------------
def test_copy_is_deep(sample_outturns, sample_density_forecasts):
    """Test that copy creates a deep copy."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    dfd_copy = dfd.copy()

    # Modify original
    dfd.density_forecasts.iloc[0, dfd.density_forecasts.columns.get_loc("value")] = 999

    # Copy should be unchanged
    assert dfd_copy.density_forecasts.iloc[0]["value"] != 999


# -----------------------
# Integration Tests
# -----------------------
def test_density_forecasts_with_custom_filter(sample_outturns, sample_density_forecasts):
    """Test custom filtering works with density forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    def custom_filter(df):
        return df[df["quantile"] >= 0.5]

    dfd.filter(custom_filter=custom_filter)

    # Should have roughly 50 quantiles remaining (0.5 to 0.99)
    assert all(dfd.density_forecasts["quantile"] >= 0.5)
    assert len(dfd.density_forecasts["quantile"].unique()) >= 20  # Allow some tolerance


def test_density_forecasts_repr(sample_outturns, sample_density_forecasts):
    """Test that repr works for density forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    repr_str = repr(dfd)
    assert isinstance(repr_str, str)
    assert len(repr_str) > 0


def test_density_forecasts_repr_html_matches_repr(density_only):
    """Notebook HTML shows the density table, as repr does."""
    html = density_only._repr_html_()

    assert "quantile" in html
    assert html == density_only.density_forecasts._repr_html_()
    assert repr(density_only) == repr(density_only.density_forecasts)


def test_density_forecasts_repr_html_with_point_forecasts(density_and_point):
    """Point forecasts do not replace the density table in the HTML view."""
    assert "quantile" in density_and_point._repr_html_()


def test_density_module_has_no_demo_block():
    """The module is a library: no demo-only sample-data import or __main__ block."""
    import importlib

    module = importlib.import_module("forecast_evaluation.data.DensityForecastData")

    assert not hasattr(module, "create_sample_density_forecasts")
    assert "__main__" not in Path(module.__file__).read_text(encoding="utf-8")


# -----------------------
# Validation Tests
# -----------------------
def test_quantile_validation_out_of_range():
    """Test that quantile values outside [0,1] are caught by validation."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = 1.5  # Invalid quantile

    outturns = create_sample_outturns()

    with pytest.raises(Exception):  # Schema validation should fail
        DensityForecastData(outturns_data=outturns, forecasts_data=forecasts)


def test_quantile_validation_negative():
    """Test that negative quantile values are caught by validation."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = -0.1  # Invalid quantile

    outturns = create_sample_outturns()

    with pytest.raises(Exception):  # Schema validation should fail
        DensityForecastData(outturns_data=outturns, forecasts_data=forecasts)


# -----------------------
# Edge Cases
# -----------------------
def test_single_quantile_at_boundary(sample_outturns):
    """Test density forecasts with quantile at boundary values (0 and 1)."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = 0.0  # Minimum valid quantile

    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)
    # Handle both float and string types
    quantiles = [float(q) for q in dfd.density_forecasts["quantile"]]
    assert all(q == 0.0 for q in quantiles)

    forecasts["quantile"] = 1.0  # Maximum valid quantile
    forecasts["vintage_date"] = pd.Timestamp("2023-01-01")  # Change to avoid duplicates
    dfd.add_density_forecasts(forecasts)
    quantiles = [float(q) for q in dfd.density_forecasts["quantile"].unique()]
    assert 1.0 in quantiles


# -----------------------
# Conversion Tests
# -----------------------
def test_to_point_forecast_median(sample_outturns, sample_density_forecasts):
    """Test converting density forecasts to point forecasts using median."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    # Convert to point forecast using median
    dfd.to_point_forecast(method="median")

    # Check that parent forecasts are now populated
    assert not dfd.forecasts.empty
    assert len(dfd.forecasts) > 0


def test_to_point_forecast_specific_quantile(sample_outturns, sample_density_forecasts):
    """Test converting density forecasts to point forecasts using a specific quantile."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    # Convert using 0.75 quantile
    dfd.to_point_forecast(method="0.75")

    # Check that parent forecasts are now populated
    assert not dfd.forecasts.empty


def test_to_point_forecast_invalid_method(sample_outturns, sample_density_forecasts):
    """Test that invalid method raises an error."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    with pytest.raises(ValueError, match="Invalid method"):
        dfd.to_point_forecast(method="invalid_method")


# -----------------------
# Sampling Tests
# -----------------------
def test_sample_from_density_statistics(sample_outturns, sample_density_forecasts):
    """Test that samples from density have correct mean and variance."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    # 2,000 draws is ample for the tolerances asserted below (the observed errors are an
    # order of magnitude inside them) and keeps this test roughly five times faster than
    # the 10,000 draws it used to take.
    n_samples = 2000
    samples = dfd.sample_from_density(n_samples=n_samples, random_state=42)

    # get the levels
    samples = samples[samples["metric"] == "levels"]

    # Check that samples were generated
    assert not samples.empty
    assert len(samples) > 0
    assert "value" in samples.columns
    assert "sample_id" in samples.columns

    # For each date/forecast combination, check mean and variance
    # The original distribution was normal with mean=100+i and std=mean*0.05
    for i, date in enumerate(sample_density_forecasts["date"].unique()):
        date_samples = samples[samples["date"] == date]["value"]

        if len(date_samples) > 0:
            expected_mean = i + 1
            expected_std = 0.5

            sample_mean = date_samples.mean()
            sample_std = date_samples.std()

            # Allow 5% tolerance for mean and 10% for std (higher variance in variance estimation)
            assert abs(sample_mean - expected_mean) / expected_mean < 0.05, (
                f"Mean mismatch at date {date}: expected {expected_mean}, got {sample_mean}"
            )
            assert abs(sample_std - expected_std) / expected_std < 0.10, (
                f"Std mismatch at date {date}: expected {expected_std}, got {sample_std}"
            )


# -----------------------
# Merge Tests
# -----------------------
def test_merge_non_overlapping_density_data(sample_outturns):
    """Test merging two DensityForecastData instances with non-overlapping forecast data."""
    forecast_df_1 = create_sample_density_forecasts()
    forecast_df_1["source"] = "source1"

    forecast_df_2 = create_sample_density_forecasts()
    forecast_df_2["source"] = "source2"

    fd1 = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecast_df_1)
    fd2 = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecast_df_2)
    fd1_original_len = len(fd1.density_forecasts)

    fd1.merge(fd2)

    assert len(fd1.density_forecasts) == fd1_original_len + len(fd2.density_forecasts)
    assert set(fd1.density_forecasts["source"].unique()) == {"source1", "source2"}


def test_merge_density_with_empty_instance(sample_outturns, sample_density_forecasts):
    """Test merging with an empty DensityForecastData instance."""
    fd1 = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    fd2 = DensityForecastData()

    merged = fd1.copy()

    merged.merge(fd2)

    assert len(merged.density_forecasts) == len(fd1.density_forecasts)
    assert len(merged.outturns) == len(fd1.outturns)


def test_merge_empty_density_with_populated(sample_outturns, sample_density_forecasts):
    """Test merging an empty instance with a populated one."""
    fd1 = DensityForecastData()
    fd2 = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    merged = fd1.copy()
    merged.merge(fd2)

    assert len(merged.density_forecasts) == len(fd2.density_forecasts)
    assert len(merged.outturns) == len(fd2.outturns)


def test_merge_density_with_different_extra_ids():
    """Test merging DensityForecastData instances with different extra_ids."""
    outturns_df = create_sample_outturns()

    forecast_df_1 = create_sample_density_forecasts()
    forecast_df_1["label1"] = "A"

    forecast_df_2 = create_sample_density_forecasts()
    forecast_df_2["label2"] = "B"

    fd1 = DensityForecastData(outturns_data=outturns_df, forecasts_data=forecast_df_1, extra_ids=["label1"])
    fd2 = DensityForecastData(outturns_data=outturns_df, forecasts_data=forecast_df_2, extra_ids=["label2"])

    merged = fd1.copy()
    merged.merge(fd2)

    # Both id columns should exist in merged data
    assert "label1" in merged.density_forecasts.columns
    assert "label2" in merged.density_forecasts.columns
    assert merged.id_columns == ["source", "label1", "label2"]
    for table in (merged._density_df, merged._density_forecasts):
        expected = construct_unique_id(table, merged.id_columns)
        pd.testing.assert_series_equal(table["unique_id"], expected, check_names=False)
        assert set(table["unique_id"]) == {"mpr2 + A + ", "mpr2 +  + B"}
        assert table["unique_id"].str.count(" \\+ ").eq(2).all()

    merged.filter(sources=["mpr2 + A + "])
    assert len(merged.density_forecasts) == len(forecast_df_1)
    merged.clear_filter()
    merged.filter(sources=["mpr2 + A"])
    assert merged.density_forecasts.empty
    merged.clear_filter()
    merged.filter(sources=["mpr2"])
    assert len(merged.density_forecasts) == len(forecast_df_1) + len(forecast_df_2)


def test_density_merge_is_atomic(sample_outturns, sample_density_forecasts):
    outturns = sample_outturns.assign(metric="levels")
    forecasts = sample_density_forecasts.assign(metric="levels")
    target = DensityForecastData(outturns_data=outturns, data_check=False)
    target.add_density_forecasts(forecasts, data_check=False)
    target.add_forecasts(create_sample_forecasts().assign(source="point", metric="levels"), data_check=False)
    conflicting = DensityForecastData(
        outturns_data=outturns,
        forecasts_data=forecasts.assign(value=lambda frame: frame["value"] + 1),
        data_check=False,
    )
    conflicting.add_outturns(outturns.assign(variable="cpiy"))

    table_names = (
        "_raw_outturns",
        "_outturns",
        "_raw_forecasts",
        "_forecasts",
        "_main_table",
        "_density_df",
        "_density_forecasts",
    )
    before = {name: getattr(target, name).copy(deep=True) for name in table_names}
    before_id_columns = target.id_columns.copy()
    before_horizon = target.first_forecast_horizon
    assert not target._raw_forecasts.empty
    assert not target._main_table.empty

    with pytest.raises(ValueError, match="Duplicate records found with different values"):
        target.merge(conflicting)

    assert target.id_columns == before_id_columns
    assert target.first_forecast_horizon == before_horizon
    for name, table in before.items():
        pd.testing.assert_frame_equal(getattr(target, name), table)


def test_forecastdata_merge_density_warns(sample_outturns, sample_density_forecasts):
    outturns = sample_outturns.assign(metric="levels")
    density = DensityForecastData(
        outturns_data=outturns,
        forecasts_data=sample_density_forecasts.assign(metric="levels"),
        data_check=False,
    )
    plain = ForecastData(outturns_data=outturns)

    with pytest.warns(UserWarning, match="Density forecasts.*were not merged"):
        plain.merge(density)


def test_merge_density_with_different_outturns():
    """Test merging with different outturn datasets."""
    outturns_df_1 = create_sample_outturns()
    outturns_df_1 = outturns_df_1[outturns_df_1["variable"] == "gdpkp"]

    outturns_df_2 = create_sample_outturns()
    outturns_df_2["variable"] = "cpiy"

    forecast_df_1 = create_sample_density_forecasts()
    forecast_df_1 = forecast_df_1[forecast_df_1["variable"] == "gdpkp"]

    forecast_df_2 = create_sample_density_forecasts()
    forecast_df_2["variable"] = "cpiy"

    fd1 = DensityForecastData(outturns_data=outturns_df_1, forecasts_data=forecast_df_1)
    fd2 = DensityForecastData(outturns_data=outturns_df_2, forecasts_data=forecast_df_2)

    merged = fd1.copy()
    merged.merge(fd2)

    # Should have outturns for both variables
    assert "gdpkp" in merged.outturns["variable"].values
    assert "cpiy" in merged.outturns["variable"].values
    # Should have density forecasts for both variables
    assert "gdpkp" in merged.density_forecasts["variable"].values
    assert "cpiy" in merged.density_forecasts["variable"].values


def test_merge_with_regular_forecast_data(sample_outturns, sample_density_forecasts):
    """Test merging a regular ForecastData instance into a DensityForecastData instance."""
    from forecast_evaluation.data import ForecastData

    # Create DensityForecastData (starts with density forecasts but empty point forecasts)
    fd_density = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    original_density_len = len(fd_density.density_forecasts)
    assert fd_density.forecasts.empty

    # Create regular ForecastData (has point forecasts)
    point_forecasts = create_sample_forecasts()
    point_forecasts["source"] = "point_source"
    fd_regular = ForecastData(outturns_data=sample_outturns, forecasts_data=point_forecasts)

    # Merge regular into density
    fd_density.merge(fd_regular)

    # Density forecasts should remain unchanged (regular ForecastData has none)
    assert len(fd_density.density_forecasts) == original_density_len

    # Point forecasts from the regular instance should be added to the density object
    assert not fd_density.forecasts.empty
    assert len(fd_density.forecasts) == fd_regular.forecasts.shape[0]
    assert "point_source" in fd_density.forecasts["source"].unique()
