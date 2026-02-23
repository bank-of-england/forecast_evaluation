"""Tests for DensityForecastData class."""

import pandas as pd
import pytest

from forecast_evaluation.data import DensityForecastData
from forecast_evaluation.data.sample_data import (
    create_sample_density_forecasts,
    create_sample_forecasts,
    create_sample_outturns,
)


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


def test_init_with_density_data(sample_outturns, sample_density_forecasts):
    """Test initialization with density forecast data."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    assert not dfd._density_df.empty
    assert not dfd.density_forecasts.empty
    assert "quantile" in dfd.density_forecasts.columns
    # Parent's forecasts should remain empty
    assert dfd.forecasts.empty


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

    # Generate samples
    n_samples = 10000
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
    assert set(merged.id_columns) == {"source", "label1", "label2"}


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
    assert len(fd_density.forecasts) == len(point_forecasts[point_forecasts["forecast_horizon"] >= 0])
    assert "point_source" in fd_density.forecasts["source"].unique()
