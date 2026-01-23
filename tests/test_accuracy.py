"""
Tests for accuracy analysis functions.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.tests.accuracy import (
    compare_to_benchmark,
    compute_accuracy_statistics,
    create_comparison_table,
)


@pytest.fixture
def sample_forecast_data():
    """
    Create a ForecastData object with known forecast errors for testing accuracy calculations.

    Creates forecasts with controlled errors:
    - Source 'model_a': errors = [1, 2, 3, 4] -> RMSE = sqrt(7.5) ≈ 2.739, MAE = 2.5, RMedSE = sqrt(6.5) ≈ 2.550
    - Source 'model_b': errors = [2, 4, 6, 8] -> RMSE = 5.477, MAE = 5.0, RMedSE = 5.099
    - Source 'benchmark': errors = [3, 3, 3, 3] -> RMSE = 3.0, MAE = 3.0, RMedSE = 3.0
    """
    # Create forecasts with known errors
    dates = pd.date_range(start="2022-01-01", periods=4, freq="QE")

    forecasts_data = []
    outturns_data = []

    # Model A forecasts (errors: 1, 2, 3, 4)
    for i, date in enumerate(dates):
        forecasts_data.append(
            {
                "date": date,
                "variable": "test",
                "vintage_date": date,
                "source": "model_a",
                "frequency": "Q",
                "forecast_horizon": 0,
                "value": 100 + i + 1,  # forecast values: 101, 102, 103, 104
            }
        )

    # Model B forecasts (errors: 2, 4, 6, 8)
    for i, date in enumerate(dates):
        forecasts_data.append(
            {
                "date": date,
                "variable": "test",
                "vintage_date": date,
                "source": "model_b",
                "frequency": "Q",
                "forecast_horizon": 0,
                "value": 100 + 2 * (i + 1),  # forecast values: 102, 104, 106, 108
            }
        )

    # Benchmark forecasts (errors: 3, 3, 3, 3)
    for i, date in enumerate(dates):
        forecasts_data.append(
            {
                "date": date,
                "variable": "test",
                "vintage_date": date,
                "source": "benchmark",
                "frequency": "Q",
                "forecast_horizon": 0,
                "value": 103,  # constant forecast value: 103
            }
        )

    # Outturns (actual values: 100, 100, 100, 100)
    for i, date in enumerate(dates):
        outturns_data.append(
            {
                "date": date,
                "variable": "test",
                "vintage_date": date + pd.offsets.QuarterEnd(1),
                "frequency": "Q",
                "forecast_horizon": -1,
                "value": 100,
            }
        )

    forecasts_df = pd.DataFrame(forecasts_data)
    outturns_df = pd.DataFrame(outturns_data)

    # Create ForecastData object
    forecast_data = ForecastData(outturns_data=outturns_df, forecasts_data=forecasts_df)

    return forecast_data


@pytest.fixture
def sample_multi_horizon_data():
    """
    Create ForecastData with multiple forecast horizons for testing.
    """
    dates = pd.date_range(start="2022-01-01", periods=3, freq="QE")
    horizons = [0, 1, 2]

    forecasts_data = []
    outturns_data = []

    # Forecasts
    for date in dates:
        for horizon in horizons:
            # Model A
            forecasts_data.append(
                {
                    "date": (date + pd.offsets.QuarterEnd(horizon)),
                    "variable": "gdpkp",
                    "vintage_date": date,
                    "source": "model_a",
                    "frequency": "Q",
                    "forecast_horizon": horizon,
                    "value": 100 + horizon,  # Errors will be horizon-dependent
                }
            )

            # Benchmark
            forecasts_data.append(
                {
                    "date": (date + pd.offsets.QuarterEnd(horizon)),
                    "variable": "gdpkp",
                    "vintage_date": date,
                    "source": "benchmark",
                    "frequency": "Q",
                    "forecast_horizon": horizon,
                    "value": 102,
                }
            )

    # Outturns
    outturns_dates = pd.date_range(start="2022-01-01", periods=5, freq="QE")
    for date in outturns_dates:
        outturns_data.append(
            {
                "date": date,
                "variable": "gdpkp",
                "vintage_date": date + pd.offsets.QuarterEnd(1),
                "frequency": "Q",
                "forecast_horizon": -1,
                "value": 100,
            }
        )

    forecasts_df = pd.DataFrame(forecasts_data)
    outturns_df = pd.DataFrame(outturns_data)

    # Create ForecastData object
    forecast_data = ForecastData(outturns_data=outturns_df, forecasts_data=forecasts_df)

    return forecast_data


def test_compute_accuracy_statistics_basic(sample_forecast_data):
    """
    Test that compute_accuracy_statistics correctly calculates RMSE, MAE, and RMedSE.
    """
    result = compute_accuracy_statistics(data=sample_forecast_data, k=0)

    assert result is not None, "Result should not be None"

    df = result.to_df()

    # Check that all sources are present
    assert set(df["unique_id"].unique()) == {"model_a", "model_b", "benchmark"}

    # Check model_a statistics (errors: 1, 2, 3, 4)
    model_a = df[df["unique_id"] == "model_a"].iloc[0]
    expected_rmse_a = np.sqrt(np.mean([1**2, 2**2, 3**2, 4**2]))  # sqrt(7.5) ≈ 2.739
    expected_mae_a = np.mean([1, 2, 3, 4])  # 2.5
    expected_rmedse_a = np.sqrt(np.median([1**2, 2**2, 3**2, 4**2]))  # sqrt(6.5) ≈ 2.550

    assert np.isclose(model_a["rmse"], expected_rmse_a, rtol=1e-3), (
        f"Model A RMSE: expected {expected_rmse_a}, got {model_a['rmse']}"
    )
    assert np.isclose(model_a["mean_abs_error"], expected_mae_a, rtol=1e-3), (
        f"Model A MAE: expected {expected_mae_a}, got {model_a['mean_abs_error']}"
    )
    assert np.isclose(model_a["rmedse"], expected_rmedse_a, rtol=1e-3), (
        f"Model A RMedSE: expected {expected_rmedse_a}, got {model_a['rmedse']}"
    )

    # Check model_b statistics (errors: 2, 4, 6, 8)
    model_b = df[df["unique_id"] == "model_b"].iloc[0]
    expected_rmse_b = np.sqrt(np.mean([2**2, 4**2, 6**2, 8**2]))  # sqrt(30) ≈ 5.477
    expected_mae_b = np.mean([2, 4, 6, 8])  # 5.0
    expected_rmedse_b = np.sqrt(np.median([2**2, 4**2, 6**2, 8**2]))  # sqrt(26) ≈ 5.099

    assert np.isclose(model_b["rmse"], expected_rmse_b, rtol=1e-3), (
        f"Model B RMSE: expected {expected_rmse_b}, got {model_b['rmse']}"
    )
    assert np.isclose(model_b["mean_abs_error"], expected_mae_b, rtol=1e-3), (
        f"Model B MAE: expected {expected_mae_b}, got {model_b['mean_abs_error']}"
    )
    assert np.isclose(model_b["rmedse"], expected_rmedse_b, rtol=1e-3), (
        f"Model B RMedSE: expected {expected_rmedse_b}, got {model_b['rmedse']}"
    )

    # Check benchmark statistics (errors: 3, 3, 3, 3)
    benchmark = df[df["unique_id"] == "benchmark"].iloc[0]
    expected_rmse_bench = 3.0
    expected_mae_bench = 3.0
    expected_rmedse_bench = 3.0

    assert np.isclose(benchmark["rmse"], expected_rmse_bench, rtol=1e-3), (
        f"Benchmark RMSE: expected {expected_rmse_bench}, got {benchmark['rmse']}"
    )
    assert np.isclose(benchmark["mean_abs_error"], expected_mae_bench, rtol=1e-3), (
        f"Benchmark MAE: expected {expected_mae_bench}, got {benchmark['mean_abs_error']}"
    )
    assert np.isclose(benchmark["rmedse"], expected_rmedse_bench, rtol=1e-3), (
        f"Benchmark RMedSE: expected {expected_rmedse_bench}, got {benchmark['rmedse']}"
    )

    # Check number of observations
    assert all(df["n_observations"] == 4), "All sources should have 4 observations"


def test_compute_accuracy_statistics_filtering(sample_forecast_data):
    """
    Test that filtering by source and variable works correctly.
    """
    # Test source filtering
    result = compute_accuracy_statistics(data=sample_forecast_data, source="model_a", k=0)

    df = result.to_df()
    assert len(df["unique_id"].unique()) == 1, "Should only have one source"
    assert df["unique_id"].unique()[0] == "model_a", "Should only have model_a"

    # Test multiple source filtering
    result = compute_accuracy_statistics(data=sample_forecast_data, source=["model_a", "model_b"], k=0)

    df = result.to_df()
    assert len(df["unique_id"].unique()) == 2, "Should have two sources"
    assert set(df["unique_id"].unique()) == {"model_a", "model_b"}


def test_compute_accuracy_statistics_metadata(sample_forecast_data):
    """
    Test that metadata is correctly stored in the TestResult.
    """
    result = compute_accuracy_statistics(data=sample_forecast_data, source="model_a", variable="gdpkp", k=0)

    assert result._metadata["test_name"] == "compute_accuracy_statistics"
    assert result._metadata["parameters"]["k"] == 0
    assert result._metadata["parameters"]["same_date_range"] is True
    assert result._metadata["filters"]["unique_id"] == "model_a"
    assert result._metadata["filters"]["variable"] == "gdpkp"
    assert "date_range" in result._metadata


def test_compare_to_benchmark_basic(sample_forecast_data):
    """
    Test that compare_to_benchmark correctly calculates ratios.
    """
    # First get accuracy statistics
    result = compute_accuracy_statistics(data=sample_forecast_data, k=0)

    df = result.to_df()

    # Compare to benchmark using RMSE
    comparison_df = compare_to_benchmark(df, benchmark_model="benchmark", statistic="rmse")

    # Check that the benchmark column exists
    assert "rmse_benchmark" in comparison_df.columns
    assert "rmse_to_benchmark" in comparison_df.columns

    # Verify benchmark RMSE is 3.0 for all rows
    assert all(np.isclose(comparison_df["rmse_benchmark"], 3.0, rtol=1e-3))

    # Check model_a ratio (RMSE ≈ 2.739, benchmark = 3.0)
    model_a_ratio = comparison_df[comparison_df["unique_id"] == "model_a"]["rmse_to_benchmark"].iloc[0]
    expected_ratio_a = np.sqrt(7.5) / 3.0  # ≈ 0.913
    assert np.isclose(model_a_ratio, expected_ratio_a, rtol=1e-3), (
        f"Model A ratio: expected {expected_ratio_a}, got {model_a_ratio}"
    )

    # Check model_b ratio (RMSE ≈ 5.477, benchmark = 3.0)
    model_b_ratio = comparison_df[comparison_df["unique_id"] == "model_b"]["rmse_to_benchmark"].iloc[0]
    expected_ratio_b = np.sqrt(30) / 3.0  # ≈ 1.826
    assert np.isclose(model_b_ratio, expected_ratio_b, rtol=1e-3), (
        f"Model B ratio: expected {expected_ratio_b}, got {model_b_ratio}"
    )

    # Check benchmark ratio (should be 1.0)
    benchmark_ratio = comparison_df[comparison_df["unique_id"] == "benchmark"]["rmse_to_benchmark"].iloc[0]
    assert np.isclose(benchmark_ratio, 1.0, rtol=1e-3), f"Benchmark ratio should be 1.0, got {benchmark_ratio}"


def test_compare_to_benchmark_mae(sample_forecast_data):
    """
    Test compare_to_benchmark with mean_abs_error statistic.
    """
    result = compute_accuracy_statistics(data=sample_forecast_data, k=0)

    df = result.to_df()

    # Compare using MAE
    comparison_df = compare_to_benchmark(df, benchmark_model="benchmark", statistic="mean_abs_error")

    # Check that the benchmark column exists
    assert "mean_abs_error_benchmark" in comparison_df.columns
    assert "mean_abs_error_to_benchmark" in comparison_df.columns

    # Check model_a ratio (MAE = 2.5, benchmark = 3.0)
    model_a_ratio = comparison_df[comparison_df["unique_id"] == "model_a"]["mean_abs_error_to_benchmark"].iloc[0]
    expected_ratio_a = 2.5 / 3.0
    assert np.isclose(model_a_ratio, expected_ratio_a, rtol=1e-3)


def test_compare_to_benchmark_rmedse(sample_forecast_data):
    """
    Test compare_to_benchmark with rmedse statistic.
    """
    result = compute_accuracy_statistics(data=sample_forecast_data, k=0)

    df = result.to_df()

    # Compare using RMedSE
    comparison_df = compare_to_benchmark(df, benchmark_model="benchmark", statistic="rmedse")

    # Check that the benchmark column exists
    assert "rmedse_benchmark" in comparison_df.columns
    assert "rmedse_to_benchmark" in comparison_df.columns

    # Check model_a ratio (RMedSE ≈ 2.550, benchmark = 3.0)
    model_a_ratio = comparison_df[comparison_df["unique_id"] == "model_a"]["rmedse_to_benchmark"].iloc[0]
    expected_ratio_a = np.sqrt(6.5) / 3.0
    assert np.isclose(model_a_ratio, expected_ratio_a, rtol=1e-3)


def test_compare_to_benchmark_invalid_model(sample_forecast_data):
    """
    Test that compare_to_benchmark raises an error for invalid benchmark model.
    """
    result = compute_accuracy_statistics(data=sample_forecast_data, k=0)

    df = result.to_df()

    with pytest.raises(ValueError, match="Benchmark model .* not found"):
        compare_to_benchmark(df, benchmark_model="nonexistent_model", statistic="rmse")


def test_create_comparison_table_basic(sample_multi_horizon_data):
    """
    Test that create_comparison_table produces the correct pivot table structure.
    """
    # Get accuracy statistics
    result = compute_accuracy_statistics(data=sample_multi_horizon_data, k=0)

    df = result.to_df()

    # Create comparison table
    table = create_comparison_table(
        df=df,
        variable="gdpkp",
        metric="levels",
        frequency="Q",
        benchmark_model="benchmark",
        statistic="rmse",
        horizons=[0, 1, 2],
    )

    # Check structure
    assert isinstance(table, pd.DataFrame), "Should return a DataFrame"
    assert isinstance(table.columns, pd.MultiIndex), "Should have MultiIndex columns"
    assert table.columns.get_level_values(0)[0] == "Forecast horizon"

    # Check that horizons are in columns
    horizons_in_table = table.columns.get_level_values(1).tolist()
    assert horizons_in_table == [0, 1, 2], f"Expected horizons [0, 1, 2], got {horizons_in_table}"

    # Check that benchmark model is excluded
    assert "benchmark" not in table.index, "Benchmark should not be in the table"

    # Check that model_a is in the table
    assert "model_a" in table.index, "model_a should be in the table"


def test_create_comparison_table_values(sample_multi_horizon_data):
    """
    Test that create_comparison_table calculates correct ratio values.
    """
    result = compute_accuracy_statistics(data=sample_multi_horizon_data, k=0)

    df = result.to_df()

    # Create comparison table
    table = create_comparison_table(
        df=df,
        variable="gdpkp",
        metric="levels",
        frequency="Q",
        benchmark_model="benchmark",
        statistic="rmse",
        horizons=[0, 1, 2],
    )

    # All values should be ratios (non-negative numbers)
    assert (table >= 0).all().all(), "All ratios should be non-negative"

    # Check that table has expected shape
    assert table.shape[0] >= 1, "Should have at least one model (excluding benchmark)"
    assert table.shape[1] == 3, "Should have 3 horizons as columns"


def test_create_comparison_table_horizon_filtering(sample_multi_horizon_data):
    """
    Test that create_comparison_table correctly filters horizons.
    """
    result = compute_accuracy_statistics(data=sample_multi_horizon_data, k=0)

    df = result.to_df()

    # Create table with only horizons 0 and 2
    table = create_comparison_table(
        df=df,
        variable="gdpkp",
        metric="levels",
        frequency="Q",
        benchmark_model="benchmark",
        statistic="rmse",
        horizons=[0, 2],
    )

    # Check that only selected horizons are in the table
    horizons_in_table = table.columns.get_level_values(1).tolist()
    assert horizons_in_table == [0, 2], f"Expected horizons [0, 2], got {horizons_in_table}"


def test_compute_accuracy_statistics_date_columns(sample_forecast_data):
    """
    Test that start_date and end_date columns are correctly populated.
    """
    result = compute_accuracy_statistics(data=sample_forecast_data, k=0)

    df = result.to_df()

    # Check that date columns exist
    assert "start_date" in df.columns
    assert "end_date" in df.columns

    # Check that dates are valid
    for _, row in df.iterrows():
        assert pd.notna(row["start_date"]), "start_date should not be NaN"
        assert pd.notna(row["end_date"]), "end_date should not be NaN"
        assert row["start_date"] <= row["end_date"], "start_date should be <= end_date"
