"""
Tests for bias analysis functions.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.tests.bias import bias_analysis, evaluate_bias


@pytest.fixture
def unbiased_forecast_data():
    """
    Create a ForecastData object with unbiased forecasts.

    Forecasts have random errors centered around zero (mean = 0).
    - Source 'unbiased_model': errors with mean ≈ 0, should fail to reject null hypothesis
    """
    np.random.seed(42)  # For reproducibility
    n = 100  # Large sample for reliable test
    dates = pd.date_range(start="2020-01-01", periods=n, freq="QE")

    forecasts_data = []
    outturns_data = []

    # Generate random errors centered at zero
    errors = np.random.normal(0, 2, n)  # mean=0, std=2

    # Create forecasts with these errors
    actual_values = np.random.normal(100, 10, n)

    for i, date in enumerate(dates):
        # Forecast = Actual + Error (so forecast error = actual - forecast = -error)
        forecasts_data.append(
            {
                "date": date,
                "variable": "gdpkp",
                "vintage_date": date,
                "source": "unbiased_model",
                "frequency": "Q",
                "forecast_horizon": 0,
                "value": actual_values[i] + errors[i],
            }
        )

        # Outturns
        outturns_data.append(
            {
                "date": date,
                "variable": "gdpkp",
                "vintage_date": date + pd.offsets.QuarterEnd(1),
                "frequency": "Q",
                "forecast_horizon": -1,
                "value": actual_values[i],
            }
        )

    forecasts_df = pd.DataFrame(forecasts_data)
    outturns_df = pd.DataFrame(outturns_data)

    return ForecastData(outturns_data=outturns_df, forecasts_data=forecasts_df)


@pytest.fixture
def biased_forecast_data():
    """
    Create a ForecastData object with systematically biased forecasts.

    Creates two sources with known biases:
    - 'optimistic_model': consistently over-predicts by 5 units (positive bias)
    - 'pessimistic_model': consistently under-predicts by 3 units (negative bias)
    """
    np.random.seed(123)
    n = 50
    dates = pd.date_range(start="2020-01-01", periods=n, freq="QE")

    forecasts_data = []
    outturns_data = []

    actual_values = np.random.normal(100, 10, n)

    for i, date in enumerate(dates):
        # Optimistic model: forecasts are 5 units too high (over-predicts)
        # So forecast_error = actual - forecast = -5 on average
        forecasts_data.append(
            {
                "date": date,
                "variable": "gdpkp",
                "vintage_date": date,
                "source": "optimistic_model",
                "frequency": "Q",
                "forecast_horizon": 0,
                "value": actual_values[i] + 5 + np.random.normal(0, 1),  # forecast too high means negative error
            }
        )

        # Pessimistic model: forecasts are 3 units too low (under-predicts)
        # So forecast_error = actual - forecast = +3 on average
        forecasts_data.append(
            {
                "date": date,
                "variable": "gdpkp",
                "vintage_date": date,
                "source": "pessimistic_model",
                "frequency": "Q",
                "forecast_horizon": 0,
                "value": actual_values[i] - 3 + np.random.normal(0, 1),  # forecast too low means positive error
            }
        )

        # Outturns
        outturns_data.append(
            {
                "date": date,
                "variable": "gdpkp",
                "vintage_date": date + pd.offsets.QuarterEnd(1),
                "frequency": "Q",
                "forecast_horizon": -1,
                "value": actual_values[i],
            }
        )

    forecasts_df = pd.DataFrame(forecasts_data)
    outturns_df = pd.DataFrame(outturns_data)

    return ForecastData(outturns_data=outturns_df, forecasts_data=forecasts_df)


@pytest.fixture
def multi_horizon_bias_data():
    """
    Create ForecastData with multiple forecast horizons showing horizon-dependent bias.

    - Horizon 0: unbiased (error mean ≈ 0)
    - Horizon 1: small negative bias (forecast too high by 2, error = -2)
    - Horizon 2: larger negative bias (forecast too high by 4, error = -4)
    """
    np.random.seed(456)
    n = 40
    dates = pd.date_range(start="2020-01-01", periods=n, freq="QE")
    horizons = [0, 1, 2]
    horizon_biases = {0: 0, 1: 2, 2: 4}  # Bias for each horizon

    forecasts_data = []
    outturns_data = []

    actual_values_base = np.random.normal(100, 10, n + 2)  # +2 for horizons

    for i, date in enumerate(dates):
        for horizon in horizons:
            actual_value = actual_values_base[i + horizon]
            bias = horizon_biases[horizon]

            forecasts_data.append(
                {
                    "date": date + pd.offsets.QuarterEnd(horizon),
                    "variable": "gdpkp",
                    "vintage_date": date,
                    "source": "horizon_model",
                    "frequency": "Q",
                    "forecast_horizon": horizon,
                    "value": actual_value + bias + np.random.normal(0, 0.5),
                }
            )

    # Create outturns
    outturns_dates = pd.date_range(start="2020-01-01", periods=n + 2, freq="QE")
    for i, date in enumerate(outturns_dates):
        outturns_data.append(
            {
                "date": date,
                "variable": "gdpkp",
                "vintage_date": date + pd.offsets.QuarterEnd(1),
                "frequency": "Q",
                "forecast_horizon": -1,
                "value": actual_values_base[i],
            }
        )

    forecasts_df = pd.DataFrame(forecasts_data)
    outturns_df = pd.DataFrame(outturns_data)

    return ForecastData(outturns_data=outturns_df, forecasts_data=forecasts_df)


def test_evaluate_bias_unbiased_forecasts(unbiased_forecast_data):
    """
    Test that evaluate_bias correctly identifies unbiased forecasts.

    With forecasts that have errors centered at zero, the p-value should
    be > 0.05, failing to reject the null hypothesis of no bias.
    """
    # Get main table
    df = unbiased_forecast_data._main_table.copy()

    # Run bias test
    result = evaluate_bias(
        df=df,
        variable="gdpkp",
        source="unbiased_model",
        metric="levels",
        frequency="Q",
        forecast_horizon=0,
        verbose=False,
    )

    assert result is not None, "Result should not be None"

    # Extract p-value
    p_value = result.pvalues.iloc[0]
    bias_estimate = result.params.iloc[0]

    # Should fail to reject null hypothesis (p > 0.05)
    assert p_value > 0.05, f"Unbiased forecasts should have p-value > 0.05, got {p_value}"

    # Bias estimate should be close to zero (within 1 unit given sampling variation)
    assert abs(bias_estimate) < 1.0, f"Bias estimate should be near 0, got {bias_estimate}"


def test_evaluate_bias_optimistic_forecasts(biased_forecast_data):
    """
    Test that evaluate_bias correctly identifies optimistically biased forecasts.

    Optimistic forecasts consistently over-predict, leading to positive forecast errors.
    Should reject null hypothesis (p < 0.05) with positive bias estimate.
    """
    df = biased_forecast_data._main_table.copy()

    result = evaluate_bias(
        df=df,
        variable="gdpkp",
        source="optimistic_model",
        metric="levels",
        frequency="Q",
        forecast_horizon=0,
        verbose=False,
    )

    assert result is not None

    p_value = result.pvalues.iloc[0]
    bias_estimate = result.params.iloc[0]

    # Should reject null hypothesis (forecasts are biased)
    assert p_value < 0.05, f"Biased forecasts should have p-value < 0.05, got {p_value}"

    # Bias should be negative (over-prediction: forecast > actual, so actual - forecast < 0)
    assert bias_estimate < -3, f"Optimistic bias should be < -3, got {bias_estimate}"


def test_evaluate_bias_pessimistic_forecasts(biased_forecast_data):
    """
    Test that evaluate_bias correctly identifies pessimistically biased forecasts.

    Pessimistic forecasts consistently under-predict, leading to negative forecast errors.
    Should reject null hypothesis (p < 0.05) with negative bias estimate.
    """
    df = biased_forecast_data._main_table.copy()

    result = evaluate_bias(
        df=df,
        variable="gdpkp",
        source="pessimistic_model",
        metric="levels",
        frequency="Q",
        forecast_horizon=0,
        verbose=False,
    )

    assert result is not None

    p_value = result.pvalues.iloc[0]
    bias_estimate = result.params.iloc[0]

    # Should reject null hypothesis (forecasts are biased)
    assert p_value < 0.05, f"Biased forecasts should have p-value < 0.05, got {p_value}"

    # Bias should be positive (under-prediction: forecast < actual, so actual - forecast > 0)
    assert bias_estimate > 2, f"Pessimistic bias should be > 2, got {bias_estimate}"


def test_bias_analysis_returns_test_result(unbiased_forecast_data):
    """
    Test that bias_analysis returns a TestResult object with correct structure.
    """
    result = bias_analysis(data=unbiased_forecast_data, k=0, verbose=False)

    # Check that result is a TestResult object
    from forecast_evaluation.tests.results import TestResult

    assert isinstance(result, TestResult), "Should return a TestResult object"

    # Check that underlying DataFrame has required columns
    df = result.to_df()
    required_columns = [
        "unique_id",
        "variable",
        "metric",
        "frequency",
        "forecast_horizon",
        "bias_estimate",
        "std_error",
        "t_statistic",
        "p_value",
        "bias_conclusion",
        "n_observations",
        "ci_lower",
        "ci_upper",
    ]

    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"


def test_bias_analysis_unbiased_conclusion(unbiased_forecast_data):
    """
    Test that bias_analysis correctly concludes forecasts are unbiased.
    """
    result = bias_analysis(data=unbiased_forecast_data, k=0, verbose=False)

    df = result.to_df()

    # All forecasts should be classified as "Unbiased"
    assert all(df["bias_conclusion"] == "Unbiased"), "All forecasts should be classified as Unbiased"

    # All p-values should be >= 0.05
    assert all(df["p_value"] >= 0.05), "All p-values should be >= 0.05 for unbiased forecasts"


def test_bias_analysis_biased_conclusion(biased_forecast_data):
    """
    Test that bias_analysis correctly concludes forecasts are biased.
    """
    result = bias_analysis(data=biased_forecast_data, k=0, verbose=False)

    df = result.to_df()

    # All forecasts should be classified as "Biased"
    assert all(df["bias_conclusion"] == "Biased"), "All forecasts should be classified as Biased"

    # All p-values should be < 0.05
    assert all(df["p_value"] < 0.05), "All p-values should be < 0.05 for biased forecasts"

    # Check specific biases
    optimistic = df[df["unique_id"] == "optimistic_model"].iloc[0]
    pessimistic = df[df["unique_id"] == "pessimistic_model"].iloc[0]

    assert optimistic["bias_estimate"] < -3, "Optimistic model should have negative bias < -3"
    assert pessimistic["bias_estimate"] > 2, "Pessimistic model should have positive bias > 2"


def test_bias_analysis_confidence_intervals(biased_forecast_data):
    """
    Test that confidence intervals are properly calculated and exclude zero for biased forecasts.
    """
    result = bias_analysis(data=biased_forecast_data, k=0, verbose=False)

    df = result.to_df()

    for _, row in df.iterrows():
        # Check that confidence interval bounds are ordered correctly
        assert row["ci_lower"] < row["ci_upper"], "CI lower bound should be less than upper bound"

        # Check that bias estimate is within confidence interval
        assert row["ci_lower"] <= row["bias_estimate"] <= row["ci_upper"], (
            "Bias estimate should be within confidence interval"
        )

        # For biased forecasts, confidence interval should not contain zero
        if row["bias_conclusion"] == "Biased":
            if row["bias_estimate"] < 0:
                assert row["ci_upper"] < 0, "For negative bias, CI should not include zero"
            else:
                assert row["ci_lower"] > 0, "For positive bias, CI should not include zero"


def test_bias_analysis_source_filtering(biased_forecast_data):
    """
    Test that source filtering works correctly in bias_analysis.
    """
    # Test single source
    result = bias_analysis(data=biased_forecast_data, source="optimistic_model", k=0, verbose=False)

    df = result.to_df()
    assert len(df["unique_id"].unique()) == 1, "Should only have one source"
    assert df["unique_id"].iloc[0] == "optimistic_model", "Should only have optimistic_model"

    # Test multiple sources
    result = bias_analysis(
        data=biased_forecast_data, source=["optimistic_model", "pessimistic_model"], k=0, verbose=False
    )

    df = result.to_df()
    assert len(df["unique_id"].unique()) == 2, "Should have two sources"
    assert set(df["unique_id"].unique()) == {"optimistic_model", "pessimistic_model"}


def test_bias_analysis_variable_filtering(biased_forecast_data):
    """
    Test that variable filtering works correctly in bias_analysis.
    """
    result = bias_analysis(data=biased_forecast_data, variable="gdpkp", k=0, verbose=False)

    df = result.to_df()
    assert all(df["variable"] == "gdpkp"), "All results should be for gdpkp"


def test_bias_analysis_metadata(unbiased_forecast_data):
    """
    Test that metadata is correctly stored in the TestResult.
    """
    result = bias_analysis(data=unbiased_forecast_data, source="unbiased_model", variable="gdpkp", k=0, verbose=False)

    assert result._metadata["test_name"] == "bias_analysis"
    assert result._metadata["parameters"]["k"] == 0
    assert result._metadata["parameters"]["same_date_range"] is True
    assert result._metadata["parameters"]["verbose"] is False
    assert result._metadata["filters"]["unique_id"] == "unbiased_model"
    assert result._metadata["filters"]["variable"] == "gdpkp"
    assert "date_range" in result._metadata


def test_bias_analysis_horizon_dependence(multi_horizon_bias_data):
    """
    Test that bias increases with forecast horizon as expected.
    """
    result = bias_analysis(data=multi_horizon_bias_data, k=0, verbose=False)

    df = result.to_df()

    # Sort by horizon
    df = df.sort_values("forecast_horizon")

    # Extract bias estimates
    h0_bias = df[df["forecast_horizon"] == 0]["bias_estimate"].iloc[0]
    h1_bias = df[df["forecast_horizon"] == 1]["bias_estimate"].iloc[0]
    h2_bias = df[df["forecast_horizon"] == 2]["bias_estimate"].iloc[0]

    # Horizon 0 should be approximately unbiased
    assert abs(h0_bias) < 1.0, f"Horizon 0 should be unbiased, got {h0_bias}"

    # Horizon 1 should have negative bias around -2 (forecast too high)
    assert -3.0 < h1_bias < -1.0, f"Horizon 1 bias should be ~-2, got {h1_bias}"

    # Horizon 2 should have larger negative bias around -4 (forecast even higher)
    assert -5.0 < h2_bias < -3.0, f"Horizon 2 bias should be ~-4, got {h2_bias}"

    # Bias should become more negative with horizon (over-prediction increases)
    assert h0_bias > h1_bias > h2_bias, "Bias should become more negative with forecast horizon"

    # Check conclusions
    assert df[df["forecast_horizon"] == 0]["bias_conclusion"].iloc[0] == "Unbiased"
    assert df[df["forecast_horizon"] == 1]["bias_conclusion"].iloc[0] == "Biased"
    assert df[df["forecast_horizon"] == 2]["bias_conclusion"].iloc[0] == "Biased"


def test_bias_analysis_n_observations(unbiased_forecast_data):
    """
    Test that n_observations is correctly recorded.
    """
    result = bias_analysis(data=unbiased_forecast_data, k=0, verbose=False)

    df = result.to_df()

    # Should have 100 observations (as defined in fixture)
    assert all(df["n_observations"] == 100), "Should have 100 observations"


def test_evaluate_bias_standard_errors(biased_forecast_data):
    """
    Test that HAC standard errors are reasonable.
    """
    df = biased_forecast_data._main_table.copy()

    result = evaluate_bias(
        df=df,
        variable="gdpkp",
        source="optimistic_model",
        metric="levels",
        frequency="Q",
        forecast_horizon=0,
        verbose=False,
    )

    std_error = result.bse.iloc[0]

    # Standard error should be positive and reasonable (not too large)
    assert std_error > 0, "Standard error should be positive"
    assert std_error < 5, f"Standard error seems unreasonably large: {std_error}"


def test_evaluate_bias_t_statistic(biased_forecast_data):
    """
    Test that t-statistic is calculated correctly (bias / std_error).
    """
    df = biased_forecast_data._main_table.copy()

    result = evaluate_bias(
        df=df,
        variable="gdpkp",
        source="optimistic_model",
        metric="levels",
        frequency="Q",
        forecast_horizon=0,
        verbose=False,
    )

    bias = result.params.iloc[0]
    std_error = result.bse.iloc[0]
    t_stat = result.tvalues.iloc[0]

    # t-statistic should equal bias / std_error
    expected_t = bias / std_error
    assert np.isclose(t_stat, expected_t, rtol=1e-10), f"t-statistic mismatch: {t_stat} vs {expected_t}"


def test_bias_analysis_empty_result():
    """
    Test behavior when filtering produces no results.
    """
    # Create minimal data
    dates = pd.date_range(start="2020-01-01", periods=5, freq="QE")
    forecasts_data = []
    outturns_data = []

    for i, date in enumerate(dates):
        forecasts_data.append(
            {
                "date": date,
                "variable": "gdpkp",
                "vintage_date": date,
                "source": "test_model",
                "frequency": "Q",
                "forecast_horizon": 0,
                "value": 100,
            }
        )
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
    data = ForecastData(outturns_data=outturns_df, forecasts_data=forecasts_df)

    # Filter for non-existent source
    with pytest.raises(ValueError, match="No data available after filtering."):
        bias_analysis(data=data, source="nonexistent_source", k=0, verbose=False)


def test_bias_analysis_test_result_methods(unbiased_forecast_data):
    """
    Test that TestResult methods work correctly for bias_analysis.
    """
    result = bias_analysis(data=unbiased_forecast_data, k=0, verbose=False)

    # Test len()
    assert len(result) > 0, "Result should have at least one row"

    # Test describe()
    desc = result.describe()
    assert isinstance(desc, pd.DataFrame), "describe() should return a DataFrame"

    # Test filter()
    filtered = result.filter(source="unbiased_model")
    assert len(filtered) > 0, "Filtered result should have data"
    assert all(filtered.to_df()["unique_id"] == "unbiased_model")

    # Test to_df()
    df = result.to_df()
    assert isinstance(df, pd.DataFrame), "to_df() should return a DataFrame"


def test_bias_analysis_multiple_variables():
    """
    Test bias_analysis with multiple variables.
    """
    np.random.seed(789)
    n = 30
    dates = pd.date_range(start="2020-01-01", periods=n, freq="QE")

    forecasts_data = []
    outturns_data = []

    variables = ["gdpkp", "cpisa"]
    actual_values = {var: np.random.normal(100, 10, n) for var in variables}

    for i, date in enumerate(dates):
        for var in variables:
            # Different bias for each variable
            bias = 2 if var == "gdpkp" else -1

            forecasts_data.append(
                {
                    "date": date,
                    "variable": var,
                    "vintage_date": date,
                    "source": "test_model",
                    "frequency": "Q",
                    "forecast_horizon": 0,
                    "value": actual_values[var][i] + bias + np.random.normal(0, 0.5),
                }
            )

            outturns_data.append(
                {
                    "date": date,
                    "variable": var,
                    "vintage_date": date + pd.offsets.QuarterEnd(1),
                    "source": "outturn",
                    "frequency": "Q",
                    "forecast_horizon": -1,
                    "value": actual_values[var][i],
                }
            )

    forecasts_df = pd.DataFrame(forecasts_data)
    outturns_df = pd.DataFrame(outturns_data)
    data = ForecastData(outturns_data=outturns_df, forecasts_data=forecasts_df)

    result = bias_analysis(data=data, k=0, verbose=False)

    df = result.to_df()

    # Should have results for both variables
    assert set(df["variable"].unique()) == {"gdpkp", "cpisa"}

    # Check that different variables have different biases
    gdpkp_bias = df[df["variable"] == "gdpkp"]["bias_estimate"].iloc[0]
    cpisa_bias = df[df["variable"] == "cpisa"]["bias_estimate"].iloc[0]

    assert gdpkp_bias < -1.0, "gdpkp should have negative bias"
    assert cpisa_bias > 0, "cpisa should have positive bias"
