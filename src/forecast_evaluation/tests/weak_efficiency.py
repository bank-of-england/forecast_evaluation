from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.tools import add_constant

from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import ensure_consistent_date_range, filter_k


def weak_efficiency_test(
    df: pd.DataFrame,
    variable: str,
    source: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    forecast_horizon: int,
    verbose: bool = True,
) -> Optional[RegressionResultsWrapper]:
    """
    Perform weak efficiency test on forecasts using the Mincer-Zarnowitz regression framework.

    This function tests for weak efficiency in forecasts by regressing actual values against
    a constant term and forecasted values. It performs a joint hypothesis test of whether the
    constant coefficient equals 0 and the forecast coefficient equals 1 (indicating weak
    efficiency). Uses HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
    to account for potential serial correlation and heteroskedasticity.

    The regression equation is: value_outturn = β₀ + β₁ * value_forecast + ε
    Weak efficiency requires: β₀ = 0 and β₁ = 1

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast accuracy data with required columns:

        - 'variable' : str - Variable identifier (e.g., 'gdpkp', 'cpisa', 'unemp')
        - 'source' : str - Forecast source identifier (e.g., 'compass conditional', 'mpr')
        - 'metric' : str - Metric identifier (e.g., 'levels', 'yoy')
        - 'frequency' : str - Data frequency identifier (e.g., 'Q', 'M')
        - 'forecast_horizon' : int - Forecast horizon identifier (0-12)
        - 'value_forecast' : float - Forecast values
        - 'value_outturn' : float - Actual observed values

    variable : str
        Variable to analyze (must exist in df['variable'])

    source : str
        Forecast source to analyze (must exist in df['source'])

    metric : Literal['levels', 'pop', 'yoy']
        Metric to analyze: 'levels' for raw values, 'pop' for period-on-period percentage change,
        'yoy' for year-on-year percentage change (must exist in df['metric'])

    forecast_horizon : int
        Forecast horizon to analyze (must exist in df['forecast_horizon'])

    verbose : bool, default=True
        If True, prints detailed test results including coefficient estimates,
        F-statistic, p-value, and conclusion. If False, returns results silently.

    Returns
    -------
    Optional[RegressionResultsWrapper]
        Dictionary containing test results with keys:

        - 'source' : str - Forecast source identifier
        - 'variable' : str - Variable identifier
        - 'metric' : str - Metric identifier
        - 'frequency' : str - Data frequency identifier
        - 'forecast_horizon' : int - Forecast horizon identifier
        - 'forecast_coef' : float - Coefficient on value_outturn (β₁)
        - 'constant_coef' : float - Constant term coefficient (β₀)
        - 'forecast_se' : float - HAC-corrected standard error of forecast coefficient
        - 'constant_se' : float - HAC-corrected standard error of constant coefficient
        - 'joint_test_fstat' : float - F-statistic for joint hypothesis test
        - 'joint_test_pvalue' : float - P-value for joint hypothesis test
        - 'reject_weak_efficiency' : bool - True if null hypothesis of weak efficiency is rejected (p < 0.05)
        - 'n_observations' : int - Number of observations used in regression
        - 'ols_model' : statsmodels regression results object - Full OLS model results for additional analysis

        Returns None if insufficient data (< 10 observations)
    """

    # Filter data for the specific combination
    subset = df[
        (df["variable"] == variable)
        & (df["unique_id"] == source)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
        & (df["forecast_horizon"] == forecast_horizon)
    ].copy()

    # Check if we have enough observations after filtering
    if len(subset) < 10:
        raise ValueError(
            f"Insufficient data for {variable} from {source} after date filtering. "
            f"Only {len(subset)} observations available (minimum 10 required)"
        )

    # Add constant for regression
    X = add_constant(subset["value_forecast"])

    # Regress forecasts against X
    # Fit with HAC standard errors (Newey-West)
    maxlags = forecast_horizon
    try:
        ols_model = OLS(subset["value_outturn"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    except Exception as e:
        raise ValueError(
            f"OLS regression failed for {variable} from source {source} "
            f"with metric {metric} at horizon {forecast_horizon}. Error: {str(e)}"
        )

    # Extract results
    forecast_value_coef = ols_model.params.iloc[1]
    forecast_value_se = ols_model.bse.iloc[1]
    const_coef = ols_model.params.iloc[0]
    const_se = ols_model.bse.iloc[0]

    # Joint hypothesis test for weak efficiency
    # H0: β0 = 0 AND β1 = 1 (weak efficiency)
    # H1: β0 ≠ 0 OR β1 ≠ 1  (not weakly efficient)
    # Hypothesis given as Rb = q,
    # where R is the restriction matrix, b is the vector of coefficients
    # and q is the vector of restrictions
    hypothesis = (np.eye(2), np.array([0, 1]))
    joint_test = ols_model.f_test(hypothesis)

    # Display results if verbose
    if verbose:
        print(f"\n=== Weak Efficiency Test Results for {variable} (horizon {forecast_horizon}) ===")
        print(f"Estimated coefficient on value_forecast: {forecast_value_coef:.4f}")
        print(f"Estimated constant term: {const_coef:.4f}")
        print("\nJoint Hypothesis Test (H0: β0=0, β1=1):")
        print(f"F-statistic: {joint_test.fvalue:.4f}")
        print(f"p-value: {joint_test.pvalue:.4f}")
        print(f"Reject H0 (not weakly efficient): {joint_test.pvalue < 0.05}")

    # Return comprehensive results
    if ols_model is not None:
        return {
            "unique_id": source,
            "variable": variable,
            "metric": metric,
            "frequency": frequency,
            "forecast_horizon": forecast_horizon,
            "forecast_coef": forecast_value_coef,
            "constant_coef": const_coef,
            "forecast_se": forecast_value_se,
            "constant_se": const_se,
            "joint_test_fstat": joint_test.fvalue,
            "joint_test_pvalue": joint_test.pvalue,
            "reject_weak_efficiency": joint_test.pvalue < 0.05,
            "n_observations": len(subset),
            "ols_model": ols_model,
        }
    else:
        return {
            "unique_id": source,
            "variable": variable,
            "metric": metric,
            "frequency": frequency,
            "forecast_horizon": forecast_horizon,
            "forecast_coef": None,
            "constant_coef": None,
            "forecast_se": None,
            "constant_se": None,
            "joint_test_fstat": None,
            "joint_test_pvalue": None,
            "reject_weak_efficiency": False,
            "n_observations": 0,
            "ols_model": None,
        }


def weak_efficiency_analysis(
    data: ForecastData,
    source: Union[None, str, list[str]] = None,
    variable: Union[None, str, list[str]] = None,
    k: int = 12,
    same_date_range: bool = True,
    verbose: bool = False,
) -> TestResult:
    """
    Run weak efficiency tests for all unique combinations in the dataset.

    This function systematically performs weak efficiency testing across all available
    combinations of variable, source, metric, and forecast_horizon in the
    dataset using the weak_efficiency_test function. It provides a comprehensive
    analysis of forecast efficiency across different variables, sources, and horizons.

    Parameters
    ----------
    data : ForecastData
        Class containing the main table.

    source : None, str, or list of str, default=None
        Filter for specific forecast source(s). If None, includes all sources.
        Can be a single source name or a list of source names.

    variable : None, str, or list of str, default=None
        Filter for specific variable(s). If None, includes all variables.
        Can be a single variable name or a list of variable names.

    k : int, default=12
        Number of revisions used to define the outturns.

    same_date_range : bool, default=True
        If True, ensures consistent date ranges across sources when multiple sources are analyzed.
        If False, uses all available data for each source independently.

    verbose : bool, default=False
        If True, prints detailed results for each individual weak efficiency test
        including coefficient estimates, test statistics, and conclusions.
        If False, only prints summary progress information.

    Returns
    -------
    TestResult
        TestResult object containing a DataFrame with results for each combination and
        metadata about the test parameters. The underlying DataFrame contains columns:

        - 'source' : str - Forecast source identifier
        - 'variable' : str - Variable identifier
        - 'metric' : str - Metric identifier
        - 'frequency' : str - Data frequency identifier
        - 'forecast_horizon' : int - Forecast horizon identifier
        - 'forecast_coef' : float - Coefficient on value_outturn from Mincer-Zarnowitz regression
        - 'constant_coef' : float - Constant term coefficient from Mincer-Zarnowitz regression
        - 'forecast_se' : float - HAC-corrected standard error of forecast coefficient
        - 'constant_se' : float - HAC-corrected standard error of constant coefficient
        - 'joint_test_fstat' : float - F-statistic for joint hypothesis test of weak efficiency
        - 'joint_test_pvalue' : float - P-value for joint hypothesis test
        - 'reject_weak_efficiency' : bool - True if weak efficiency is rejected at 5% significance level
        - 'n_observations' : int - Number of observations used in each test
        - 'ols_model' : object - Full OLS model results (statsmodels RegressionResults object)
    """
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()

    # We first align the main table with what is used in this function
    df = filter_k(df, k)

    # Filter by source if specified
    if source is not None:
        if isinstance(source, str):
            df = df[df["unique_id"] == source]
        else:
            df = df[df["unique_id"].isin(source)]

    # Use consistent date ranges across sources
    if same_date_range and len(df["unique_id"].unique()) > 1:
        df = ensure_consistent_date_range(df)

    # Filter by variable if specified
    if variable is not None:
        if isinstance(variable, str):
            df = df[df["variable"] == variable]
        else:
            df = df[df["variable"].isin(variable)]

    # Get all unique combinations
    combinations = df[["variable", "unique_id", "metric", "frequency", "forecast_horizon"]].drop_duplicates()

    # Preallocate results list for better performance
    n_combinations = len(combinations)
    results_list = [None] * n_combinations

    # Run weak efficiency test for each combination
    for i, (idx, row) in enumerate(combinations.iterrows()):
        variable = row["variable"]
        source = row["unique_id"]
        metric = row["metric"]
        frequency = row["frequency"]
        forecast_horizon = row["forecast_horizon"]

        # Run weak efficiency test with date exclusion
        result = weak_efficiency_test(df, variable, source, metric, frequency, forecast_horizon, verbose=verbose)

        # Store results in preallocated list
        results_list[i] = result

    # Drop None results
    results_list = [res for res in results_list if res is not None]

    results_df = pd.DataFrame(results_list)

    # Create metadata for result object
    metadata = {
        "test_name": "weak_efficiency_analysis",
        "parameters": {
            "k": k,
            "same_date_range": same_date_range,
        },
        "filters": {
            "unique_id": source,
            "variable": variable,
        },
        "date_range": (
            results_df["ols_model"].iloc[0].model.data.orig_endog.index.min() if len(results_df) > 0 else None,
            results_df["ols_model"].iloc[0].model.data.orig_endog.index.max() if len(results_df) > 0 else None,
        )
        if len(results_df) > 0
        else (None, None),
    }

    return TestResult(results_df, data.id_columns, metadata)
