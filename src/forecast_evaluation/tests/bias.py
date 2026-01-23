from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper

from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import ensure_consistent_date_range, filter_k


def evaluate_bias(
    df: pd.DataFrame,
    variable: str,
    source: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    forecast_horizon: int,
    verbose: bool = True,
) -> Optional[RegressionResultsWrapper]:
    """
    Evaluate forecast bias using regression-based statistical testing.

    This function tests for systematic bias in forecasts by regressing
    forecast errors against a constant term and testing whether the constant
    is statistically significantly different from zero. Uses HAC
    (Heteroskedasticity and Autocorrelation Consistent) standard errors to
    account for potential serial correlation and heteroskedasticity in forecast errors.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing forecast accuracy data with required columns:

        - 'variable' : str - Variable identifier (e.g., 'gdpkp', 'cpisa', 'unemp')
        - 'source' : str - Forecast source identifier (e.g., 'compass conditional', 'mpr')
        - 'metric' : str - Metric identifier
        - 'forecast_horizon' : int - Forecast horizon identifier (0-12)
        - 'forecast_error' : float - Forecast errors (actual - forecast)
        - 'date' : datetime - Date for time series ordering
    variable : str
        Variable to analyze (must exist in df['variable'])
    source : str
        Forecast source to analyze (must exist in df['source'])
    metric : str
        Metric to analyze (must exist in df['metric'])
    forecast_horizon : int
        Forecast horizon to analyze (must exist in df['forecast_horizon'])
    verbose : bool, default=True
        If True, prints detailed test results including bias estimate, p-value,
        and conclusion. If False, returns results silently.

    Returns
    -------
    Optional[RegressionResultsWrapper]
        OLS regression results object containing:
        - Parameter estimates and standard errors
        - Test statistics and p-values
        - Model diagnostics and summary statistics
        Returns None if insufficient data (< 10 observations)

    Notes
    -----
    Statistical Test:
        The bias test regresses forecast errors on a constant:

        .. math::

            error_t = α + ε_t

        where:

        - error_t = actual_t - forecast_t (forecast error)
        - α = bias parameter (constant term)
        - ε_t = regression residual

        Null Hypothesis: H₀: α = 0 (unbiased forecasts)
        Alternative: H₁: α ≠ 0 (biased forecasts)

    HAC Standard Errors:
        Uses Newey-West HAC standard errors to correct for:

        - Heteroskedasticity (non-constant error variance)
        - Autocorrelation (serial correlation in forecast errors)
        - Maximum lags = 2

    Interpretation:
        - If p-value < 0.05: Reject H₀, conclude forecasts are biased
        - If p-value ≥ 0.05: Fail to reject H₀, conclude forecasts are unbiased
        - Positive α: Forecasts systematically under-predict (optimistic bias)
        - Negative α: Forecasts systematically over-predict (pessimistic bias)
    """

    # Filter data for the specific combination
    subset = df[
        (df["variable"] == variable)
        & (df["unique_id"] == source)
        & (df["metric"] == metric)
        & (df["forecast_horizon"] == forecast_horizon)
    ].copy()

    # Add constant for regression
    X = sm.add_constant(np.ones(len(subset)))

    # Regress forecast errors against a constant
    try:
        model = OLS(subset["forecast_error"], X)
        # Fit with HAC standard errors (Newey-West)
        maxlags = forecast_horizon
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    except Exception as e:
        raise ValueError(
            f"OLS regression failed for {variable} from source {source} "
            f"with metric {metric} at horizon {forecast_horizon}. Error: {str(e)}"
        )

    # Extract results
    bias = results.params.iloc[0]
    p_value = results.pvalues.iloc[0]

    if verbose:
        print(f"\nBias test for {variable} from {source}")
        print(f"Average Forecast Error: {bias:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Conclusion: {'Biased' if p_value < 0.05 else 'Unbiased'} forecasts")

    return results


# Run bias tests for all combinations and create summary table
def bias_analysis(
    data: ForecastData,
    source: Union[None, str, list[str]] = None,
    variable: Union[None, str, list[str]] = None,
    k: int = 12,
    same_date_range: bool = True,
    verbose: bool = False,
) -> TestResult:
    """
    Run bias tests for all unique combinations of variable, source, metric,
    and forecast_horizon.

    This function performs systematic bias testing across all available combinations
    in the dataset using the evaluate_bias function. It runs regression-based bias
    tests and aggregates results into a comprehensive summary.

    Parameters
    ----------
    data : ForecastData
        An instance of the ForecastData class containing ForecastData._main_table.
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
        If True, prints detailed results for each individual bias test.
        If False, only prints summary statistics at the end.

    Returns
    -------
    TestResult
        TestResult object containing the summary DataFrame with bias test results and metadata.
        The underlying DataFrame contains columns:

        - 'source' : str - Forecast source identifier
        - 'variable' : str - Variable identifier
        - 'metric' : str - Metric identifier
        - 'frequency' : str - Data frequency identifier
        - 'forecast_horizon' : int - Forecast horizon identifier
        - 'bias_estimate' : float - Estimated bias coefficient (constant term from regression)
        - 'std_error' : float - HAC-corrected standard error of bias estimate
        - 't_statistic' : float - t-statistic for bias test
        - 'p_value' : float - Two-tailed p-value for bias test
        - 'bias_conclusion' : str - 'Biased' if p < 0.05, 'Unbiased' if p >= 0.05
        - 'n_observations' : int - Number of observations used in the test
        - 'ci_lower' : float - Lower bound of 95% confidence interval for bias estimate
        - 'ci_upper' : float - Upper bound of 95% confidence interval for bias estimate
    """
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    data_filtered = data.copy()
    data_filtered.filter(sources=source, variables=variable)

    if data_filtered._main_table.empty:
        raise ValueError("No data available after filtering.")

    df = data_filtered._main_table.copy()

    # Store original date range for metadata
    original_date_range = (df["date"].min(), df["date"].max())

    # We first align the main table with what is used in this function
    df = filter_k(df, k)

    # Store filter information for metadata
    source_filter = source
    variable_filter = variable

    # Ensure consistent date range across all sources for each variable
    if same_date_range and (df["unique_id"].nunique() > 1):
        df = ensure_consistent_date_range(df)

    # Get all unique combinations
    combinations = df[["variable", "unique_id", "metric", "frequency", "forecast_horizon"]].drop_duplicates()

    # Preallocate results list for better performance
    n_combinations = len(combinations)
    results_list = [None] * n_combinations

    # Run bias test for each combination
    for i, (idx, row) in enumerate(combinations.iterrows()):
        variable = row["variable"]
        source = row["unique_id"]
        metric = row["metric"]
        frequency = row["frequency"]
        forecast_horizon = row["forecast_horizon"]

        # Run bias test
        result = evaluate_bias(df, variable, source, metric, frequency, forecast_horizon, verbose=verbose)

        # Extract key statistics
        bias_estimate = result.params.iloc[0]
        p_value = result.pvalues.iloc[0]
        std_error = result.bse.iloc[0]
        t_stat = result.tvalues.iloc[0]
        n_obs = result.nobs

        # Determine if biased (at 5% significance level)
        is_biased = p_value < 0.05

        # Get confidence interval
        conf_int = result.conf_int()
        ci_lower = conf_int.iloc[0, 0]
        ci_upper = conf_int.iloc[0, 1]

        # Store results in preallocated list
        results_list[i] = {
            "unique_id": source,
            "variable": variable,
            "metric": metric,
            "frequency": frequency,
            "forecast_horizon": forecast_horizon,
            "bias_estimate": bias_estimate,
            "std_error": std_error,
            "t_statistic": t_stat,
            "p_value": p_value,
            "bias_conclusion": "Biased" if is_biased else "Unbiased",
            "n_observations": n_obs,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    # Create summary DataFrame
    bias_summary = pd.DataFrame(results_list)

    # Create metadata
    metadata = {
        "test_name": "bias_analysis",
        "parameters": {
            "k": k,
            "same_date_range": same_date_range,
            "verbose": verbose,
        },
        "filters": {
            "unique_id": source_filter,
            "variable": variable_filter,
        },
        "date_range": (df["date"].min(), df["date"].max()) if len(df) > 0 else original_date_range,
    }

    return TestResult(bias_summary, data.id_columns, metadata)
