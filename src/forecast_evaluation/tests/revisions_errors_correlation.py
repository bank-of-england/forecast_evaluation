from typing import Literal, Optional, Union

import pandas as pd
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.tools import add_constant

from forecast_evaluation.core.revisions_table import create_revision_dataframe
from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import ensure_consistent_date_range


def revisions_errors_regression(
    df: pd.DataFrame,
    variable: str,
    source: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    forecast_horizon: int,
) -> Optional[RegressionResultsWrapper]:
    """
    Perform a regression of forecast errors on forecast revisions and test for
    statistical significance of the coefficients.

    This function tests whether forecast errors and revisions are correlated,
    which would indicate inefficient information processing according to forecast
    evaluation literature (Nordhaus, 1987; Coibion & Gorodnichenko, 2012).

    The regression equation is:
    forecast_error_t = α + β * revision_t + ε_t

    If β is statistically significant, it suggests forecasters could have used
    information more efficiently.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast revision data with columns:
        'variable', 'source', 'metric', 'forecast_horizon',
        'revision', 'forecast_error'
    variable : str
        Economic variable name (e.g., 'cpisa', 'gdpkp', 'unemp', 'aweagg')
    source : str
        Data source identifier (e.g., 'mpr', 'compass conditional')
    metric : Literal["levels", "pop", "yoy"]
        Metric type: 'levels' for raw values, 'pop' for period-on-period percentage change,
        'yoy' for year-on-year percentage change
    forecast_horizon : int
        Forecast horizon (e.g., 1, 2, 3, 4)

    Returns
    -------
    Optional[RegressionResultsWrapper]
        OLS regression results with HAC standard errors containing:
        - params: regression coefficients [α, β]
        - pvalues: p-values for statistical significance tests
        - rsquared: coefficient of determination
        - nobs: number of observations
        Returns None if no data available for the specified parameters.

    Raises
    ------
    ValueError
        If the filtered DataFrame is empty (no data for specified parameters)

    Notes
    -----
    - Uses HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
      with Newey-West correction and maxlags=H
    - Tests null hypothesis H0: β = 0 (no correlation between revisions and errors)
    - Rejecting H0 suggests forecast inefficiency
    """

    # Filter data for the specific combination
    subset = df[
        (df["variable"] == variable)
        & (df["unique_id"] == source)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
        & (df["forecast_horizon"] == forecast_horizon)
    ].copy()

    # Check if the DataFrame is empty
    if subset.empty:
        raise ValueError("Filtered DataFrame is empty. Check the input parameters.")

    # Add constant for regression
    X = add_constant(subset["revision"])
    y = subset["forecast_error"]

    # Regress forecast errors against constant and forecast errors
    try:
        model = OLS(y, X)
        # Fit with HAC standard errors (Newey-West)
        maxlags = forecast_horizon
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    except Exception as e:
        raise ValueError(
            f"OLS regression failed for {variable} from source {source} "
            f"with metric {metric} at horizon {forecast_horizon}. Error: {str(e)}"
        )

    return results


def revisions_errors_correlation_analysis(
    data: ForecastData,
    source: Union[None, str, list[str]] = None,
    variable: Union[None, str, list[str]] = None,
    k: int = 12,
    same_date_range: bool = True,
) -> TestResult:
    """
    Run regressions of forecast revisions against forecast errors for all unique
    combinations of variable, source, metric, and forecast_horizon.

    This function systematically tests forecast efficiency across all available
    forecast series by running the revisions-errors regression for each unique
    combination of forecasting parameters.

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

    k : int, optional, default=12
        Number of revisions used to define the outturns.

    same_date_range : bool, default=True
        If True, ensures consistent date ranges across sources when multiple sources are analyzed.
        If False, uses all available data for each source independently.

    Returns
    -------
    TestResult
        TestResult object containing the summary DataFrame with test results and metadata.
        The underlying DataFrame contains:

        - source: str - forecast source identifier
        - variable: str - economic variable name
        - metric: str - measurement type
        - forecast_horizon: int - forecast horizon
        - const: float - intercept coefficient (α)
        - const_se: float - standard error of intercept
        - beta: float - slope coefficient (β)
        - beta_se: float - standard error of slope
        - const_pvalue: float - p-value for intercept test
        - beta_pvalue: float - p-value for slope test
        - correlated: bool - True if β is significant at 5% level
        - rsquared: float - coefficient of determination
        - n_observations: int - number of observations in regression
    """
    if data._main_table is None or data._forecasts is None:
        raise ValueError("ForecastData missing data. Please ensure data has been added and processed.")

    df = create_revision_dataframe(data._main_table, data._forecasts, k)

    # Filter by source if specified
    if source is not None:
        if isinstance(source, str):
            df = df[df["unique_id"] == source]
        else:
            df = df[df["unique_id"].isin(source)]

    # Ensure consistent date range across all sources for each variable
    if same_date_range and (df["unique_id"].nunique() > 1):
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

    # Run regression for each combination
    for i, (idx, row) in enumerate(combinations.iterrows()):
        variable = row["variable"]
        source = row["unique_id"]
        metric = row["metric"]
        frequency = row["frequency"]
        forecast_horizon = row["forecast_horizon"]

        try:
            results = revisions_errors_regression(df, variable, source, metric, frequency, forecast_horizon)
            results_list[i] = {
                "unique_id": source,
                "variable": variable,
                "metric": metric,
                "frequency": frequency,
                "forecast_horizon": forecast_horizon,
                "const": results.params.iloc[0],
                "const_se": results.bse.iloc[0],
                "beta": results.params.iloc[1],
                "beta_se": results.bse.iloc[1],
                "const_pvalue": results.pvalues.iloc[0],
                "beta_pvalue": results.pvalues.iloc[1],
                "correlated": results.pvalues.iloc[1] < 0.05,
                "rsquared": results.rsquared,
                "n_observations": results.nobs,
            }
        except Exception as e:
            print(
                f"Failed regression for variable={variable}, source={source}, metric={metric}, "
                + f"frequency={frequency}, h={forecast_horizon}: {e}"
            )

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Create metadata
    metadata = {
        "test_name": "revisions_errors_correlation_analysis",
        "parameters": {
            "same_date_range": same_date_range,
        },
        "filters": {
            "unique_id": source,
            "variable": variable,
        },
    }

    return TestResult(results_df, data.id_columns, metadata)
