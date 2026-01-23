from typing import Literal, Union

import numpy as np
import pandas as pd
from pydantic import PositiveInt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import ensure_consistent_date_range


def revision_test(
    df: pd.DataFrame,
    variable: str,
    source: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    n_revisions: int = 5,
) -> dict:
    """
    Perform a regression of revisions on lagged revisions and conduct an F-test
    for joint significance of the lagged revisions.

    This function tests whether forecast revisions are predictable based on past
    revisions, which would indicate inefficient information processing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast revision data with columns:
        'variable', 'source', 'metric', 'frequency', 'forecast_horizon', 'revision', 'date'
    variable : str
        Economic variable name (e.g., 'cpisa', 'gdpkp', 'unemp')
    source : str
        Data source identifier (e.g., 'mpr', 'compass conditional')
    metric : Literal["levels", "pop", "yoy"]
        Metric type: 'levels' for raw values, 'pop' for period-on-period percentage change,
        'yoy' for year-on-year percentage change
    frequency : Literal["Q", "M"]
        Data frequency: 'Q' for quarterly, 'M' for monthly
    n_revisions : int, default=5
        Maximum number of forecast horizons/revisions to include in analysis

    Returns
    -------
    dict
        Dictionary containing:

        - 'model': OLS regression results with HAC standard errors
        - 'joint_test': F-test results for joint significance of lagged revisions

    Notes
    -----
    - Uses HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
    - Tests null hypothesis that all lagged revision coefficients are zero
    - Rejecting null suggests forecast revisions are predictable (inefficient)
    - Regression equation: revision_t = α + β₁*revision_{t-1} + ... + βₖ*revision_{t-k} + ε_t,
      where k is the number of revisions (up to n_revisions).
    """

    # Validate input parameters
    if (
        not isinstance(variable, str)
        or not isinstance(source, str)
        or not isinstance(metric, str)
        or not isinstance(frequency, str)
    ):
        raise ValueError("All parameters must be strings.")
    if not isinstance(n_revisions, int) or n_revisions < 1:
        raise ValueError("n_revisions must be a positive integer.")

    # Filter the DataFrame based on the input parameters
    df_filtered = df[
        (df["variable"] == variable)
        & (df["unique_id"] == source)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
        & (df["forecast_horizon"] <= n_revisions)
    ].copy()

    # Check if the DataFrame is empty
    if df_filtered.empty:
        raise ValueError("Filtered DataFrame is empty. Check the input parameters.")

    df_pivot = (
        df_filtered.pivot(index=["date"], columns=["variable", "forecast_horizon"], values=["revision"])
        .reset_index()
        .dropna()
    )

    X = add_constant(df_pivot.iloc[:, 2:])
    y = df_pivot.iloc[:, 1]

    # Here y is the revision to the nowcast, which in principle should not be autocorrelated
    # So we do not set maxlags to H as usual
    # In practice the revisions are still likely to be autocorrelated so we use maxlags=1
    try:
        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    except Exception as e:
        raise ValueError(
            f"OLS regression failed for {variable} from source {source} "
            f"with metric {metric} and frequency {frequency}. Error: {str(e)}"
        )

    restriction_matrix = np.eye(len(X.columns))[1:]
    joint_test = model.f_test(restriction_matrix)

    return model, joint_test


def revision_predictability_analysis(
    data,
    variable: Union[str, list[str]] = None,
    source: Union[None, str, list[str]] = None,
    frequency: Literal["Q", "M"] = "Q",
    n_revisions: PositiveInt = 5,
    same_date_range: bool = True,
) -> TestResult:
    """
    Run the revision test for all unique combinations of variables in the dataset.

    This function systematically applies the revision test to every unique combination
    of variable, source, metric, and frequency in the provided dataset.

    Parameters:
    -----------
    data: An instance of the ForecastData class containing ForecastData._forecasts.
    variable: Single variable name or list of variable names to analyze.
    source: Single source or list of forecast sources to include.
    frequency: Frequency of the data, either quarterly ("Q") or monthly ("M").
    n_revisions: Maximum number of forecast horizons/revisions to include in each test
    same_date_range: If True, ensures consistent date ranges across sources when multiple sources are analyzed.

    Returns
    -------
    TestResult
        TestResult object containing the summary DataFrame with test results and metadata.
        The underlying DataFrame contains columns:

        - 'variable': str - variable name
        - 'source': str - data source
        - 'metric': str - metric type
        - 'frequency': str - data frequency
        - 'joint_test_fstat': float - F-statistic for joint test
        - 'joint_test_pvalue': float - p-value for joint test
        - 'reject_null': bool - whether to reject null at 5% level
        - 'n_observations': int - number of observations in regression

        Returns None if data is not available. Failed tests are excluded from the results.
    """
    if data._forecasts is None:
        raise ValueError(
            "ForecastData forecasts data is not available." + " Please ensure data has been added and processed."
        )

    df = data._forecasts.copy()

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

    # Sort and compute revisions
    df = df.sort_values(by=["variable", "unique_id", "metric", "frequency", "date", "vintage_date"], ascending=True)
    df["revision"] = df.groupby(["variable", "unique_id", "metric", "frequency", "date"])["value"].diff()

    # Get all unique combinations
    combinations = df[["variable", "unique_id", "metric", "frequency"]].copy().drop_duplicates().reset_index(drop=True)

    # Preallocate results list for better performance
    n_combinations = len(combinations)
    results_list = []

    for i, row in combinations.iterrows():
        variable = row["variable"]
        source = row["unique_id"]
        metric = row["metric"]
        frequency = row["frequency"]

        # Run revision test
        try:
            model, joint_test = revision_test(df, variable, source, metric, frequency, n_revisions)

            # Extract F-statistic - handle both array and scalar cases
            if hasattr(joint_test.fvalue, "__getitem__"):
                fstat = joint_test.fvalue[0][0]
            else:
                fstat = float(joint_test.fvalue)

            result_dict = {
                "variable": variable,
                "unique_id": source,
                "metric": metric,
                "frequency": frequency,
                "joint_test_fstat": fstat,
                "joint_test_pvalue": float(joint_test.pvalue),
                "reject_null": joint_test.pvalue < 0.05,
                "n_observations": int(model.nobs),
            }

            results_list.append(result_dict)
        except Exception as e:
            print(f"Combination {i + 1}/{n_combinations}: {variable}, {source}, {metric}, {frequency} - Failed: {e}")

    # Convert list of dictionaries to DataFrame
    if len(results_list) == 0:
        raise ValueError("No successful results to return.")

    results_df = pd.DataFrame(results_list)

    # Create metadata
    metadata = {
        "test_name": "revision_predictability_analysis",
        "parameters": {
            "n_revisions": n_revisions,
            "same_date_range": same_date_range,
            "frequency": frequency,
        },
        "filters": {
            "unique_id": source,
            "variable": variable,
        },
    }

    return TestResult(results_df, data.id_columns, metadata)
