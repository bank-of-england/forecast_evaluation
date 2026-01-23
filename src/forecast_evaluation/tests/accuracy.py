from typing import Literal, Union

import numpy as np
import pandas as pd

from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import ensure_consistent_date_range, filter_k


def compute_accuracy_statistics(
    data: ForecastData,
    source: Union[None, str, list[str]] = None,
    variable: Union[None, str, list[str]] = None,
    k: int = 12,
    same_date_range: bool = True,
) -> TestResult:
    """
    Calculate accuracy statistics for all unique combinations of variable, source, metric
    and forecast_horizon.

    This function computes the Root Mean Square Error (RMSE), Mean Absolute Error (MAE),
    Root Median Square Error (RMedSE) and the number of observations for each combination.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing the main table with forecast accuracy data.
    source : None, str, or list of str, default=None
        Filter for specific forecast source(s). If None, includes all sources.
        Can be a single source name or a list of source names.
    variable : None, str, or list of str, default=None
        Filter for specific variable(s). If None, includes all variables.
        Can be a single variable name or a list of variable names.
    k : int, optional, default=12
        Number of revisions used to define the outturns.
    same_date_range : bool, optional, default=True
        If True, ensures consistent date ranges across sources when multiple sources are analyzed.
        If False, uses all available data for each source independently.

    Returns
    -------
    TestResult
        TestResult object containing the summary DataFrame with accuracy statistics and metadata.
        The underlying DataFrame contains columns:

        - 'variable' : str - Variable identifier
        - 'source' : str - Forecast source identifier
        - 'metric' : str - Metric identifier
        - 'frequency' : str - Frequency identifier
        - 'forecast_horizon' : int - Forecast horizon identifier
        - 'rmse' : float - Root Mean Square Error
        - 'rmedse' : float - Root Median Square Error
        - 'mean_abs_error' : float - Mean Absolute Error
        - 'n_observations' : int - Number of observations used in the calculation
        - 'start_date' : datetime - Earliest forecast vintage date in the group
        - 'end_date' : datetime - Latest forecast vintage date in the group
    """
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()

    # Store original date range for metadata
    original_date_range = (df["vintage_date_forecast"].min(), df["vintage_date_forecast"].max())

    # Filter dataset for particular value of k used to determined the outturns
    df = filter_k(df, k)

    # Store filter information for metadata
    source_filter = source
    variable_filter = variable

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

    # Pre-calculate squared errors and absolute errors
    df["squared_error"] = df["forecast_error"] ** 2
    df["abs_error"] = df["forecast_error"].abs()

    # Group by all combinations and calculate statistics
    groupby_cols = ["variable", "unique_id", "metric", "frequency", "forecast_horizon"]

    # Use agg to calculate multiple statistics at once
    accuracy_summary = (
        df.groupby(groupby_cols)
        .agg(
            {
                "squared_error": ["mean", "median"],
                "abs_error": "mean",
                "forecast_error": "count",
                "vintage_date_forecast": ["min", "max"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    accuracy_summary.columns = [
        "variable",
        "unique_id",
        "metric",
        "frequency",
        "forecast_horizon",
        "mse",
        "median_se",
        "mean_abs_error",
        "n_observations",
        "start_date",
        "end_date",
    ]

    # Calculate accuracy metrics: RMSE, RMedSE
    accuracy_summary["rmse"] = np.sqrt(accuracy_summary["mse"])
    accuracy_summary["rmedse"] = np.sqrt(accuracy_summary["median_se"])

    # Drop intermediate column
    accuracy_summary = accuracy_summary.drop("median_se", axis=1)

    # Create metadata
    metadata = {
        "test_name": "compute_accuracy_statistics",
        "parameters": {
            "k": k,
            "same_date_range": same_date_range,
        },
        "filters": {
            "unique_id": source_filter,
            "variable": variable_filter,
        },
        "date_range": (
            (accuracy_summary["start_date"].min(), accuracy_summary["end_date"].max())
            if len(accuracy_summary) > 0
            else original_date_range
        ),
    }

    return TestResult(accuracy_summary, data.id_columns, metadata)


def compare_to_benchmark(
    df: pd.DataFrame, benchmark_model: str, statistic: Literal["rmse", "rmedse", "mean_abs_error"] = "rmse"
) -> pd.DataFrame:
    """
    Compare each model's accuracy statistic to a benchmark model's statistic.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing accuracy statistics with columns:

        - 'variable' : str - Variable identifier
        - 'source' : str - Forecast source identifier
        - 'metric' : str - Metric identifier
        - 'forecast_horizon' : int - Forecast horizon identifier
        - 'rmse' : float - Root Mean Square Error
        - 'rmedse' : float - Root Median Square Error
        - 'mean_abs_error' : float - Mean Absolute Error
        - 'n_observations' : int - Number of observations
    benchmark_model : str
        The model to use as the benchmark for comparison. (e.g., 'mpr')
    statistic : str, optional
        The accuracy statistic to compare. Must be one of 'rmse', 'rmedse', or 'mean_abs_error'.
        Default is 'rmse'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional column:

        - '{statistic}_to_benchmark' : float - Ratio of model's statistic to benchmark model's statistic
    """
    # Validate benchmark_model parameter
    if benchmark_model not in df["unique_id"].unique():
        raise ValueError(f"Benchmark model '{benchmark_model}' not found in the accuracy DataFrame sources.")

    # Separate benchmark data
    benchmark_df = df[df["unique_id"] == benchmark_model].copy()
    benchmark_df = benchmark_df[["variable", "metric", "frequency", "forecast_horizon", statistic]]
    benchmark_df = benchmark_df.rename(columns={statistic: f"{statistic}_benchmark"})

    # Merge benchmark statistics back into the main DataFrame
    comparison_df = df.merge(benchmark_df, on=["variable", "metric", "frequency", "forecast_horizon"], how="left")

    # Calculate the ratio from benchmark
    comparison_df[f"{statistic}_to_benchmark"] = comparison_df[statistic] / comparison_df[f"{statistic}_benchmark"]

    return comparison_df


def create_comparison_table(
    df: pd.DataFrame,
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    benchmark_model: str,
    statistic: Literal["rmse", "rmedse", "mse", "mean_abs_error"] = "rmse",
    horizons: list[int] = [0, 1, 2, 4, 8, 12],
) -> pd.DataFrame:
    """
    Create a comparison table showing the ratio of each model's accuracy statistic
    to a benchmark model's statistic across selected forecast horizons.

    This function filters the data for a specific variable, metric and frequency combination,
    then creates a pivot table with forecast sources as rows and forecast horizons as columns.
    The values represent the ratio of each model's accuracy statistic to the benchmark model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing accuracy statistics with columns:

        - 'variable' : str - Variable identifier (e.g., 'gdpkp', 'cpisa', 'unemp')
        - 'source' : str - Forecast source identifier (e.g., 'compass conditional', 'mpr')
        - 'metric' : str - Metric identifier (e.g., 'yoy', 'levels')
        - 'forecast_horizon' : int - Forecast horizon identifier (forecast horizon)
        - 'rmse' : float - Root Mean Square Error
        - 'rmedse' : float - Root Median Square Error
        - 'mean_abs_error' : float - Mean Absolute Error
        - 'n_observations' : int - Number of observations
    variable : str
        Variable to analyze (e.g., 'aweagg', 'cpisa', 'gdpkp', 'unemp').
    metric : str
        Metric to analyze (e.g., 'yoy', 'levels').
    benchmark_model : str
        The forecast source to use as the benchmark for comparison (e.g., 'mpr').
    statistic : str, optional
        The accuracy statistic to compare. Must be one of 'rmse', 'rmedse', or 'mean_abs_error'.
        Default is 'rmse'.
    horizons : list of int, optional
        List of forecast horizons to include in the table. Default is [0, 1, 2, 4, 8, 12].

    Returns
    -------
    pandas.DataFrame
        Pivot table with MultiIndex columns where:

        - Index: forecast sources (excluding baseline models and benchmark model)
        - Columns: MultiIndex with 'Forecast horizon' as top level and forecast horizons as second level
        - Values: ratio of model's accuracy statistic to benchmark model's statistic
    """
    # Compute comparison to benchmark model
    df = compare_to_benchmark(df, benchmark_model=benchmark_model, statistic=statistic)

    # Extract the ratio column name
    ratio_col = f"{statistic}_to_benchmark"

    # Filter data for the specific combination
    mask = (df["variable"] == variable) & (df["metric"] == metric) & (df["frequency"] == frequency)

    df = df.loc[mask].copy()

    # Filter horizons
    df = df[df["forecast_horizon"].isin(horizons)]
    # Remove benchmark model
    df = df[df["unique_id"] != benchmark_model]

    # Select columns for the table
    table_df = df[["unique_id", "forecast_horizon", ratio_col]].copy()

    # Pivot the table to have sources as rows and forecast_horizon as columns
    table_df = table_df.pivot(index="unique_id", columns="forecast_horizon", values=ratio_col)

    # Sort the table by source
    table_df = table_df.sort_index()

    # Remove the index name
    table_df.index.name = None

    # Create MultiIndex columns with "Forecast horizon" as the top level
    horizons = table_df.columns
    table_df.columns = pd.MultiIndex.from_product([["Forecast horizon"], horizons], names=[None, None])

    return table_df


# Example usage:
if __name__ == "__main__":
    import pandas as pd

    from forecast_evaluation.data.ForecastData import ForecastData

    # Initialise with fer ---------
    forecast_data = ForecastData(load_fer=True)

    # Run DM test comparison table
    accuracy_results = compute_accuracy_statistics(
        data=forecast_data,
        k=12,
    )
