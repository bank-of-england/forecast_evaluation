import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acovf

# internal imports
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import ensure_consistent_date_range, filter_k


def diebold_mariano_test(
    error_difference: pd.Series,
    horizon: int,
) -> dict:
    """
    Perform the Diebold-Mariano test to compare forecast accuracy between two models.

    The Diebold-Mariano test assesses whether the difference in forecast accuracy
    between two models is statistically significant. It accounts for autocorrelation
    in forecast errors using Newey-West HAC standard errors.

    We use the correction for small sample of Harvey, Leybourne, and Newbold (1997).
    We use the approach of Harvey, Leybourne, and Whitehouse (2017) which suggests
    using only the Bartlett kernel when the variance is negative. The standard
    variance estimator has better small sample properties.

    Parameters
    ----------
    error_difference : pandas.Series
        Differential if forecast errors from the model being evaluated.
        error = (actual - forecast)
        error_difference = (errors_model)**2 - (errors_benchmark)**2; doesnt have to be square func

    horizon : int, optional
        The forecast horizon (h-step ahead). Used to determine the number of lags
        for HAC standard errors.

    Returns
    -------
    dict
        Dictionary containing test results:

        - 'dm_statistic': float - The Diebold-Mariano test statistic
        - 'p_value': float - Two-tailed p-value (tests if losses are significantly different)
        - 'mean_loss_diff': float - Mean difference in losses (model - benchmark)
        - 'interpretation': str - Interpretation of the results

    Notes
    -----
    - Null hypothesis: The two models have equal predictive accuracy
    - Alternative hypothesis: The models have different predictive accuracy
    - A negative DM statistic indicates the model is more accurate than the benchmark
    - Uses Newey-West HAC standard errors with lags = horizon (because we start a horizon = 0)

    References
    Diebold and Mariano (1995) : https://doi.org/10.2307/1392185
    Harvey, Leybourne, and Newbold (1997) : https://doi.org/10.1016/S0169-2070(96)00719-4
    Harvey, Leybourne, and Whitehouse (2017) https://doi.org/10.1016/j.ijforecast.2017.05.001
    """

    # Mean loss differential
    mean_loss_diff = error_difference.mean()

    # Number of observations
    n = len(error_difference)

    if n < 2:
        raise ValueError("Insufficient observations for Diebold-Mariano test (need at least 2)")

    # WARNING in the future we should work only with horizon starting at 1
    # Nowcast should be horizon=1 - otherwise we can't guess if a user's horizon = 1 corresponds
    # to one-step ahead forecasts or two-step ahead forecasts
    # In the meantime add one to horizon
    H = horizon + 1

    # Calculate HAC standard error using Newey-West ------------
    maxlags = H - 1

    # Get autocovariances from lag 0 to maxlags
    error_cov = acovf(error_difference, nlag=maxlags, fft=False)

    # standard variance estimator
    variance = (error_cov[0] + 2 * np.sum(error_cov[1:])) / n

    if variance < 0:
        # Use Bartlett kernel weights
        variance = error_cov[0] / n
        for i in range(1, H):
            variance += error_cov[i] * 2 * (1 - i / H) / n
    # -----------------------------------------------------------

    # Calculate DM statistic
    dm_statistic = mean_loss_diff / np.sqrt(variance)

    if (np.abs(mean_loss_diff) < 1e-6) & (np.sqrt(variance) < 1e-6):
        # more or less identical forecasts which gives a variance too close to zero
        # this inflates the DM stat, and we don't want that
        warnings.warn(
            "Diebold-Mariano test: Mean loss difference and its variance are both very close to zero. "
            "This suggests identical forecasts. "
            "Setting DM statistic to zero to avoid artificially high stats."
        )
        dm_statistic = 0.0

    # Adjustment for small sample sizes (Harvey et al. 1997, eq. 9)
    # https://doi.org/10.1016/S0169-2070(96)00719-4
    harvey_adj = np.sqrt((n + 1 - 2 * H + (H / n) * (H - 1)) / n)
    dm_statistic = dm_statistic * harvey_adj

    # Calculate p-value (two-tailed test using t-distribution)
    p_value = 2 * stats.t.cdf(-abs(dm_statistic), df=n - 1)

    return pd.Series({"dm_statistic": dm_statistic, "p_value": p_value, "n_observations": n})


def diebold_mariano_table(
    data,
    benchmark_model: str,
    k: int = 12,
    loss_function: Literal["mse", "mae"] = "mse",
    horizons: list[int] = None,
) -> TestResult:
    """
    Run Diebold-Mariano tests comparing all models to a benchmark across all series.

    This function performs DM tests for every combination of variable, metric, frequency,
    and forecast horizon, comparing each model's forecast errors to the benchmark model.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing the main table with forecast accuracy data.
    benchmark_model : str
        The forecast source to use as the benchmark (e.g., 'mpr')
    k : int, optional
        Number of revisions used to define the outturns. Default is 12.
    loss_function : Literal["mse", "mae"], optional
        Loss function to use for comparison. Default is "mse".
    horizons : list of int, optional
        List of forecast horizons to test. Default is all horizons in the data.

    Returns
    -------
    TestResult
        TestResult object containing the summary DataFrame with test results and metadata.
        The underlying DataFrame contains columns:

        - 'variable': Variable identifier
        - 'metric': Metric identifier
        - 'frequency': Frequency identifier
        - 'source': Model being compared to benchmark
        - 'forecast_horizon': Forecast horizon
        - 'dm_statistic': DM test statistic
        - 'p_value': P-value from DM test
        - 'n_observations': Number of observations used
        - 'rmse_ratio': Ratio of model RMSE to benchmark RMSE
        - 'benchmark_source': Benchmark model name
    """
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()

    # retrieve id columns but unique_id
    id_cols = [col for col in data.id_columns if col != "unique_id"]
    df.drop(columns=id_cols, errors="ignore", inplace=True)

    # If the use provided a subset of the horizons, filter accordingly
    if horizons is not None:
        # check that horizons are in the data
        available_horizons = df["forecast_horizon"].unique().tolist()
        horizons_valid = [h for h in horizons if h in available_horizons]
        df = df[df["forecast_horizon"].isin(horizons_valid)]

    # Filter dataset for particular value of k used to determined the outturns
    df = filter_k(df, k)

    # Ensure consistent date range across all sources for each variable
    df = ensure_consistent_date_range(df)

    # Pre-calculate squared errors and absolute errors
    if loss_function == "mse":
        df["forecast_error_transformed"] = df["forecast_error"] ** 2
    elif loss_function == "mae":
        df["forecast_error_transformed"] = df["forecast_error"].abs()

    # Compute error differences with respect to the benchmark model
    # diff = error_model - error_benchmark
    # first drop unused columns to avoid duplication in merge
    df.drop(columns=["value_outturn", "value_forecast"], errors="ignore", inplace=True)
    merge_on = df.columns.difference(["unique_id", "forecast_error", "forecast_error_transformed"]).tolist()

    benchmark_df = df[df["unique_id"] == benchmark_model]
    merged_df = pd.merge(
        df,
        benchmark_df[merge_on + ["forecast_error", "forecast_error_transformed"]],
        on=merge_on,
        suffixes=("", "_benchmark"),
    )
    merged_df["error_difference"] = (
        merged_df["forecast_error_transformed"] - merged_df["forecast_error_transformed_benchmark"]
    )

    # Drop the benchmark
    merged_df = merged_df[merged_df["unique_id"] != benchmark_model]

    # Group by all combinations and calculate statistics
    groupby_cols = ["variable", "unique_id", "metric", "frequency", "forecast_horizon"]

    # Define function to apply DM test to each group
    def apply_dm_test(group):
        # need this here because the horizon is given in the grouping
        test_results = diebold_mariano_test(
            error_difference=group["error_difference"], horizon=group["forecast_horizon"].iloc[0]
        )

        # also return RMSE ratio
        rmse_ratio = np.sqrt(np.mean(group["forecast_error"] ** 2)) / (
            np.sqrt(np.mean(group["forecast_error_benchmark"] ** 2))
        )

        # Combine results
        test_results["rmse_ratio"] = rmse_ratio

        return test_results

    # Run DM test for each group using groupby + apply
    results_df = (
        merged_df.groupby(groupby_cols, group_keys=True)[
            groupby_cols + ["error_difference"] + ["forecast_error"] + ["forecast_error_benchmark"]
        ]
        .apply(apply_dm_test)
        .reset_index()
    )

    # Add benchmark model column
    results_df["benchmark_source"] = benchmark_model

    # Create metadata
    metadata = {
        "test_name": "diebold_mariano_table",
        "parameters": {
            "benchmark_model": benchmark_model,
            "k": k,
            "loss_function": loss_function,
            "horizons": horizons,
        },
    }

    return TestResult(results_df, data.id_columns, metadata)


# Example usage:
if __name__ == "__main__":
    import pandas as pd

    from forecast_evaluation.data.ForecastData import ForecastData

    # Initialise with fer ---------
    forecast_data = ForecastData(load_fer=True)

    # Run DM test comparison table
    results_dm = diebold_mariano_table(
        data=forecast_data,
        benchmark_model="mpr",
    )
