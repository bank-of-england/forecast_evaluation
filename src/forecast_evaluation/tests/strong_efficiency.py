from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import filter_k


def strong_efficiency_test(
    df_pivot: pd.DataFrame,
    outcome_variable: str,
    h: int,
    instrument_variable: str,
    j: int = 2,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Perform a strong efficiency test on forecast data.

    Parameters
    ----------
    df_pivot : pd.DataFrame
        Pivoted dataframe with forecast data.
    outcome_variable : str
        Variable for forecast error analysis.
    h : int
        Forecast horizon of the outcome variable (target horizon).
    instrument_variable : str
        Variable used as the instrument.
    j : int, optional
        Forecast horizon of the instrument variable, by default 2.
    alpha : float, optional
        Significance level for confidence intervals, by default 0.05.
    verbose : bool, optional
        Whether to print detailed results, by default True.

    Returns
    -------
    dict
        Dictionary with test results including:

        - horizon: Forecast horizon
        - instrument_horizon: Instrument horizon
        - outcome_variable: Name of outcome variable
        - instrument_variable: Name of instrument variable
        - n_observations: Number of observations
        - ols_coefficient: OLS coefficient estimate
        - ols_se: Standard error of coefficient
        - coeff_ci_lower: Lower bound of confidence interval
        - coeff_ci_upper: Upper bound of confidence interval
        - z_stat: Z-statistic
        - p_value: P-value for hypothesis test
        - alpha: Significance level
        - significant: Whether result is statistically significant
        - ols_model: Fitted OLS model object
    """
    # Variable names for the specific horizon h
    outcome_var = ("forecast_error", outcome_variable, h)
    instrument = ("value_forecast", instrument_variable, j)

    # Check if required columns exist
    required_cols = [outcome_var, instrument]
    missing_cols = [col for col in required_cols if col not in df_pivot.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df_pivot: {missing_cols}")

    # Prepare analysis data
    analysis_data = df_pivot[required_cols].dropna().reset_index(drop=True)

    if len(analysis_data) == 0:
        raise ValueError(f"No data available for horizon h={h}")

    # Step 1: OLS regression of forecast errors on instrument
    X_robust = add_constant(analysis_data[instrument])
    y_robust = analysis_data[outcome_var]

    # OLS regression with HAC standard errors
    maxlags = h
    try:
        ols_model = OLS(y_robust, X_robust).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    except Exception as e:
        raise ValueError(
            f"OLS regression failed for {outcome_variable} at horizon {h} "
            f"with instrument {instrument_variable}. Error: {str(e)}"
        )

    # Step 2: Extract coefficients and standard errors
    ols_coefficient = ols_model.params.iloc[1]  # Coefficient on instrument
    ols_se = ols_model.bse.iloc[1]  # Standard error

    # Step 3: Statistical inference
    z_crit = norm.ppf(1 - alpha / 2)

    # Confidence intervals
    coeff_ci_lower = ols_coefficient - z_crit * ols_se
    coeff_ci_upper = ols_coefficient + z_crit * ols_se

    # Hypothesis tests
    # Test H0: ols_coefficient = 0
    z_stat = ols_coefficient / ols_se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    # Organize results
    results = {
        "horizon": h,
        "instrument_horizon": j,
        "outcome_variable": outcome_variable,
        "instrument_variable": instrument_variable,
        "n_observations": len(analysis_data),
        "ols_coefficient": ols_coefficient,
        "ols_se": ols_se,
        "coeff_ci_lower": coeff_ci_lower,
        "coeff_ci_upper": coeff_ci_upper,
        "z_stat": z_stat,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "ols_model": ols_model,
    }

    # Print results if verbose
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"STRONG EFFICIENCY TEST (h={h})")
        print(f"{'=' * 80}")
        print(f"Outcome variable:      {outcome_variable} (forecast error, horizon {h})")
        print(f"Instrument:            {instrument_variable} (forecast value, horizon {j})")
        print(f"Observations:          {len(analysis_data)}")
        print("\nCOEFFICIENT ESTIMATES:")
        print(f"OLS coefficient:       {ols_coefficient:.6f} (SE: {ols_se:.6f})")
        print(f"{100 * (1 - alpha):.0f}% CI for ols coefficient:      [{coeff_ci_lower:.6f}, {coeff_ci_upper:.6f}]")
        print("\nHYPOTHESIS TEST:")
        print("\nH0: ols coeff = 0")
        print(f"  z-statistic:         {z_stat:.4f}")
        print(f"  p-value:             {p_value:.4f}")
        print(f"  Significant at {alpha * 100:.0f}%:     {'Yes' if p_value < alpha else 'No'}")

    return results


def strong_efficiency_analysis(
    data: ForecastData,
    source: str,
    outcome_variable: str,
    outcome_metric: Literal["levels", "pop", "yoy"],
    instrument_variable: str,
    instrument_metric: Literal["levels", "pop", "yoy"],
    horizons: np.ndarray = np.arange(13),
    j: int = 2,
    frequency: Literal["Q", "M"] = "Q",
    k: int = 12,
    alpha: float = 0.05,
) -> TestResult:
    """Run strong efficiency tests across multiple forecast horizons.

    This function performs strong efficiency tests by regressing forecast errors on
    instrument variables across multiple forecast horizons. It helps assess whether
    forecasts efficiently incorporate available information.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data.
    source : str
        Source of the forecasts (e.g., 'MPR', 'OBR').
    outcome_variable : str
        Name of the outcome variable for which forecast errors are analyzed.
    outcome_metric : Literal["levels", "pop", "yoy"]
        Metric type for the outcome variable:
        - "levels": Raw values
        - "pop": Period-on-period percentage change
        - "yoy": Year-on-year percentage change
    instrument_variable : str
        Name of the instrument variable used in the regression.
    instrument_metric : Literal["levels", "pop", "yoy"]
        Metric type for the instrument variable.
    horizons : np.ndarray, optional
        Array of forecast horizons to test, by default np.arange(1, 13).
    j : int, optional
        Forecast horizon of the instrument variable, by default 2.
    frequency : Literal["Q", "M"], optional
        Frequency of the data, either quarterly ("Q") or monthly ("M"), by default "Q".
    k : int, optional
        Number of revisions used to define the outturns, by default 12.
    alpha : float, optional
        Significance level for confidence intervals, by default 0.05.

    Returns
    -------
    TestResult
        TestResult object containing a DataFrame with results for each horizon
        and metadata about the test parameters.
    """
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()

    # We first align the main table with what is used in this function
    df = filter_k(df, k)

    # Filter variables, sources and frequency
    df = (
        df[
            (
                ((df["variable"] == outcome_variable) & (df["metric"] == outcome_metric))
                | ((df["variable"] == instrument_variable) & (df["metric"] == instrument_metric))
            )
            & (df["unique_id"] == source)
            & (df["frequency"] == frequency)
        ]
        .reset_index(drop=True)
        .drop(columns=["unique_id", "metric", "frequency"])
    )

    if df.empty:
        raise ValueError(f"No data available for source '{source}'")

    # Pivot data wider
    df_pivot = df.pivot(
        index=["vintage_date_forecast"],
        columns=["variable", "forecast_horizon"],
        values=["value_forecast", "value_outturn", "forecast_error"],
    ).reset_index()

    results_list = []

    for h in horizons:
        try:
            result = strong_efficiency_test(df_pivot, outcome_variable, h, instrument_variable, j, alpha, verbose=False)
            results_list.append(result)

        except Exception as e:
            print(f"Error testing horizon h={h}: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    results_df["unique_id"] = source

    # Create metadata for result object
    metadata = {
        "test_name": "strong_efficiency_analysis",
        "parameters": {
            "unique_id": source,
            "outcome_variable": outcome_variable,
            "outcome_metric": outcome_metric,
            "instrument_variable": instrument_variable,
            "instrument_metric": instrument_metric,
            "horizons": horizons.tolist(),
            "j": j,
            "frequency": frequency,
            "k": k,
            "alpha": alpha,
        },
        "filters": {
            "unique_id": source,
            "variable": outcome_variable,
        },
        "date_range": (None, None),
    }

    return TestResult(results_df, data.id_columns, metadata)
