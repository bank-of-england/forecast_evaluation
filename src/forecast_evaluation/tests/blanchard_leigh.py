from collections import OrderedDict
from typing import Literal

import numpy as np
import pandas as pd
from linearmodels.system import SUR
from scipy.stats import norm
from statsmodels.tools import add_constant

from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import filter_k, flatten_col_name


def blanchard_leigh_efficiency_test(
    df_pivot: pd.DataFrame,
    outcome_variable: str,
    h: int,
    instrument_variable: str,
    j: int = 2,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Perform Blanchard-Leigh efficiency test using SUR regression.

    Parameters
    ----------
    df_pivot : pd.DataFrame
        Pivoted dataframe with forecast data
    outcome_variable : str
        Variable for forecast error analysis
    h : int
        Forecast horizon of outcome variable (target horizon)
    instrument_variable : str
        Variable used as instrument
    j : int, default=2
        Forecast horizon of instrument variable
    alpha : float, default=0.05
        Significance level for confidence intervals
    verbose : bool, default=True
        Whether to print detailed results

    Returns
    -------
    dict
        Dictionary with test results including coefficients, standard errors,
        ratio estimates, confidence intervals, and hypothesis test results
    """
    # Variable names for the specific horizon h
    outcome_var = ("forecast_error", outcome_variable, h)
    instrument = ("value_forecast", instrument_variable, j)
    actuals = ("value_outturn", instrument_variable, j)

    # Check if required columns exist
    required_cols = [outcome_var, instrument, actuals]
    missing_cols = [col for col in required_cols if col not in df_pivot.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df_pivot: {missing_cols}")

    # Prepare analysis data
    analysis_data = df_pivot[required_cols].dropna().reset_index(drop=True)

    if len(analysis_data) == 0:
        raise ValueError(f"No data available for horizon h={h}")

    equations = OrderedDict()

    # Step 1: Forecast error equation
    dependent = analysis_data[outcome_var]
    exog = add_constant(analysis_data[instrument])
    equations["forecast_error"] = {"dependent": flatten_col_name(dependent), "exog": flatten_col_name(exog)}

    # Step 2: Instrument equation
    dependent = analysis_data[actuals]
    exog = add_constant(analysis_data[instrument])
    equations["instrument"] = {"dependent": flatten_col_name(dependent), "exog": flatten_col_name(exog)}

    # Step 3: Estimate and extract coefficients and standard errors
    sur_model = SUR(equations)
    sur_results = sur_model.fit(
        method="ols", cov_type="kernel", kernel="bartlett", bandwidth=min(h, len(analysis_data) - 1)
    )

    first_coeff = sur_results.params.iloc[1]
    first_se = np.sqrt(sur_results.cov.iloc[1, 1])  # Standard error

    second_coeff = sur_results.params.iloc[3]
    second_se = np.sqrt(sur_results.cov.iloc[3, 3])  # Standard error

    cov = sur_results.cov.iloc[1, 3]  # Covariance between the two estimates

    # Step 4: Compute ratio using delta method
    def delta_method_ratio(coeff1, se1, coeff2, se2, cov):
        """
        Compute standard error for ratio using delta method

        Parameters:
        - coeff1, se1: coefficient and standard error of numerator
        - coeff2, se2: coefficient and standard error of denominator
        - cov: covariance between the two coefficients (default 0)
        """
        ratio = coeff1 / coeff2

        # Delta method formula for ratio r = a/b
        # Var(r) = (∂r/∂a)² Var(a) + (∂r/∂b)² Var(b) + 2(∂r/∂a)(∂r/∂b) Cov(a,b)
        # ∂r/∂a = 1/b, ∂r/∂b = -a/b²

        var_ratio = (
            (1 / coeff2) ** 2 * se1**2
            + (-coeff1 / coeff2**2) ** 2 * se2**2
            + 2 * (1 / coeff2) * (-coeff1 / coeff2**2) * cov
        )
        se_ratio = np.sqrt(var_ratio)

        return ratio, se_ratio

    ratio, ratio_se = delta_method_ratio(first_coeff, first_se, second_coeff, second_se, cov)

    # Step 5: Statistical inference
    z_crit = norm.ppf(1 - alpha / 2)

    # Confidence intervals
    ratio_ci_lower = ratio - z_crit * ratio_se
    ratio_ci_upper = ratio + z_crit * ratio_se

    # Hypothesis tests
    # Test H0: ratio = 0 (robust coefficient is zero)
    z_stat_ratio = ratio / ratio_se
    p_value_ratio = 2 * (1 - norm.cdf(abs(z_stat_ratio)))

    # Organize results
    results = {
        "horizon": h,
        "instrument_horizon": j,
        "outcome_variable": outcome_variable,
        "instrument_variable": instrument_variable,
        "n_observations": len(analysis_data),
        "first_coefficient": first_coeff,
        "first_se": first_se,
        "second_coefficient": second_coeff,
        "second_se": second_se,
        "ratio": ratio,
        "ratio_se": ratio_se,
        "ratio_ci_lower": ratio_ci_lower,
        "ratio_ci_upper": ratio_ci_upper,
        "z_stat": z_stat_ratio,
        "p_value": p_value_ratio,
        "alpha": alpha,
        "significant": p_value_ratio < alpha,
        "sur_model": sur_model,
    }

    # Print results if verbose
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"BLANCHARD-LEIGH EFFICIENCY TEST (h={h})")
        print(f"{'=' * 80}")
        print(f"Outcome variable:      {outcome_variable} (forecast error, horizon {h})")
        print(f"'Instrument':            {instrument_variable} (forecast value, horizon {j})")
        print(f"Observations:          {len(analysis_data)}")
        print("\nCOEFFICIENT ESTIMATES:")
        print(f"Forecast error equation coefficient:    {first_coeff:.6f} (SE: {first_se:.6f})")
        print(f"'Instrument' equation coefficient:       {second_coeff:.6f} (SE: {second_se:.6f})")
        print("\nRATIO ANALYSIS:")
        print(f"Ratio (Forecast_err/'Instrument'):    {ratio:.6f} (SE: {ratio_se:.6f})")
        print(f"{100 * (1 - alpha):.0f}% CI for ratio:      [{ratio_ci_lower:.6f}, {ratio_ci_upper:.6f}]")
        print("\nHYPOTHESIS TEST:")
        print("\nH0: ratio = 0 (forecast error equation coeff = 0)")
        print(f"  z-statistic:         {z_stat_ratio:.4f}")
        print(f"  p-value:             {p_value_ratio:.4f}")
        print(f"  Significant at {alpha * 100:.0f}%:     {'Yes' if p_value_ratio < alpha else 'No'}")

    return results


def blanchard_leigh_horizon_analysis(
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
    """Run Blanchard-Leigh efficiency tests across multiple forecast horizons.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data
    source : str
        Source of the forecasts (e.g., 'MPR', 'OBR')
    outcome_variable : str
        Variable for forecast error analysis
    outcome_metric : Literal["levels", "pop", "yoy"]
        Metric type for the outcome variable
    instrument_variable : str
        Variable used as instrument
    instrument_metric : Literal["levels", "pop", "yoy"]
        Metric type for the instrument variable
    horizons : np.ndarray, default=np.arange(1, 13)
        Array of forecast horizons to test
    j : int, default=2
        Forecast horizon of instrument variable
    frequency : Literal["Q", "M"], default='Q'
        Frequency of the data (quarterly or monthly)
    k : int, default=12
        Number of revisions used to define the outturn
    alpha : float, default=0.05
        Significance level for confidence intervals

    Returns
    -------
    TestResult
        TestResult object containing a DataFrame with results for each horizon
        and metadata about the test parameters.
    """

    # Validating inputs:
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()

    # Ensure horizons is a NumPy array
    horizons = np.array(horizons)

    # Filter the table for a particular k
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
        raise ValueError(
            f"No data available for variables '{outcome_variable}' with metric '{outcome_metric}',"
            f"'{instrument_variable}' with metric '{instrument_metric}' and source '{source}'"
            f" and frequency '{frequency}'."
        )

    # Pivot data wider
    df_pivot = df.pivot(
        index=["vintage_date_forecast"],
        columns=["variable", "forecast_horizon"],
        values=["value_forecast", "value_outturn", "forecast_error"],
    ).reset_index()

    results = {}

    for h in horizons:
        try:
            result = blanchard_leigh_efficiency_test(
                df_pivot, outcome_variable, h, instrument_variable, j, alpha, verbose=False
            )
            results[h] = result

        except Exception as e:
            print(f"Error testing horizon h={h}: {e}")
            results[h] = None

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df["unique_id"] = source

    # Create metadata for result object
    metadata = {
        "test_name": "blanchard_leigh_horizon_analysis",
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
        "date_range": (df_pivot["vintage_date_forecast"].min(), df_pivot["vintage_date_forecast"].max()),
    }

    return TestResult(results_df, data.id_columns, metadata)
