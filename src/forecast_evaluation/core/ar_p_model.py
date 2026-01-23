from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from forecast_evaluation.core.transformations import transform_forecast_to_levels, transform_series
from forecast_evaluation.data import ForecastData


def build_ar_p_model(
    data: ForecastData,
    variable: str,
    metric: Literal["levels", "diff", "pop", "yoy"],
    frequency: Literal["Q", "M"] = "Q",
    forecast_periods: int = 13,
    *,
    estimation_start_date: pd.Timestamp = pd.Timestamp("1997-07-01"),
):
    """
    Build an AR(p) forecast model for the specified variable, metric, and frequency.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing outturn data.
    variable : str
        The variable to build the model for (e.g., 'gdpkp').
    metric : str
        The metric to build the model for (e.g., 'levels').
    frequency : Literal["Q", "M"], optional
        The frequency of the data ('Q' for quarterly, 'M' for monthly). Default is 'Q'.
    forecast_periods : int, optional
        Number of periods to forecast ahead. Default is 13.
    estimation_start_date : pd.Timestamp, optional
        The date from which to start including data for model estimation. Default is '1997-07-01'.
        Set to None to include all data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the AR(p) forecasts.

    Notes
    -----
    The function fits an AR(p) model with Student t-distributed errors using Maximum Likelihood Estimation.
    The optimal lag length p is selected based on the Bayesian Information Criterion (BIC).
    """
    # Filter data for the specified variable and frequency
    df = data._raw_outturns[
        (data._raw_outturns["variable"] == variable) & (data._raw_outturns["frequency"] == frequency)
    ].copy()

    # Check data availability after filtering
    if df.empty:
        raise ValueError(f"No outturn data available for variable '{variable}' and frequency '{frequency}'.")

    # Transform outturns depending on metric
    df = transform_series(df, transform=metric, frequency=frequency)

    # Ensure date and vintage_date is in datetime format
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "vintage_date" in df.columns:
        df["vintage_date"] = pd.to_datetime(df["vintage_date"])

    # Filter by estimation_start_date if provided
    if estimation_start_date is not None:
        df = df[df["date"] > pd.to_datetime(estimation_start_date)]

    # Group by unique combinations
    grouped = df.groupby(["variable", "metric", "frequency", "vintage_date"])

    if forecast_periods < 0:
        raise ValueError("forecast_periods must be >= 0")

    # Collect forecasts
    forecasts = []

    date_freq = "QE" if frequency == "Q" else "ME"
    date_offset = pd.offsets.QuarterEnd() if frequency == "Q" else pd.offsets.MonthEnd()

    for (grp_variable, grp_metric, grp_frequency, grp_vintage_date), group in grouped:
        group = group.sort_values("date")

        try:
            # Get the latest date and value
            latest_date = group["date"].max()
            latest_value = group[group["date"] == latest_date]["value"].iloc[0]

            # Check if we have enough data points
            if len(group) < 8:
                print(
                    f"Not enough data points for variable {grp_variable}, vintage_date {grp_vintage_date}. Using AR(1)"
                )
                optimal_lag = 1
            else:
                # Prepare data for AR model (no differencing)
                model_data = group.set_index("date")
                model_data.index = pd.DatetimeIndex(model_data.index, freq=date_freq)

                # Select optimal lag using BIC with MLE estimation (on original series)
                max_possible_lag = 2
                best_bic = np.inf
                optimal_lag = 1

                for lag in range(1, max_possible_lag + 1):
                    try:
                        # Fit AR(lag) model with t-distribution using MLE on original series
                        model_result = fit_ar_t_mle(model_data["value"], lag)
                        if model_result["success"] and model_result["bic"] < best_bic:
                            best_bic = model_result["bic"]
                            optimal_lag = lag
                    except Exception as e:
                        print(f"  AR({lag}): Failed to fit - {e}")
                        continue

            # Include the latest value in the forecast
            forecasts.append(
                {
                    "date": latest_date,
                    "variable": grp_variable,
                    "metric": grp_metric,
                    "vintage_date": grp_vintage_date,
                    "value": latest_value,
                    "frequency": grp_frequency,
                    "optimal_lag": optimal_lag,
                    "source": "baseline ar(p) model",
                    "forecast_horizon": -1,
                }
            )

            # Prepare data for final AR model fitting (no differencing)
            model_data = group.set_index("date")
            model_data.index = pd.DatetimeIndex(model_data.index, freq=date_freq)

            # Fit final AR model with t-distribution using MLE on original series
            final_model = fit_ar_t_mle(model_data["value"], optimal_lag)

            if not final_model["success"]:
                print(f"  Warning: Final model fit failed for {grp_variable}, {grp_vintage_date}. Using fallback.")
                # Fallback to simple forecasts
                forecast_values = np.full(forecast_periods, latest_value)
            else:
                # Generate forecasts directly (no integration needed)
                forecast_values = generate_ar_t_forecasts(model_data["value"], final_model, steps=forecast_periods)

            # Generate forecast dates
            forecast_dates = pd.date_range(start=latest_date + date_offset, periods=forecast_periods, freq=date_freq)

            for period, (date, value) in enumerate(zip(forecast_dates, forecast_values), start=0):
                forecasts.append(
                    {
                        "date": date,
                        "variable": grp_variable,
                        "metric": grp_metric,
                        "vintage_date": grp_vintage_date,
                        "value": value,
                        "frequency": grp_frequency,
                        "optimal_lag": optimal_lag,
                        "source": "baseline ar(p) model",
                        "forecast_horizon": period,
                    }
                )

        except Exception as e:
            print(f"Skipping group {grp_variable}, {grp_frequency}, {grp_vintage_date} due to error: {e}")
            continue

    forecast_df = pd.DataFrame(forecasts)

    if not forecast_df.empty:
        forecast_df = forecast_df.sort_values(["variable", "frequency", "vintage_date", "date"])

    return forecast_df


def fit_ar_t_mle(y, p):
    """
    Fit AR(p) model with Student t-distributed errors using Maximum Likelihood Estimation.

    This function estimates an autoregressive model of order p where the error terms
    follow a Student t-distribution instead of the standard normal distribution.
    The t-distribution provides more robust estimation in the presence of outliers and heavy-tailed data.

    Model specification:
    y_t = c + φ₁*y_{t-1} + φ₂*y_{t-2} + ... + φₚ*y_{t-p} + ε_t
    where ε_t ~ t(0, σ², ν) with degrees of freedom ν

    Parameters:
    -----------
    y : array-like
        Time series data (univariate)
    p : int
        Order of the autoregressive model (number of lags)

    Returns:
    --------
    dict
        Dictionary containing model results:
        - 'constant' : float
            Estimated constant term (c)
        - 'ar_coeffs' : array
            Estimated AR coefficients [φ₁, φ₂, ..., φₚ]
        - 'scale' : float
            Scale parameter (σ) of the t-distribution
        - 'df' : float
            Degrees of freedom (ν) of the t-distribution
        - 'log_likelihood' : float
            Log-likelihood value at optimum
        - 'bic' : float
            Bayesian Information Criterion
        - 'success' : bool
            Whether optimization converged successfully
        - 'error' : str (optional)
            Error message if optimization failed

    Notes:
    ------
    - Uses L-BFGS-B optimization method for parameter estimation
    - Initial parameter estimates obtained via OLS regression
    - Scale parameter constrained to be positive (> 0.001)
    - Degrees of freedom constrained to be > 2 and ≤ 30
    """
    y = np.array(y)
    n = len(y)

    # Create lagged variables
    Y = y[p:]
    X = np.ones((n - p, p + 1))  # Include constant term

    # Create design matrix with lagged values
    for i in range(p):
        X[:, i + 1] = y[p - 1 - i : n - 1 - i]

    # Initial parameter estimates using OLS
    try:
        ols_params = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - np.dot(X, ols_params)
        initial_scale = np.std(residuals)
        initial_df = 5.0  # Conservative starting point for degrees of freedom

        initial_params = np.concatenate([ols_params, [initial_scale, initial_df]])
    except Exception:
        # Fallback to simple initialization
        initial_params = np.concatenate([np.zeros(p + 1), [1.0, 5.0]])

    # Parameter bounds
    bounds = [(None, None)] * (p + 1) + [(0.001, None), (2.1, 30)]  # scale > 0, df > 2

    def neg_log_likelihood(params):
        """
        Calculate the negative log-likelihood for AR(p) model with Student t-distributed errors.

        This function computes the negative log-likelihood used in maximum likelihood estimation
        of autoregressive models with Student t-distributed error terms. The negative is used
        because scipy.optimize.minimize performs minimization, while we want to maximize the
        likelihood function.

        Parameters
        ----------
        params : array-like
            Parameter vector containing:
            - params[0] : float
                Constant term (c) of the AR model
            - params[1:p+1] : array
                Raw AR coefficients [φ₁, φ₂, ..., φₚ] (before stationarity transformation)
            - params[p+1] : float
                Scale parameter (σ) of the t-distribution (must be > 0)
            - params[p+2] : float
                Degrees of freedom (ν) of the t-distribution (must be > 2)

        Returns
        -------
        float
            Negative log-likelihood value. Returns np.inf if:
            - Parameter vector has incorrect length
            - Scale parameter ≤ 0
            - Degrees of freedom ≤ 2
            - Numerical issues in likelihood calculation

        Notes
        -----
        Mathematical Specification:
        The AR(p) model with Student t errors is:
        y_t = c + φ₁*y_{t-1} + φ₂*y_{t-2} + ... + φₚ*y_{t-p} + ε_t
        where ε_t ~ t(0, σ², ν)

        Log-likelihood Calculation:
        For each residual r_t = y_t - ŷ_t, the log-likelihood contribution is:
        ℓ_t = log(pdf_t(r_t/σ, ν)) - log(σ)

        Total log-likelihood: L = Σ ℓ_t

        The function returns -L for minimization.
        """
        if len(params) != p + 3:  # AR coeffs + constant + scale + df
            return np.inf

        const = params[0]
        ar_coeffs = transform_ar_coeffs(params[1 : p + 1])
        scale = params[p + 1]
        df = params[p + 2]

        if scale <= 0 or df <= 2:  # Invalid parameters
            return np.inf

        # Calculate residuals
        y_pred = const + np.dot(X[:, 1:], ar_coeffs)
        residuals = Y - y_pred

        # Student t log-likelihood
        log_likelihood = np.sum(stats.t.logpdf(residuals / scale, df) - np.log(scale))

        return -log_likelihood

    # Optimize
    try:
        result = minimize(neg_log_likelihood, initial_params, bounds=bounds, method="L-BFGS-B")

        if result.success:
            params = result.x
            const = params[0]
            ar_coeffs = transform_ar_coeffs(params[1 : p + 1])
            scale = params[p + 1]
            df = params[p + 2]

            # Calculate BIC
            log_likelihood = -result.fun
            bic = -2 * log_likelihood + (p + 3) * np.log(n - p)

            return {
                "constant": const,
                "ar_coeffs": ar_coeffs,
                "scale": scale,
                "df": df,
                "log_likelihood": log_likelihood,
                "bic": bic,
                "success": True,
            }
        else:
            return {"success": False}

    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_ar_t_forecasts(y, model_result, steps):
    """
    Generate out-of-sample forecasts from an AR(p) model with Student t-distributed errors.

    Parameters
    ----------
    y : array-like
        The time series data used for model fitting (should be stationary if differencing was applied).
    model_result : dict
        The result dictionary returned by `fit_ar_t_mle`, containing AR coefficients, constant, etc.
    steps : int
        Number of periods ahead to forecast.

    Returns
    -------
    np.ndarray
        Array of forecasted values for the specified number of steps ahead.

    Notes
    -----
    - Forecasts are conditional means (point forecasts) from the AR(p) model.
    - If model fitting failed, returns a flat forecast using the mean of y.
    - No uncertainty or random draws are added; only the conditional mean is returned.
    """
    if not model_result["success"]:
        # Fallback to simple mean forecast
        return np.full(steps, np.mean(y))

    y = np.array(y)
    p = len(model_result["ar_coeffs"])
    const = model_result["constant"]
    ar_coeffs = model_result["ar_coeffs"]

    # Initialize forecast list
    forecasts = []

    # Use last p observations as starting values
    history = list(y[-p:])

    for step in range(steps):
        # Point forecast (conditional mean)
        forecast = const + np.dot(ar_coeffs, history[-p:])

        # Add small random component based on t-distribution
        # For point forecasts, we use the mean (which is 0 for t-distribution with df > 1)
        # But we could add uncertainty if desired

        forecasts.append(forecast)
        history.append(forecast)

    return np.array(forecasts)


# --- Helper functions ---
def stationary_bound(y, lower, upper):
    """
    Transform unbounded parameter to bounded range using sigmoid transformation.

    This function applies a sigmoid-like transformation to map any real number to a
    specified bounded interval. It's used to ensure AR coefficients remain within
    the stationarity region during optimization.

    Parameters
    ----------
    y : float or array-like
        Unbounded input parameter(s) to be transformed
    lower : float
        Lower bound of the target interval
    upper : float
        Upper bound of the target interval

    Returns
    -------
    float or array-like
        Transformed parameter(s) bounded within [lower, upper]

    Notes
    -----
    The transformation uses the formula:
    x = lower + (upper - lower) * (1/(1 + exp(-y)))

    This ensures:
    - The output is always in the range (lower, upper)
    - The transformation is smooth and differentiable
    - As y → ±∞, x approaches the bounds asymptotically
    """
    x = lower + (upper - lower) * (1 / (1 + np.exp(-y)))
    return x


def transform_ar_coeffs(ar_coeffs):
    """
    Transform AR coefficients to ensure stationarity constraints are satisfied.

    This function applies appropriate transformations to autoregressive coefficients
    to ensure the resulting AR model is stationary. The transformations are
    differentiable, making them suitable for gradient-based optimization methods
    like L-BFGS-B used in maximum likelihood estimation.

    Parameters
    ----------
    ar_coeffs : array-like
        Raw AR coefficients from optimization (potentially unbounded).
        Length determines the AR order:
        - Length 1: AR(1) model
        - Length 2: AR(2) model
        - Length > 2: Not supported (raises ValueError)

    Returns
    -------
    numpy.ndarray
        Transformed AR coefficients that satisfy stationarity constraints.
        Same length as input but with values constrained to ensure stability.

    Raises
    ------
    ValueError
        If more than 2 AR coefficients are provided. Only AR(1) and AR(2)
        models are currently supported.

    Notes
    -----
    Stationarity Constraints by Model Order:

    **AR(1) Model:** y_t = φ₁*y_{t-1} + ε_t
    - Constraint: |φ₁| < 1 (coefficient must be within unit circle)
    - Transformation: φ₁ = tanh(raw_φ₁) * 0.99
    - Properties: tanh maps ℝ → (-1, 1), ensuring stability

    **AR(2) Model:** y_t = φ₁*y_{t-1} + φ₂*y_{t-2} + ε_t
    - Constraints (stationarity triangle):
        * φ₁ + φ₂ < 1    (sum constraint)
        * φ₂ - φ₁ < 1    (difference constraint)
        * |φ₂| < 1       (absolute value constraint)
    - Transformation sequence:
        1. φ₂ = tanh(raw_φ₂) * 0.99 → ensures |φ₂| < 1
        2. φ₁ = stationary_bound(raw_φ₁, φ₂-1, 1-φ₂) → ensures other constraints

    Mathematical Background
    -----------------------
    For an AR(p) model to be stationary, all roots of the characteristic polynomial
    must lie outside the unit circle:

    1 - φ₁*z - φ₂*z² - ... - φₚ*zᵖ = 0

    For AR(1): |φ₁| < 1 is necessary and sufficient.
    For AR(2): The constraints form a triangular region in (φ₁, φ₂) space.
    """
    if len(ar_coeffs) == 1:
        ar_coeffs = np.tanh(ar_coeffs) * 0.99  # For AR(1), tanh ensures stability (range [-1, 1])
    elif len(ar_coeffs) == 2:
        b = np.tanh(ar_coeffs[1]) * 0.99
        a = stationary_bound(ar_coeffs[0], b - 1, 1 - b)
        ar_coeffs = np.array([a, b])
    elif len(ar_coeffs) > 2:
        raise ValueError("AR coefficients transformation only implemented for AR(1) and AR(2).")

    return ar_coeffs


def add_ar_p_forecasts(
    data: ForecastData,
    variable: str | Iterable[str] | None = None,
    metric: Literal["levels", "diff", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] | Iterable[Literal["Q", "M"]] | None = None,
    forecast_periods: int = 13,
    *,
    estimation_start_date: pd.Timestamp = pd.Timestamp("1997-07-01"),
):
    """
    Wrapper function to build AR(p) model and add forecasts to ForecastData.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing outturn data.
    variable : str | Iterable[str] | None
        Variable name (e.g., 'gdpkp') or multiple variable names.
        If None, builds forecasts for all variable x frequency combinations present in the outturns.
    metric : str, optional
        The metric to build the model for. Default is 'levels'.
    frequency : Literal["Q", "M"] | Iterable[Literal["Q", "M"]] | None, optional
        Frequency (or multiple frequencies) to build forecasts for.
        If None, frequencies are inferred from the outturns' `frequency` column for the requested variable(s).
    forecast_periods : int, optional
        Number of periods to forecast ahead. Default is 13.
    estimation_start_date : pd.Timestamp, optional
        The date from which to start including data for model estimation. Default is '1997-07-01'.
        Set to None to include all data.

    Returns
    -------
    None
        The function modifies the ForecastData object in place by adding AR(p) forecasts.
    """
    if variable is None:
        variables: list[str] | None = None
    else:
        variables = [variable] if isinstance(variable, str) else list(variable)
        if len(variables) == 0:
            raise ValueError("variable must be None, a non-empty string, or a non-empty iterable of strings")

    if frequency is None:
        frequencies: list[str] | None = None
    else:
        frequencies = [frequency] if isinstance(frequency, str) else list(frequency)
        if len(frequencies) == 0:
            raise ValueError("frequency must be None, a frequency string, or a non-empty iterable of frequencies")
        invalid = sorted(set(frequencies) - {"Q", "M"})
        if invalid:
            raise ValueError(f"Unsupported frequency value(s): {invalid}. Supported: 'Q', 'M'.")

    # Filter variable and frequency combinations from outturns
    outturns = data._raw_outturns
    subset = outturns[["variable", "frequency"]].dropna()
    if variables is not None:
        subset = subset[subset["variable"].isin(variables)]
    if frequencies is not None:
        subset = subset[subset["frequency"].isin(frequencies)]

    pairs = list(subset.drop_duplicates().itertuples(index=False, name=None))
    if len(pairs) == 0:
        raise ValueError("No (variable, frequency) combinations available in outturns for the requested filters")

    forecast_frames = [
        build_ar_p_model(
            data=data,
            variable=var,
            metric=metric,
            frequency=freq,
            forecast_periods=forecast_periods,
            estimation_start_date=estimation_start_date,
        )
        for (var, freq) in pairs
    ]

    ar_forecasts = pd.concat(forecast_frames, ignore_index=True) if len(forecast_frames) > 1 else forecast_frames[0]

    # Transform forecasts to levels
    ar_forecasts_in_levels = transform_forecast_to_levels(
        outturns=data._raw_outturns,
        forecasts=ar_forecasts,
    )

    # Append AR(p) forecasts to existing forecasts
    data.add_forecasts(ar_forecasts_in_levels)

    return None
