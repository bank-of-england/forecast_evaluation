"""Intra-period accuracy and bias analysis for nowcasting models."""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from forecast_evaluation.data import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData


def _prepare_intra_period_data(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: str = "levels",
    frequency: str = "Q",
    forecast_horizon: Optional[int] = None,
) -> pd.DataFrame:
    """Filter and prepare data for intra-period analysis.

    Computes ``days_to_target`` as the number of days between the
    forecast vintage and the end of the target period.

    Parameters
    ----------
    forecast_horizon : int or None
        If given, restrict to a single horizon. If ``None`` (default),
        include all horizons so the full days-to-target range is visible.
    """
    if isinstance(data, ForecastData):
        if not isinstance(data, NowcastData):
            raise ValueError("Intra-period analysis requires a NowcastData instance.")
        df = data.df.copy()
    elif hasattr(data, "df"):
        df = data.df.copy()
    else:
        df = data.copy()

    for col in ("vintage_date_forecast", "vintage_date_outturn"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Pass a ForecastData instance "
                "or a DataFrame with vintage_date_forecast and vintage_date_outturn columns."
            )

    mask = (df["variable"] == variable) & (df["metric"] == metric) & (df["frequency"] == frequency)
    if forecast_horizon is not None:
        mask = mask & (df["forecast_horizon"] == forecast_horizon)

    df = df.loc[mask].copy()

    if df.empty:
        raise ValueError(
            f"No data for variable='{variable}', metric='{metric}', "
            f"frequency='{frequency}'"
            + (f", forecast_horizon={forecast_horizon}" if forecast_horizon is not None else "")
        )

    # Days from forecast vintage to the end of the target period,
    # rounded to the nearest 7 days so that weekly vintages whose
    # day-of-week alignment drifts across years are binned together.
    raw_days = (pd.to_datetime(df["date"]) - pd.to_datetime(df["vintage_date_forecast"])).dt.days
    df["days_to_target"] = (raw_days / 7).round().astype(int) * 7

    return df


def compute_intra_period_accuracy(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    forecast_horizon: Optional[int] = None,
    statistic: Literal["rmse", "mae"] = "rmse",
) -> pd.DataFrame:
    """Compute forecast accuracy grouped by days to target.

    Parameters
    ----------
    data : ForecastData or pd.DataFrame
        Data with ``vintage_date_forecast`` and ``vintage_date_outturn`` columns.
    variable : str
        Variable to analyse.
    metric : str
        Metric to analyse.
    frequency : str
        Data frequency ('Q' or 'M').
    forecast_horizon : int or None
        Forecast horizon to evaluate. ``None`` includes all horizons.
    statistic : str
        'rmse' or 'mae'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``source``, ``days_to_target``, ``value``, ``se``.
        ``se`` is the standard error of the statistic.
    """
    df = _prepare_intra_period_data(data, variable, metric, frequency, forecast_horizon)

    grouped = df.groupby(["source", "days_to_target"])["forecast_error"]

    if statistic == "rmse":
        mse = grouped.apply(lambda x: np.mean(x**2))
        rmse = np.sqrt(mse)
        # Delta method: SE(RMSE) ≈ std(e²) / (2 * sqrt(n) * RMSE)
        se_rmse = grouped.apply(
            lambda x: (
                np.std(x**2, ddof=1) / (2 * np.sqrt(len(x)) * np.sqrt(np.mean(x**2)))
                if len(x) > 1 and np.mean(x**2) > 0
                else np.nan
            )
        )
        result = pd.DataFrame({"value": rmse, "se": se_rmse}).reset_index()
    elif statistic == "mae":
        mae = grouped.apply(lambda x: np.mean(np.abs(x)))
        se_mae = grouped.apply(lambda x: np.std(np.abs(x), ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
        result = pd.DataFrame({"value": mae, "se": se_mae}).reset_index()
    else:
        raise ValueError(f"Unknown statistic: {statistic}. Use 'rmse' or 'mae'.")

    return result.sort_values(["source", "days_to_target"], ascending=[True, False]).reset_index(drop=True)


def compute_intra_period_bias(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    forecast_horizon: Optional[int] = None,
) -> pd.DataFrame:
    """Compute forecast bias (mean error) grouped by days to target.

    Parameters
    ----------
    data : ForecastData or pd.DataFrame
        Data with ``vintage_date_forecast`` and ``vintage_date_outturn`` columns.
    variable : str
        Variable to analyse.
    metric : str
        Metric to analyse.
    frequency : str
        Data frequency ('Q' or 'M').
    forecast_horizon : int or None
        Forecast horizon to evaluate. ``None`` includes all horizons.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``source``, ``days_to_target``, ``value``, ``se``.
        ``se`` is the standard error of the mean error.
    """
    df = _prepare_intra_period_data(data, variable, metric, frequency, forecast_horizon)

    grouped = df.groupby(["source", "days_to_target"])["forecast_error"]
    mean_err = grouped.mean()
    se_mean = grouped.apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
    result = pd.DataFrame({"value": mean_err, "se": se_mean}).reset_index()
    return result.sort_values(["source", "days_to_target"], ascending=[True, False]).reset_index(drop=True)
