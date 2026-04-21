"""Intra-period accuracy and bias analysis for nowcasting models."""

from typing import Literal, Union

import numpy as np
import pandas as pd

from forecast_evaluation.data import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData


def _prepare_intra_period_data(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: str = "levels",
    frequency: str = "Q",
    forecast_horizon: int = 0,
) -> pd.DataFrame:
    """Filter and prepare data for intra-period analysis.

    Computes ``days_to_publication`` as the number of days between the
    forecast vintage and the outturn vintage. Assigns a ``vintage_rank``
    within each (source, variable, date) group and maps it to the median
    ``days_to_publication`` so that each point on the x-axis represents
    the same release number.
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

    mask = (
        (df["variable"] == variable)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
        & (df["forecast_horizon"] == forecast_horizon)
    )
    df = df.loc[mask].copy()

    if df.empty:
        raise ValueError(
            f"No data for variable='{variable}', metric='{metric}', "
            f"frequency='{frequency}', forecast_horizon={forecast_horizon}"
        )

    # Days from forecast vintage to the end of the target period
    df["days_to_publication"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["vintage_date_forecast"])).dt.days

    # Rank vintages chronologically within each (source, variable, date) group.
    # This aligns the "1st release", "2nd release", etc. across periods.
    df["vintage_rank"] = (
        df.groupby(["source", "variable", "date"])["days_to_publication"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    # Map each rank to its median days_to_publication across all periods,
    # so the x-axis stays in "days" but groups the same release together.
    median_day = df.groupby(["source", "vintage_rank"])["days_to_publication"].median()
    df["median_days_to_publication"] = df.set_index(["source", "vintage_rank"]).index.map(median_day).values

    return df


def compute_intra_period_accuracy(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    forecast_horizon: int = 0,
    statistic: Literal["rmse", "mae"] = "rmse",
) -> pd.DataFrame:
    """Compute forecast accuracy by vintage release within a period.

    Groups vintages by their chronological rank within each period and
    computes the chosen accuracy statistic. The x-axis value is the median
    ``days_to_publication`` for each rank.

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
    forecast_horizon : int
        Forecast horizon to evaluate.
    statistic : str
        'rmse' or 'mae'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``source``, ``days_to_publication``, ``value``.
    """
    df = _prepare_intra_period_data(data, variable, metric, frequency, forecast_horizon)

    if statistic == "rmse":
        agg = df.groupby(["source", "median_days_to_publication"])["forecast_error"].apply(
            lambda x: np.sqrt(np.mean(x**2))
        )
    elif statistic == "mae":
        agg = df.groupby(["source", "median_days_to_publication"])["forecast_error"].apply(lambda x: np.mean(np.abs(x)))
    else:
        raise ValueError(f"Unknown statistic: {statistic}. Use 'rmse' or 'mae'.")

    result = agg.reset_index()
    result.columns = ["source", "days_to_publication", "value"]
    return result.sort_values(["source", "days_to_publication"], ascending=[True, False]).reset_index(drop=True)


def compute_intra_period_bias(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    forecast_horizon: int = 0,
) -> pd.DataFrame:
    """Compute forecast bias (mean error) by vintage release within a period.

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
    forecast_horizon : int
        Forecast horizon to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``source``, ``days_to_publication``, ``value``.
    """
    df = _prepare_intra_period_data(data, variable, metric, frequency, forecast_horizon)

    agg = df.groupby(["source", "median_days_to_publication"])["forecast_error"].mean()
    result = agg.reset_index()
    result.columns = ["source", "days_to_publication", "value"]
    return result.sort_values(["source", "days_to_publication"], ascending=[True, False]).reset_index(drop=True)
