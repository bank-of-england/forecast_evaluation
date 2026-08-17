"""Intra-period accuracy and bias analysis for nowcasting models."""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from forecast_evaluation.data import ForecastData
from forecast_evaluation.data.utils import construct_unique_id
from forecast_evaluation.utils import filter_k, reconstruct_id_cols_from_unique_id

AXIS_COLUMNS = {"period_end": "days_to_period_end", "publication": "days_to_publication"}


def _prepare_intra_period_data(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: str = "levels",
    frequency: str = "Q",
    horizon: Optional[int] = None,
    k: Optional[int] = None,
    axis: Literal["period_end", "publication"] = "period_end",
) -> tuple[pd.DataFrame, list[str], str, int, bool]:
    """Filter and prepare data for intra-period analysis.

    Adds the requested time axis column, measured in days from the forecast
    vintage and rounded to the nearest 7 days so that weekly vintages whose
    day-of-week alignment drifts across years are binned together.

    Parameters
    ----------
    horizon : int or None
        If given, restrict to a single horizon. If ``None`` (default),
        include all horizons so the full axis range is visible.
    k : int or None
        Outturn revision index used to select the outturn. If ``None``
        (default), uses ``data.default_k`` for a ``ForecastData`` instance,
        or 0 for a DataFrame.
    axis : {'period_end', 'publication'}
        Time axis to compute.

        - ``'period_end'`` (default) gives ``days_to_period_end`` =
          ``date - vintage_date_forecast``: the distance from the forecast
          vintage to the end of the target period. Independent of when the
          outturn is published, so it lines up with calendar period
          boundaries across target periods.
        - ``'publication'`` gives ``days_to_publication`` =
          ``vintage_date_outturn - vintage_date_forecast``: the distance from
          the forecast vintage to the release of the selected outturn. This
          depends on ``k`` and on the publication lag of the series.

        The two axes differ by the publication lag, which is constant within
        a single target period but varies across periods and series.

    Returns
    -------
    tuple of (pd.DataFrame, list of str, str, int, bool)
        The prepared data (with ``unique_id`` and the axis column), the id
        columns encoded in ``unique_id`` (``source`` plus any ``extra_ids``),
        the name of the axis column, the resolved ``k`` used to filter, and
        whether ``filter_k`` substituted an earlier (not-yet-released)
        vintage for at least one row.
    """
    if axis not in AXIS_COLUMNS:
        raise ValueError(f"Unknown axis: {axis}. Use 'period_end' or 'publication'.")

    id_columns = ["source"]
    if isinstance(data, ForecastData):
        if not data.uses_intra_period_vintages:
            raise ValueError("Intra-period analysis requires a NowcastData instance.")
        if k is None:
            k = data.default_k
        id_columns = list(data.id_columns) if data.id_columns else id_columns
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

    # Raw DataFrames are accepted here, and only the horizon filter needs the column,
    # so a frame without it stays usable as long as no horizon is requested.
    if "target_minus_vintage" in df.columns:
        df = df.assign(horizon=lambda d: d["target_minus_vintage"].astype(int))
    elif horizon is not None:
        raise ValueError(
            "Column 'target_minus_vintage' not found, so horizons cannot be identified. "
            "Pass a ForecastData instance, or omit the 'horizon' argument."
        )

    # ForecastData already builds unique_id from source plus any extra_ids;
    # only raw DataFrames need it constructing.
    if "unique_id" not in df.columns:
        missing = [col for col in id_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found; cannot identify distinct forecasts.")
        df["unique_id"] = construct_unique_id(df, id_columns)

    k = k if k is not None else 0
    # filter_k() is a no-op (returns data unfiltered/unsubstituted) when there is
    # no outturn-vintage tracking at all, in which case a fallback can never occur.
    k_filter_is_no_op = "latest_vintage" in df.columns and df["latest_vintage"].isna().all()
    df = filter_k(df, k)

    mask = (df["variable"] == variable) & (df["metric"] == metric) & (df["frequency"] == frequency)
    if horizon is not None:
        mask = mask & (df["horizon"] == horizon)

    df = df.loc[mask].copy()

    if df.empty:
        raise ValueError(
            f"No data for variable='{variable}', metric='{metric}', "
            f"frequency='{frequency}'" + (f", horizon={horizon}" if horizon is not None else "")
        )

    # Computed on the final filtered rows only, so a substitution in a row
    # excluded by the mask above is never mistaken for a fallback in the result.
    k_fallback_used = False if k_filter_is_no_op else bool((df["k"] != k).any())

    # Rounded to the nearest 7 days so that weekly vintages whose
    # day-of-week alignment drifts across years are binned together.
    axis_column = AXIS_COLUMNS[axis]
    end_point = df["date"] if axis == "period_end" else df["vintage_date_outturn"]
    raw_days = (pd.to_datetime(end_point) - pd.to_datetime(df["vintage_date_forecast"])).dt.days
    df[axis_column] = (raw_days / 7).round().astype(int) * 7

    return df, id_columns, axis_column, k, k_fallback_used


def compute_intra_period_accuracy(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    horizon: Optional[int] = None,
    statistic: Literal["rmse", "mae"] = "rmse",
    k: Optional[int] = None,
    axis: Literal["period_end", "publication"] = "period_end",
) -> pd.DataFrame:
    """Compute forecast accuracy grouped by a within-period time axis.

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
    horizon : int or None
        Forecast horizon to evaluate. ``None`` includes all horizons.
    statistic : str
        'rmse' or 'mae'.
    k : int or None
        Outturn revision index used to select the outturn. If ``None``
        (default), uses ``data.default_k`` for a ``ForecastData`` instance.
    axis : {'period_end', 'publication'}
        Time axis to group by. ``'period_end'`` (default) measures days from
        the forecast vintage to the end of the target period;
        ``'publication'`` measures days to the release of the selected
        outturn. The two differ by the publication lag.

    Returns
    -------
    pd.DataFrame
        DataFrame with the identifier columns (``source`` plus any
        ``extra_ids``), ``unique_id``, the axis column
        (``days_to_period_end`` or ``days_to_publication``), ``value`` and
        ``se``. ``se`` is the standard error of the statistic.
    """
    df, id_columns, axis_column, resolved_k, k_fallback_used = _prepare_intra_period_data(data, variable, metric, frequency, horizon, k, axis)

    grouped = df.groupby(["unique_id", axis_column])["forecast_error"]

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

    result = reconstruct_id_cols_from_unique_id(result, id_columns)
    result.attrs["target_dates"] = pd.to_datetime(df["date"]).drop_duplicates().sort_values().tolist()
    result.attrs["k"] = resolved_k
    result.attrs["k_fallback_used"] = k_fallback_used
    return result.sort_values(["unique_id", axis_column], ascending=[True, False]).reset_index(drop=True)


def compute_intra_period_bias(
    data: Union[pd.DataFrame, ForecastData],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    horizon: Optional[int] = None,
    k: Optional[int] = None,
    axis: Literal["period_end", "publication"] = "period_end",
) -> pd.DataFrame:
    """Compute forecast bias (mean error) grouped by a within-period time axis.

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
    horizon : int or None
        Forecast horizon to evaluate. ``None`` includes all horizons.
    k : int or None
        Outturn revision index used to select the outturn. If ``None``
        (default), uses ``data.default_k`` for a ``ForecastData`` instance.
    axis : {'period_end', 'publication'}
        Time axis to group by. ``'period_end'`` (default) measures days from
        the forecast vintage to the end of the target period;
        ``'publication'`` measures days to the release of the selected
        outturn. The two differ by the publication lag.

    Returns
    -------
    pd.DataFrame
        DataFrame with the identifier columns (``source`` plus any
        ``extra_ids``), ``unique_id``, the axis column
        (``days_to_period_end`` or ``days_to_publication``), ``value`` and
        ``se``. ``se`` is the standard error of the mean error.
    """
    df, id_columns, axis_column, resolved_k, k_fallback_used = _prepare_intra_period_data(data, variable, metric, frequency, horizon, k, axis)

    grouped = df.groupby(["unique_id", axis_column])["forecast_error"]
    mean_err = grouped.mean()
    se_mean = grouped.apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
    result = pd.DataFrame({"value": mean_err, "se": se_mean}).reset_index()
    result = reconstruct_id_cols_from_unique_id(result, id_columns)
    result.attrs["target_dates"] = pd.to_datetime(df["date"]).drop_duplicates().sort_values().tolist()
    result.attrs["k"] = resolved_k
    result.attrs["k_fallback_used"] = k_fallback_used
    return result.sort_values(["unique_id", axis_column], ascending=[True, False]).reset_index(drop=True)
