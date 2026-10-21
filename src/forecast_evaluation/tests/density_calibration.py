"""Calibration tests for fitted density forecasts: PIT, Knueppel (2015) and Mitchell-Weale.

The statistics port the MTP MATLAB functions ``alpha0_1234_NW`` (Knueppel), ``MW_alpha0_1234_NW`` and
``freqTestInCensoredRegion`` (Mitchell-Weale), with Newey-West covariances from ``nwX``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from forecast_evaluation.core.main_table import build_main_table
from forecast_evaluation.tests._hac import nw_hac
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import ensure_consistent_date_range, filter_k

GROUP_COLS = ["variable", "unique_id", "metric", "frequency", "horizon"]
_MIN_OBS = 3


# -----------------------
# Array-level statistics
# -----------------------


def _moment_stat(s: np.ndarray, lags, prewhite: bool) -> float:
    """Odd (1st, 3rd) plus even (2nd, 4th) moment Wald statistics of standardised PITs ``s``."""
    if s.size < _MIN_OBS:
        return np.nan
    z = np.column_stack([s, s**2 - 1, s**3, s**4 - 1.8])
    total = 0.0
    for cols in ([0, 2], [1, 3]):
        zz = z[:, cols]
        y = zz.sum(axis=0) / np.sqrt(zz.shape[0])
        try:
            total += float(y @ np.linalg.solve(nw_hac(zz, prewhite, lags)[0], y))
        except np.linalg.LinAlgError:
            return np.nan
    return total


def knuppel_statistic(pit, lags=-1, prewhite: bool = False) -> tuple[float, float]:
    """Knueppel (2015) moment test of uniform PITs: ``(stat, p_value)``, chi-square(4); NaNs dropped."""
    x = np.asarray(pit, dtype=float)
    x = x[~np.isnan(x)]
    stat = _moment_stat(np.sqrt(12) * (x - 0.5), lags, prewhite)
    return stat, float(stats.chi2.sf(stat, 4))


def mitchell_weale_statistic(pit, alpha: float = 0.10, lags=-1, prewhite: bool = False) -> dict[str, float]:
    """Mitchell-Weale censored moment test plus coverage-frequency test, chi-square(5).

    PITs inside ``[alpha/2, 1 - alpha/2]`` enter the Knueppel moments (rescaled to that region);
    the share inside is tested against ``1 - alpha``. NaNs are dropped.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    z_l, z_u = alpha / 2, 1 - alpha / 2
    x = np.asarray(pit, dtype=float)
    x = x[~np.isnan(x)]
    inside = (x >= z_l) & (x <= z_u)
    s = np.sqrt(12 / (z_u - z_l) ** 2) * (x[inside] - 0.5 * (z_l + z_u))
    stat_uncensored = _moment_stat(s, lags, prewhite)

    stat_frequency = np.nan
    if x.size >= _MIN_OBS:
        d = inside.astype(float) - (z_u - z_l)
        with np.errstate(divide="ignore", invalid="ignore"):
            stat_frequency = float((d.sum() / np.sqrt(x.size)) ** 2 / nw_hac(d, prewhite, lags)[0][0, 0])
    stat = stat_uncensored + stat_frequency
    return {
        "stat_uncensored": stat_uncensored,
        "stat_frequency": stat_frequency,
        "stat": stat,
        "p_value": float(stats.chi2.sf(stat, 5)),
    }


# -----------------------
# Evaluation table
# -----------------------


def build_density_eval_table(fitted, k: int | Literal["latest"] | None = None, *, data=None) -> pd.DataFrame:
    """Match each fitted density with its outturn and compute the PIT.

    Outturns are matched exactly as for point forecasts (``build_main_table`` then ``filter_k``),
    so ``k`` selects the outturn revision; ``"latest"`` uses each date's latest outturn vintage.

    Parameters
    ----------
    fitted : FittedDensityForecasts
        Fitted densities (from ``DensityForecastData.fit_parametric_density``).
    k : int, "latest" or None, optional
        Outturn revision; None uses ``data.default_k``.
    data : DensityForecastData, optional
        Source of outturns; defaults to ``fitted.data``.

    Returns
    -------
    pd.DataFrame
        Main-table columns (``date``, ``variable``, ``vintage_date_forecast``, ``vintage_date_outturn``,
        ``unique_id``, id columns, ``metric``, ``frequency``, ``forecast_horizon``,
        ``target_minus_vintage``, ``k``, ``latest_vintage``) plus ``horizon``, ``outturn``,
        ``row`` (position in ``fitted``) and ``pit`` (exact fitted cdf at the outturn).
    """
    data, k = _resolve(fitted, k, data)
    keys = fitted.keys.assign(value=np.arange(len(fitted), dtype=float))
    table = build_main_table(
        keys,
        data.outturns,
        list(data.id_columns),
        frequency=keys["frequency"].iloc[0],
        outturn_vintages=data._outturn_vintages,
    )
    if k == "latest":
        if not table["latest_vintage"].isna().all():
            table = table[table["vintage_date_outturn"] == table["latest_vintage"]]
    else:
        table = filter_k(table, k)
    if table.empty:
        raise ValueError("No outturns match the fitted densities.")

    rows = table["value_forecast"].to_numpy().astype(int)
    outturn = table["value_outturn"].to_numpy(dtype=float)
    pit = fitted.take(rows).cdf(outturn[:, None])[:, 0]
    table = table.drop(columns=["value_forecast", "value_outturn", "forecast_error"]).assign(
        horizon=table["target_minus_vintage"].astype(int), outturn=outturn, row=rows, pit=pit
    )
    return table.sort_values(["unique_id", "variable", "horizon", "vintage_date_forecast"]).reset_index(drop=True)


def _resolve(fitted, k, data):
    data = fitted.data if data is None else data
    if data is None:
        raise ValueError("fitted has no source data; pass data=.")
    k = data.default_k if k is None else k
    if not (k == "latest" or isinstance(k, (int, np.integer))):
        raise ValueError(f"k must be an int, 'latest' or None; got {k!r}.")
    return data, k


def _eval_table(fitted, k, same_date_range, data):
    data, k = _resolve(fitted, k, data)
    table = build_density_eval_table(fitted, k, data=data)
    if same_date_range and table["unique_id"].nunique() > 1:
        table = ensure_consistent_date_range(table)
    return table, data, k


def _metadata(test_name, fitted, df, **parameters) -> dict:
    return {
        "test_name": test_name,
        "parameters": parameters,
        "density_method": fitted.method,
        "date_range": (df["start_date"].min(), df["end_date"].max()) if "start_date" in df else None,
    }


def _grouped_test(fitted, k, same_date_range, data, stat_fn):
    table, data, k = _eval_table(fitted, k, same_date_range, data)
    rows = []
    for key, g in table.groupby(GROUP_COLS, sort=True, dropna=False):
        pit = g.sort_values("vintage_date_forecast")["pit"].to_numpy()
        rows.append(
            {
                **dict(zip(GROUP_COLS, key)),
                **stat_fn(pit),
                "n_observations": int(np.isfinite(pit).sum()),
                "start_date": g["vintage_date_forecast"].min(),
                "end_date": g["vintage_date_forecast"].max(),
            }
        )
    return pd.DataFrame(rows), data, k


# -----------------------
# Public tests
# -----------------------


def compute_pit(
    fitted, k: int | Literal["latest"] | None = None, *, same_date_range: bool = True, data=None
) -> TestResult:
    """Probability integral transforms of the outturns under the fitted densities.

    Returns
    -------
    TestResult
        One row per (density, outturn): ``variable``, ``unique_id``, ``metric``, ``frequency``,
        ``horizon``, ``date``, ``vintage_date_forecast``, ``vintage_date_outturn``, ``k``,
        ``outturn`` and ``pit``, plus the id columns.
    """
    table, data, k = _eval_table(fitted, k, same_date_range, data)
    cols = [*GROUP_COLS, "date", "vintage_date_forecast", "vintage_date_outturn", "k", "outturn", "pit"]
    df = table[cols]
    metadata = {
        "test_name": "compute_pit",
        "parameters": {"k": k, "same_date_range": same_date_range},
        "density_method": fitted.method,
        "date_range": (df["vintage_date_forecast"].min(), df["vintage_date_forecast"].max()),
    }
    return TestResult(df, data.id_columns, metadata)


def knuppel_test(
    fitted,
    k: int | Literal["latest"] | None = None,
    *,
    lags=-1,
    prewhite: bool = False,
    same_date_range: bool = True,
    data=None,
) -> TestResult:
    """Knueppel (2015) test of PIT uniformity via its first four moments, HAC (Newey-West) robust.

    Grouped by variable, unique_id, metric, frequency and horizon; PITs are ordered by forecast
    vintage. ``lags=-1`` selects the Newey-West lag automatically (Andrews 1991).

    Returns
    -------
    TestResult
        Columns: group columns, ``stat``, ``p_value`` (chi-square, 4 df), ``n_observations``,
        ``start_date`` and ``end_date`` (forecast vintages), plus the id columns. Groups with
        fewer than 3 PITs give NaN statistics.
    """

    def stat_fn(pit):
        stat, p = knuppel_statistic(pit, lags, prewhite)
        return {"stat": stat, "p_value": p}

    df, data, k = _grouped_test(fitted, k, same_date_range, data, stat_fn)
    metadata = _metadata(
        "knuppel_test", fitted, df, k=k, lags=lags, prewhite=prewhite, same_date_range=same_date_range
    )
    return TestResult(df, data.id_columns, metadata)


def mitchell_weale_test(
    fitted,
    k: int | Literal["latest"] | None = None,
    *,
    alpha: float = 0.10,
    lags=-1,
    prewhite: bool = False,
    same_date_range: bool = True,
    data=None,
) -> TestResult:
    """Mitchell-Weale censored Knueppel test on ``[alpha/2, 1 - alpha/2]`` plus coverage test.

    Returns
    -------
    TestResult
        Columns: group columns, ``stat_uncensored``, ``stat_frequency``, ``stat``, ``p_value``
        (chi-square, 5 df), ``n_observations``, ``start_date`` and ``end_date``, plus the id
        columns. Groups with fewer than 3 PITs (in the region) give NaN statistics.
    """
    df, data, k = _grouped_test(
        fitted, k, same_date_range, data, lambda pit: mitchell_weale_statistic(pit, alpha, lags, prewhite)
    )
    metadata = _metadata(
        "mitchell_weale_test",
        fitted,
        df,
        k=k,
        alpha=alpha,
        lags=lags,
        prewhite=prewhite,
        same_date_range=same_date_range,
    )
    return TestResult(df, data.id_columns, metadata)
