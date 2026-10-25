"""CRPS, quantile-weighted CRPS and sharpness of fitted density forecasts (MTP paper metrics).

Weighted CRPS follows Gneiting and Ranjan (2011): ``2 * int_0^1 v(tau) QS_tau(F^-1(tau), y) dtau``,
evaluated in value space as ``2 * int v(F(x)) (1{y < x} - F(x)) (x - y) f(x) dx`` with the exact
fitted cdf/pdf on a per-forecast grid that includes the outturn as a node (``v = 1`` gives the CRPS).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from forecast_evaluation.tests.density_calibration import GROUP_COLS, _eval_table, _metadata
from forecast_evaluation.tests.results import TestResult

WEIGHTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": np.ones_like,
    "centre": lambda t: t * (1 - t),
    "tails": lambda t: (2 * t - 1) ** 2,
    "right": lambda t: t**2,
    "left": lambda t: (1 - t) ** 2,
    "heavy_tails": lambda t: (2 * t - 1) ** 4,
}
# Paper widths: 30%, 50% and 90% central intervals plus the two outer tail bands.
DEFAULT_INTERVALS = {
    "central_30": (0.35, 0.65),
    "central_50": (0.25, 0.75),
    "central_90": (0.05, 0.95),
    "lower_tail": (0.05, 0.25),
    "upper_tail": (0.75, 0.95),
}
_QUANTILE_999 = np.arange(1, 1000) / 1000

Weight = Literal["none", "centre", "tails", "left", "right", "heavy_tails"] | Callable[[np.ndarray], np.ndarray]


def _weight_fn(weight: Weight) -> Callable[[np.ndarray], np.ndarray]:
    if callable(weight):
        return weight
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {list(WEIGHTS)} or a callable of tau; got {weight!r}.")
    return WEIGHTS[weight]


def density_scores(
    fitted, y, weight: Weight = "none", *, grid=None, n_points: int = 2000, tail: float = 1e-6
) -> np.ndarray:
    """Quantile-weighted CRPS of density ``i`` of ``fitted`` at outturn ``y[i]``.

    Parameters
    ----------
    fitted : FittedDensityForecasts
    y : array-like, shape (N,)
    weight : str or callable, optional
        Quantile weight ``v(tau)``: "none" (CRPS), "centre" tau(1-tau), "tails" (2tau-1)^2,
        "right" tau^2, "left" (1-tau)^2, "heavy_tails" (2tau-1)^4, or a vectorised callable.
    grid : None, "quantile_999" or array-like, optional
        None: ``n_points`` values from ``ppf(tail)`` to ``ppf(1 - tail)`` plus ``y`` (clipped to that
        range) as a node (default); the truncated tails add at most about ``2 * tail * |y - ppf|``.
        "quantile_999": MATLAB quantile-grid average over tau = 0.001, ..., 0.999. An array
        (M,) or (N, M) is used as given (MATLAB value-grid parity). Intended for parity only.
    """
    v = _weight_fn(weight)
    y = np.asarray(y, dtype=float).ravel()
    n = len(fitted)
    if y.size != n:
        raise ValueError(f"y must have one outturn per density ({n}); got {y.size}.")
    yc = y[:, None]

    if isinstance(grid, str):
        if grid != "quantile_999":
            raise ValueError("grid must be None, 'quantile_999' or an array of values.")
        q = fitted.ppf(_QUANTILE_999)
        return 2 * np.mean(v(_QUANTILE_999) * ((yc < q) - _QUANTILE_999) * (q - yc), axis=1)

    if grid is None:
        if int(n_points) < 2 or not 0 < tail < 0.5:
            raise ValueError("n_points must be >= 2 and tail in (0, 0.5).")
        ends = fitted.ppf([tail, 1 - tail])
        base = ends[:, :1] + (ends[:, 1:] - ends[:, :1]) * np.linspace(0, 1, int(n_points))[None, :]
        # Node at the outturn (kink of the integrand); clipped so a far outturn adds no long, poorly integrated segment.
        node = np.clip(yc, ends[:, :1], ends[:, 1:])
        x = np.sort(np.concatenate([base, node], axis=1), axis=1)
    else:
        x = np.asarray(grid, dtype=float)
        if x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[0] != n):
            raise ValueError("grid array must be (M,) or (N, M).")
        x = np.broadcast_to(x, (n, x.shape[-1]))

    F = fitted.cdf(x)
    integrand = v(F) * ((x > yc) - F) * (x - yc) * fitted.pdf(x)
    return 2 * trapezoid(integrand, x, axis=1)


def interval_widths(fitted, intervals: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Exact interval widths ``ppf(upper) - ppf(lower)``, one column per named interval."""
    intervals = DEFAULT_INTERVALS if intervals is None else intervals
    if not intervals:
        raise ValueError("intervals must not be empty.")
    for name, (lo, hi) in intervals.items():
        if not 0 < lo < hi < 1:
            raise ValueError(f"Interval {name!r} must satisfy 0 < lower < upper < 1; got {(lo, hi)}.")
    levels = np.array([lv for pair in intervals.values() for lv in pair])
    q = fitted.ppf(levels)
    return pd.DataFrame({name: q[:, 2 * i + 1] - q[:, 2 * i] for i, name in enumerate(intervals)})


def _scored_table(fitted, k, weight, grid, n_points, tail, same_date_range, data):
    table, data, k = _eval_table(fitted, k, same_date_range, data)
    sub = fitted.take(table["row"].to_numpy())
    table = table.assign(crps=density_scores(sub, table["outturn"], weight, grid=grid, n_points=n_points, tail=tail))
    return table, data, k


def compute_crps(
    fitted,
    k: int | Literal["latest"] | None = None,
    *,
    weight: Weight = "none",
    grid=None,
    n_points: int = 2000,
    tail: float = 1e-6,
    same_date_range: bool = True,
    data=None,
) -> TestResult:
    """Per-forecast (weighted) CRPS against the outturn selected by ``k`` (see ``density_scores``).

    Returns
    -------
    TestResult
        One row per (density, outturn): group columns, ``date``, ``vintage_date_forecast``,
        ``outturn`` and ``crps``, plus the id columns.
    """
    table, data, k = _scored_table(fitted, k, weight, grid, n_points, tail, same_date_range, data)
    df = table[[*GROUP_COLS, "date", "vintage_date_forecast", "outturn", "crps"]]
    metadata = {
        "test_name": "compute_crps",
        "parameters": {"k": k, "weight": weight, "grid": grid, "same_date_range": same_date_range},
        "density_method": fitted.method,
        "date_range": (df["vintage_date_forecast"].min(), df["vintage_date_forecast"].max()),
    }
    return TestResult(df, data.id_columns, metadata)


def crps_analysis(
    fitted,
    k: int | Literal["latest"] | None = None,
    *,
    weight: Weight = "none",
    grid=None,
    n_points: int = 2000,
    tail: float = 1e-6,
    same_date_range: bool = True,
    data=None,
) -> TestResult:
    """Mean (weighted) CRPS by variable, unique_id, metric, frequency and horizon.

    Returns
    -------
    TestResult
        Columns: group columns, ``crps`` (mean), ``n_observations``, ``start_date`` and
        ``end_date`` (forecast vintages), plus the id columns.
    """
    table, data, k = _scored_table(fitted, k, weight, grid, n_points, tail, same_date_range, data)
    df = (
        table.groupby(GROUP_COLS, sort=True, dropna=False)
        .agg(
            crps=("crps", "mean"),
            n_observations=("crps", "count"),
            start_date=("vintage_date_forecast", "min"),
            end_date=("vintage_date_forecast", "max"),
        )
        .reset_index()
    )
    metadata = _metadata(
        "crps_analysis", fitted, df, k=k, weight=weight, grid=grid, same_date_range=same_date_range
    )
    return TestResult(df, data.id_columns, metadata)


def sharpness_analysis(
    fitted,
    k: int | Literal["latest"] | None = None,
    *,
    intervals: dict[str, tuple[float, float]] | None = None,
    same_date_range: bool = True,
    data=None,
) -> TestResult:
    """Interval widths of the fitted densities that have an outturn, summarised by group.

    ``intervals`` maps names to (lower, upper) levels; the default is the paper's 30% and 50%
    central intervals and tail bands (0.05-0.25, 0.75-0.95) plus the 90% central interval.

    Returns
    -------
    TestResult
        Long format: group columns, ``interval``, ``lower_level``, ``upper_level``, ``mean_width``,
        ``median_width``, ``std_width``, ``n_observations``, ``start_date``, ``end_date``.
    """
    intervals = DEFAULT_INTERVALS if intervals is None else intervals
    table, data, k = _eval_table(fitted, k, same_date_range, data)
    widths = interval_widths(fitted.take(table["row"].to_numpy()), intervals)
    long = pd.concat([table[[*GROUP_COLS, "vintage_date_forecast"]], widths], axis=1).melt(
        id_vars=[*GROUP_COLS, "vintage_date_forecast"], var_name="interval", value_name="width"
    )
    df = (
        long.groupby([*GROUP_COLS, "interval"], sort=False, dropna=False)
        .agg(
            mean_width=("width", "mean"),
            median_width=("width", "median"),
            std_width=("width", "std"),
            n_observations=("width", "count"),
            start_date=("vintage_date_forecast", "min"),
            end_date=("vintage_date_forecast", "max"),
        )
        .reset_index()
    )
    df.insert(df.columns.get_loc("interval") + 1, "lower_level", df["interval"].map(lambda s: intervals[s][0]))
    df.insert(df.columns.get_loc("lower_level") + 1, "upper_level", df["interval"].map(lambda s: intervals[s][1]))
    df = df.sort_values([*GROUP_COLS, "lower_level", "upper_level"]).reset_index(drop=True)
    metadata = _metadata(
        "sharpness_analysis", fitted, df, k=k, intervals=dict(intervals), same_date_range=same_date_range
    )
    return TestResult(df, data.id_columns, metadata)
