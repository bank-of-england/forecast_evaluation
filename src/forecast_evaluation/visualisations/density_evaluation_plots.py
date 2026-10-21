"""Plots for density forecast evaluation: PIT histogram and ECDF, sharpness, CRPS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from forecast_evaluation.utils import clean_unique_id
from forecast_evaluation.visualisations.theme import create_themed_figure, set_integer_xaxis

if TYPE_CHECKING:
    from forecast_evaluation.tests.results import TestResult

RS_TABULATED_KS_5PCT = 1.339  # Rossi-Sekhposyan (2019) 5% KS critical value for one-step PITs


def _frame(result) -> pd.DataFrame:
    return result.to_df() if hasattr(result, "to_df") else result.copy()


def _pick(df: pd.DataFrame, column: str, value, label: str):
    """Filter ``df`` on ``column == value``; with ``value=None`` require a single value."""
    if value is None:
        values = df[column].unique()
        if len(values) != 1:
            raise ValueError(f"Multiple {label}s found: {list(values)}. Please specify '{label}'.")
        value = values[0]
    out = df[df[column] == value]
    if out.empty:
        raise ValueError(f"No data for {label} {value!r}.")
    return out, value


def _select(df, variable, metric, source=None, require_source=True):
    df, variable = _pick(df, "variable", variable, "variable")
    df, metric = _pick(df, "metric", metric, "metric")
    if require_source or source is not None:
        df, source = _pick(df, "unique_id", source, "source")
    return df, variable, metric, source


def _horizon_axes(horizons):
    n = len(horizons)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = create_themed_figure(nrows=nrows, ncols=ncols)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _finish(fig, axes, return_plot):
    if return_plot:
        return fig, axes
    plt.show()
    return None


def plot_pit_histogram(
    pit: pd.DataFrame | TestResult,
    variable: str | None = None,
    source: str | None = None,
    metric: str | None = None,
    horizons: list[int] | None = None,
    bins: int = 10,
    confidence: float = 0.95,
    return_plot: bool = False,
):
    """PIT histogram (equal-width bins, density scale) per horizon with a uniform line and binomial bands.

    Parameters
    ----------
    pit : pd.DataFrame or TestResult
        Output of ``compute_pit``.
    variable, source, metric : str, optional
        Required when the data hold more than one value (``source`` is a ``unique_id``).
    horizons : list of int, optional
        Horizons to show (one panel each); default all.
    bins : int, optional
        Number of equal-width bins on [0, 1]. Default 10.
    confidence : float, optional
        Coverage of the pointwise binomial band for each bin count under uniformity. Default 0.95.
    return_plot : bool, optional
        Return ``(fig, axes)`` instead of showing the figure.
    """
    df, variable, metric, source = _select(_frame(pit), variable, metric, source)
    horizons = sorted(df["horizon"].unique()) if horizons is None else list(horizons)
    fig, axes = _horizon_axes(horizons)
    edges = np.linspace(0, 1, int(bins) + 1)
    width = 1 / int(bins)
    for ax, h in zip(axes, horizons):
        values = df.loc[df["horizon"] == h, "pit"].dropna().to_numpy()
        n = values.size
        ax.hist(values, bins=edges, density=True, alpha=0.7, edgecolor="white", label="PIT")
        ax.axhline(1.0, color="black", linewidth=1, label="Uniform")
        if n:
            lo, hi = stats.binom.ppf([(1 - confidence) / 2, (1 + confidence) / 2], n, width) / (n * width)
            ax.axhspan(lo, hi, color="grey", alpha=0.2, label=f"{confidence:.0%} binomial band")
        ax.set_xlim(0, 1)
        ax.set_title(f"Horizon {h} (n={n})")
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.suptitle(f"PIT histogram: {variable} - {clean_unique_id(source)} ({metric})")
    return _finish(fig, axes, return_plot)


def plot_pit_ecdf(
    pit: pd.DataFrame | TestResult,
    variable: str | None = None,
    source: str | None = None,
    metric: str | None = None,
    horizons: list[int] | None = None,
    rs_result: pd.DataFrame | TestResult | None = None,
    level: int = 5,
    return_plot: bool = False,
):
    """Empirical CDF of the PITs per horizon against the 45-degree line, with Rossi-Sekhposyan bands.

    The band is ``r +/- c / sqrt(n)`` with ``c`` the KS critical value. With ``rs_result``
    (``rossi_sekhposyan_test`` output) ``c`` is the bootstrap value ``ks_cv_{level}`` for the
    horizon; otherwise the tabulated one-step 5% value 1.339 is used (``level`` must be 5).
    """
    df, variable, metric, source = _select(_frame(pit), variable, metric, source)
    cvs = {}
    if rs_result is not None:
        rs, *_ = _select(_frame(rs_result), variable, metric, source)
        cvs = dict(zip(rs["horizon"], rs[f"ks_cv_{level}"]))
    elif level != 5:
        raise ValueError("Without rs_result only the tabulated 5% band is available (level=5).")

    horizons = sorted(df["horizon"].unique()) if horizons is None else list(horizons)
    fig, axes = _horizon_axes(horizons)
    grid = np.linspace(0, 1, 201)
    for ax, h in zip(axes, horizons):
        values = np.sort(df.loc[df["horizon"] == h, "pit"].dropna().to_numpy())
        n = values.size
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, label="45-degree line")
        if n:
            ax.step(values, np.arange(1, n + 1) / n, where="post", label="PIT ECDF")
            c = cvs.get(h, RS_TABULATED_KS_5PCT if rs_result is None else np.nan)
            if np.isfinite(c):
                half = c / np.sqrt(n)
                ax.fill_between(grid, grid - half, grid + half, color="grey", alpha=0.2, label=f"{level}% R&S band")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Horizon {h} (n={n})")
        ax.set_xlabel("r")
    axes[0].set_ylabel("Share of PITs <= r")
    axes[0].legend()
    fig.suptitle(f"PIT ECDF: {variable} - {clean_unique_id(source)} ({metric})")
    return _finish(fig, axes, return_plot)


def _by_horizon_lines(df, value, ylabel, title, band=None, return_plot=False):
    fig, ax = create_themed_figure()
    for uid, g in df.groupby("unique_id", sort=True):
        g = g.sort_values("horizon")
        (line,) = ax.plot(g["horizon"], g[value], marker="o", label=clean_unique_id(uid))
        if band is not None:
            ax.fill_between(g["horizon"], g[value] - g[band], g[value] + g[band], color=line.get_color(), alpha=0.2)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    set_integer_xaxis(ax)
    ax.legend()
    return _finish(fig, ax, return_plot)


def plot_sharpness_by_horizon(
    sharpness: pd.DataFrame | TestResult,
    variable: str | None = None,
    metric: str | None = None,
    interval: str = "central_90",
    sources: list[str] | None = None,
    return_plot: bool = False,
):
    """Median interval width by horizon (+/- one standard deviation), one line per source."""
    df, variable, metric, _ = _select(_frame(sharpness), variable, metric, require_source=False)
    df = df[df["interval"] == interval]
    if sources is not None:
        df = df[df["unique_id"].isin(sources)]
    if df.empty:
        raise ValueError(f"No sharpness results for interval {interval!r}.")
    lo, hi = df["lower_level"].iloc[0], df["upper_level"].iloc[0]
    title = f"Sharpness: {interval} ({lo:g}-{hi:g}) width by horizon\n{variable} ({metric})"
    return _by_horizon_lines(df, "median_width", "Median width (+/- SD)", title, "std_width", return_plot)


def plot_crps_by_horizon(
    crps: pd.DataFrame | TestResult,
    variable: str | None = None,
    metric: str | None = None,
    sources: list[str] | None = None,
    return_plot: bool = False,
):
    """Mean (weighted) CRPS by horizon, one line per source (``crps_analysis`` output)."""
    df, variable, metric, _ = _select(_frame(crps), variable, metric, require_source=False)
    if sources is not None:
        df = df[df["unique_id"].isin(sources)]
    metadata = crps._metadata if hasattr(crps, "to_df") else {}
    weight = metadata.get("parameters", {}).get("weight", "none")
    label = "CRPS" if weight == "none" else f"Weighted CRPS ({weight})"
    title = f"{label} by horizon\n{variable} ({metric})"
    return _by_horizon_lines(df, "crps", f"Mean {label}", title, None, return_plot)


def plot_cumulative_crps(
    crps: pd.DataFrame | TestResult,
    horizon: int | None = None,
    variable: str | None = None,
    metric: str | None = None,
    sources: list[str] | None = None,
    benchmark: str | None = None,
    return_plot: bool = False,
):
    """Cumulative CRPS over forecast vintages for one horizon (``compute_crps`` output).

    With ``benchmark`` (a ``unique_id``) the cumulative difference ``benchmark - source`` is shown,
    on the vintages both share; positive values favour the source.
    """
    df, variable, metric, _ = _select(_frame(crps), variable, metric, require_source=False)
    df, horizon = _pick(df, "horizon", horizon, "horizon")
    if sources is not None:
        df = df[df["unique_id"].isin([*sources, *([benchmark] if benchmark else [])])]
    wide = df.pivot_table(index="vintage_date_forecast", columns="unique_id", values="crps").sort_index()
    if benchmark is not None:
        if benchmark not in wide.columns:
            raise ValueError(f"Benchmark {benchmark!r} not found.")
        wide = wide.dropna()
        wide = wide.drop(columns=benchmark).rsub(wide[benchmark], axis=0)
        ylabel = f"Cumulative CRPS difference ({clean_unique_id(benchmark)} - source)"
    else:
        ylabel = "Cumulative CRPS"
    fig, ax = create_themed_figure()
    for uid in wide.columns:
        ax.plot(wide.index, wide[uid].fillna(0).cumsum(), label=clean_unique_id(uid))
    if benchmark is not None:
        ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Forecast vintage")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cumulative CRPS, horizon {horizon}\n{variable} ({metric})")
    ax.legend()
    return _finish(fig, ax, return_plot)
