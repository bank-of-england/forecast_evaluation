"""Visualisation functions for density forecasts."""

import numbers
import warnings
from collections.abc import Sequence
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_evaluation.data import DensityForecastData
from forecast_evaluation.data.utils import compute_target_minus_vintage
from forecast_evaluation.utils import clean_unique_id
from forecast_evaluation.visualisations.theme import create_themed_figure

_DEFAULT_QUANTILES = (0.16, 0.5, 0.84)
_QUANTILE_TOLERANCE = 1e-9


def plot_density_vintage(
    data: DensityForecastData,
    variable: str,
    vintage_date: str | pd.Timestamp,
    forecast_source: Optional[list[str]] = None,
    outturn_start_date: str | pd.Timestamp | None = None,
    frequency: Optional[Literal["Q", "M"]] = None,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    *,
    quantiles: Optional[Sequence[float]] = None,
    return_plot: bool = False,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Plot density forecasts for one vintage with quantile bands and outturns.

    Parameters
    ----------
    data : DensityForecastData
        Object containing density forecasts and outturns.
    variable : str
        Variable to plot.
    vintage_date : str or pd.Timestamp
        Forecast vintage to plot.
    forecast_source : list of str, optional
        Sources to include. If None, all sources are included.
    outturn_start_date : str or pd.Timestamp, optional
        First outturn date to display (inclusive). If None, use all available dates.
    frequency : {"Q", "M"} or None, default None
        Frequency to filter on. If None, infer it from the outturns.
    metric : {"levels", "pop", "yoy"}, default "levels"
        Metric to plot. Density forecasts must already be stored in this metric.
    quantiles : sequence of float, optional
        Keyword-only quantile levels in [0, 1]. Defaults to ``(0.16, 0.5, 0.84)``. A copy is
        sorted in ascending order. Missing levels are replaced by the nearest stored level,
        with a warning naming the requested and used values. Two or more levels draw one shaded
        band between the outermost levels; inner levels are faint dashed lines. The solid line is
        the 0.5 level when requested, or the middle requested level for an odd-length list. An
        even-length list without 0.5 has no solid central line.
    return_plot : bool, default False
        Keyword-only. If True, return the matplotlib figure and axes instead of showing the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        ``(fig, ax)`` if ``return_plot`` is True; otherwise None.

    Raises
    ------
    ValueError
        If ``quantiles`` is empty, non-finite, outside [0, 1], duplicated, or if no density
        forecasts match the selection or frequency cannot be inferred.

    Warns
    -----
    UserWarning
        When a requested quantile is not stored and the nearest stored level is used.

    Notes
    -----
    Quantiles are resolved against the selected density table using a tolerance of 1e-9; the
    stored values are used for plotting. Outturns come from the latest available outturn vintage
    and are split before and after the forecast vintage. A numeric ``forecast_source`` list in
    [0, 1] is treated as deprecated positional ``quantiles`` for one release; pass
    ``quantiles=`` instead.
    """
    legacy_quantiles = (
        isinstance(forecast_source, (list, tuple))
        and bool(forecast_source)
        and all(
            isinstance(level, numbers.Real)
            and not isinstance(level, (bool, np.bool_))
            and np.isfinite(level)
            and 0 <= level <= 1
            for level in forecast_source
        )
    )
    if legacy_quantiles:
        if quantiles is not None:
            raise TypeError("A numeric forecast_source list conflicts with quantiles; pass source names as strings.")
        quantiles = forecast_source
        forecast_source = None
        warnings.warn(
            "A numeric forecast_source list is deprecated as positional quantiles; pass quantiles= instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        requested_quantiles = list(_DEFAULT_QUANTILES if quantiles is None else quantiles)
    except TypeError:
        raise ValueError("quantiles must be a non-empty sequence of numbers in [0, 1].") from None
    if not requested_quantiles:
        raise ValueError("quantiles must contain at least one level.")
    if any(isinstance(level, (bool, np.bool_)) for level in requested_quantiles):
        raise ValueError("Quantile levels must be numeric values in [0, 1], not booleans.")
    try:
        requested_quantiles = [float(level) for level in requested_quantiles]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("quantiles must be a sequence of numeric levels in [0, 1].") from None

    non_finite = [level for level in requested_quantiles if not np.isfinite(level)]
    if non_finite:
        raise ValueError(f"Quantile levels must be finite; got {non_finite}.")
    out_of_range = [level for level in requested_quantiles if not 0 <= level <= 1]
    if out_of_range:
        raise ValueError(f"Quantile levels must lie in [0, 1]; got {out_of_range}.")

    requested_quantiles.sort()
    for lower, upper in zip(requested_quantiles, requested_quantiles[1:]):
        if np.isclose(lower, upper, rtol=0, atol=_QUANTILE_TOLERANCE):
            raise ValueError(f"Duplicate quantile requests: {lower:g} and {upper:g}.")

    if frequency is None:
        inferred = data._raw_outturns["frequency"].unique()
        if len(inferred) != 1:
            raise ValueError(
                f"Could not infer a unique frequency from data; found: {list(inferred)}. "
                "Please specify the 'frequency' argument explicitly."
            )
        frequency = inferred[0]

    # add a check here
    vintage_date = pd.to_datetime(vintage_date)

    # filter forecasts
    forecasts_filtered = data.copy()

    forecasts_filtered.filter(
        variables=variable,
        metrics=metric,
        frequencies=frequency,
        sources=forecast_source,
        start_vintage=vintage_date,
        end_vintage=vintage_date,
        filter_point_forecasts=False,
        filter_density_forecasts=True,
    )
    forecasts_filtered = forecasts_filtered.density_forecasts.copy()
    if forecasts_filtered.empty:
        raise ValueError(
            f"No density forecasts found for variable='{variable}', metric='{metric}', "
            f"vintage_date='{vintage_date:%Y-%m-%d}', frequency='{frequency}'."
        )

    # filter outturns (they are not filtered with filter())(we select the last vintage only)
    outturns = data._outturns.copy()
    min_date = outturn_start_date if outturn_start_date is not None else outturns["date"].min()

    outturns = outturns[
        (outturns["vintage_date"] == outturns["vintage_date"].max())
        & (outturns["variable"].isin(forecasts_filtered["variable"].unique()))
        & (outturns["metric"] == metric)
        & (outturns["date"] <= forecasts_filtered["date"].max())
        & (outturns["date"] >= min_date)
    ].copy()

    forecasts_filtered = clean_unique_id(forecasts_filtered)

    available_quantiles = np.sort(forecasts_filtered["quantile"].astype(float).unique())
    resolutions = []
    missing_quantiles = []
    for requested in requested_quantiles:
        matching = available_quantiles[
            np.isclose(available_quantiles, requested, rtol=0, atol=_QUANTILE_TOLERANCE)
        ]
        if matching.size:
            used = float(matching[0])
        else:
            used = float(available_quantiles[np.argmin(np.abs(available_quantiles - requested))])
            missing_quantiles.append((requested, used))

        previous = next(
            (
                previous_requested
                for previous_requested, previous_used in resolutions
                if np.isclose(previous_used, used, rtol=0, atol=_QUANTILE_TOLERANCE)
            ),
            None,
        )
        if previous is not None:
            raise ValueError(
                f"Requested quantile levels {previous:g} and {requested:g} both resolve to the same stored "
                f"level {used:g}."
            )
        resolutions.append((requested, used))

    for requested, used in missing_quantiles:
        warnings.warn(
            f"Requested quantile level {requested:g} is not stored; using closest available level {used:g}.",
            UserWarning,
            stacklevel=2,
        )

    quantiles_used = sorted(used for _, used in resolutions)

    fig, ax = create_themed_figure()

    # Plot each source separately with quantile bands
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts_filtered["unique_id"].unique())))

    central_quantile = next(
        (
            used
            for requested, used in resolutions
            if np.isclose(requested, 0.5, rtol=0, atol=_QUANTILE_TOLERANCE)
        ),
        None,
    )
    if central_quantile is None and len(quantiles_used) % 2:
        central_quantile = quantiles_used[len(quantiles_used) // 2]

    for idx, forecast_id in enumerate(forecasts_filtered["unique_id"].unique()):
        source_df = forecasts_filtered[forecasts_filtered["unique_id"] == forecast_id].sort_values("date")
        color = colors[idx]

        # Group by date to get quantiles for each date
        for quantile in quantiles_used:
            quantile_data = source_df[source_df["quantile"] == quantile].sort_values("date")

            if not quantile_data.empty:
                if central_quantile is not None and np.isclose(
                    quantile, central_quantile, rtol=0, atol=_QUANTILE_TOLERANCE
                ):
                    label = (
                        f"{forecast_id} (median)"
                        if np.isclose(quantile, 0.5, rtol=0, atol=_QUANTILE_TOLERANCE)
                        else f"{forecast_id} (q={quantile:g})"
                    )
                    ax.plot(
                        quantile_data["date"],
                        quantile_data["value"],
                        label=label,
                        alpha=0.8,
                        color=color,
                    )
                else:
                    ax.plot(quantile_data["date"], quantile_data["value"], linestyle="--", alpha=0.1, color=color)

        if len(quantiles_used) >= 2:
            lower_q = source_df[source_df["quantile"] == quantiles_used[0]].sort_values("date")
            upper_q = source_df[source_df["quantile"] == quantiles_used[-1]].sort_values("date")
            if not lower_q.empty and not upper_q.empty:
                ax.fill_between(
                    lower_q["date"],
                    lower_q["value"],
                    upper_q["value"],
                    alpha=0.1,
                    color=color,
                    label=f"{forecast_id} ({quantiles_used[0]:g}-{quantiles_used[-1]:g})",
                )

    # Overlay the latest outturn vintage, split at the selected forecast vintage.
    outturns_data = compute_target_minus_vintage(outturns.assign(vintage_date=vintage_date)).sort_values("date")
    if not outturns_data.empty:
        solid_outturns = outturns_data[outturns_data["target_minus_vintage"] < 0]
        dashed_outturns = outturns_data[outturns_data["target_minus_vintage"] >= 0]

        if not solid_outturns.empty:
            ax.plot(
                solid_outturns["date"],
                solid_outturns["value"],
                color="darkblue",
                marker="o",
                markersize=3,
                label="Outturns (latest vintage, before forecast)",
            )
        if not dashed_outturns.empty:
            ax.plot(
                dashed_outturns["date"],
                dashed_outturns["value"],
                color="darkblue",
                marker="o",
                markersize=3,
                linestyle="--",
                label="Outturns (latest vintage, post forecast)",
            )

    ax.set_title(f"{variable} [{frequency}] - {metric} - Vintage: {vintage_date.date()}")
    ax.set_ylabel(f"{variable} ({metric})")
    ax.legend()

    # Return or show the plot
    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
