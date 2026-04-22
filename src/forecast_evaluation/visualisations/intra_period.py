from typing import TYPE_CHECKING, Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from forecast_evaluation.tests.intra_period import compute_intra_period_accuracy, compute_intra_period_bias
from forecast_evaluation.visualisations.theme import create_themed_figure

if TYPE_CHECKING:
    from forecast_evaluation.data.ForecastData import ForecastData


def _add_quarter_boundaries(ax, days_min, days_max):
    """Add dashed vertical lines at quarter boundaries (~91-day intervals)."""
    quarter_days = 91
    boundary = quarter_days * (int(days_min) // quarter_days)
    first = True
    while boundary <= days_max:
        if days_min <= boundary <= days_max:
            label = "Quarter boundary" if first else None
            ax.axvline(x=boundary, color="grey", linestyle="--", linewidth=1.5, alpha=0.6, label=label)
            first = False
        boundary += quarter_days


def _z_multiplier(confidence_level: int) -> float:
    """Return the z-multiplier for a given confidence level."""
    return stats.norm.ppf((1 + confidence_level / 100) / 2)


def plot_intra_period_accuracy(
    data: Union[pd.DataFrame, "ForecastData"],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    forecast_horizon: Optional[int] = None,
    statistic: Literal["rmse", "mae"] = "rmse",
    convert_to_percentage: bool = False,
    confidence_level: Optional[int] = None,
    return_plot: bool = False,
):
    """Plot forecast accuracy as a function of days to target.

    Shows how forecast accuracy evolves as the target period approaches.
    When ``forecast_horizon`` is ``None``, all horizons are shown on a
    single axis with dashed vertical lines at quarter boundaries.

    Parameters
    ----------
    data : ForecastData or pd.DataFrame
        A ForecastData instance (uses ``.df``) or a DataFrame with
        ``vintage_date_forecast`` and ``vintage_date_outturn`` columns.
    variable : str
        Variable to analyse (e.g., 'gdp', 'cpi').
    metric : str
        Metric to analyse ('levels', 'pop', or 'yoy').
    frequency : str
        Data frequency ('Q' for quarterly or 'M' for monthly).
    forecast_horizon : int or None
        Forecast horizon to plot. ``None`` (default) includes all horizons.
    statistic : str
        Accuracy statistic to compute ('rmse' or 'mae').
    convert_to_percentage : bool
        If True, multiplies values on the y-axis by 100.
    confidence_level : int or None
        If given (e.g. 90, 95, 99), shows confidence bands at that level
        around the statistic. ``None`` (default) hides bands.
    return_plot : bool
        If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        If return_plot is True, returns the figure and axes objects.
        Otherwise, displays the plot and returns None.
    """
    result = compute_intra_period_accuracy(data, variable, metric, frequency, forecast_horizon, statistic)

    multiplier = 100 if convert_to_percentage else 1
    stat_labels = {"rmse": "RMSE", "mae": "MAE"}
    stat_label = stat_labels.get(statistic, statistic.upper())

    fig, ax = create_themed_figure()

    z = _z_multiplier(confidence_level) if confidence_level is not None else None

    for source in sorted(result["source"].unique()):
        source_data = result[result["source"] == source]
        line = ax.plot(
            source_data["days_to_target"],
            multiplier * source_data["value"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=source,
        )
        if z is not None and "se" in source_data.columns:
            colour = line[0].get_color()
            ax.fill_between(
                source_data["days_to_target"],
                multiplier * (source_data["value"] - z * source_data["se"]),
                multiplier * (source_data["value"] + z * source_data["se"]),
                alpha=0.15,
                color=colour,
            )

    if not result.empty:
        _add_quarter_boundaries(ax, result["days_to_target"].min(), result["days_to_target"].max())

    horizon_str = f" - horizon {forecast_horizon}" if forecast_horizon is not None else ""
    ax.set_title(
        f"{stat_label} by Days to Target\n{variable.upper()} - {metric}{horizon_str}",
        fontsize=14,
    )
    ax.set_xlabel("Days to Target", fontsize=12)
    ax.set_ylabel(stat_label, fontsize=12)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(title="Source", loc="best")

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None


def plot_intra_period_bias(
    data: Union[pd.DataFrame, "ForecastData"],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    forecast_horizon: Optional[int] = None,
    convert_to_percentage: bool = False,
    confidence_level: Optional[int] = None,
    return_plot: bool = False,
):
    """Plot forecast bias (mean error) as a function of days to target.

    Parameters
    ----------
    data : ForecastData or pd.DataFrame
        A ForecastData instance or DataFrame with ``vintage_date_forecast``
        and ``vintage_date_outturn`` columns.
    variable : str
        Variable to analyse.
    metric : str
        Metric to analyse.
    frequency : str
        Data frequency ('Q' or 'M').
    forecast_horizon : int or None
        Forecast horizon to plot. ``None`` (default) includes all horizons.
    convert_to_percentage : bool
        If True, multiplies values on the y-axis by 100.
    confidence_level : int or None
        If given (e.g. 90, 95, 99), shows confidence bands at that level
        around the mean error. ``None`` (default) hides bands.
    return_plot : bool
        If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
    """
    result = compute_intra_period_bias(data, variable, metric, frequency, forecast_horizon)

    multiplier = 100 if convert_to_percentage else 1

    fig, ax = create_themed_figure()

    z = _z_multiplier(confidence_level) if confidence_level is not None else None

    for source in sorted(result["source"].unique()):
        source_data = result[result["source"] == source]
        line = ax.plot(
            source_data["days_to_target"],
            multiplier * source_data["value"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=source,
        )
        if z is not None and "se" in source_data.columns:
            colour = line[0].get_color()
            ax.fill_between(
                source_data["days_to_target"],
                multiplier * (source_data["value"] - z * source_data["se"]),
                multiplier * (source_data["value"] + z * source_data["se"]),
                alpha=0.15,
                color=colour,
            )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    if not result.empty:
        _add_quarter_boundaries(ax, result["days_to_target"].min(), result["days_to_target"].max())

    horizon_str = f" - horizon {forecast_horizon}" if forecast_horizon is not None else ""
    ax.set_title(
        f"Bias by Days to Target\n{variable.upper()} - {metric}{horizon_str}",
        fontsize=14,
    )
    ax.set_xlabel("Days to Target", fontsize=12)
    ax.set_ylabel("Mean Error", fontsize=12)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(title="Source", loc="best")

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
