from typing import TYPE_CHECKING, Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from forecast_evaluation.tests.intra_period import (
    AXIS_COLUMNS,
    compute_intra_period_accuracy,
    compute_intra_period_bias,
)
from forecast_evaluation.visualisations.theme import create_themed_figure

if TYPE_CHECKING:
    from forecast_evaluation.data.ForecastData import ForecastData

AXIS_LABELS = {"period_end": "Days to Period End", "publication": "Days to Publication"}


def _add_period_boundaries(ax, days_min, days_max, frequency, target_dates):
    """Add calendar month or quarter boundaries on a days-to-period-end axis."""
    if not target_dates:
        return

    anchor_date = pd.Timestamp(max(target_dates)).to_period(frequency).end_time.normalize()
    start_date = anchor_date - pd.Timedelta(days=float(days_max))
    end_date = anchor_date - pd.Timedelta(days=float(days_min))
    periods = pd.period_range(start=start_date.to_period(frequency), end=end_date.to_period(frequency), freq=frequency)

    label = f"{'Quarter' if frequency == 'Q' else 'Month'} boundary"
    first = True
    for period in periods:
        boundary = (anchor_date - period.end_time.normalize()).days
        if days_min <= boundary <= days_max:
            ax.axvline(
                x=boundary,
                color="grey",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                label=label if first else None,
            )
            first = False


def _z_multiplier(confidence_level: int) -> float:
    """Return the z-multiplier for a given confidence level."""
    if not 0 < confidence_level < 100:
        raise ValueError("confidence_level must be greater than 0 and less than 100")

    return stats.norm.ppf((1 + confidence_level / 100) / 2)


def plot_intra_period_accuracy(
    data: Union[pd.DataFrame, "ForecastData"],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    horizon: Optional[int] = None,
    statistic: Literal["rmse", "mae"] = "rmse",
    k: Optional[int] = None,
    axis: Literal["period_end", "publication"] = "period_end",
    convert_to_percentage: bool = False,
    confidence_level: Optional[int] = None,
    return_plot: bool = False,
):
    """Plot forecast accuracy as a function of a within-period time axis.

    Shows how forecast accuracy evolves as the forecast vintage approaches
    either the end of the target period or the outturn release (see ``axis``).
    When ``horizon`` is ``None``, all horizons are shown on a
    single axis; with ``axis='period_end'`` dashed vertical lines mark
    calendar period boundaries.

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
    horizon : int or None
        Forecast horizon to plot. ``None`` (default) includes all horizons.
    statistic : str
        Accuracy statistic to compute ('rmse' or 'mae').
    k : int or None
        Outturn revision index used to select the outturn. If ``None``
        (default), uses ``data.default_k`` for a ``ForecastData`` instance.
    axis : {'period_end', 'publication'}
        X-axis to plot against. ``'period_end'`` (default) is days from the
        forecast vintage to the end of the target period; ``'publication'``
        is days to the release of the selected outturn. The two differ by
        the publication lag, which is constant within a target period but
        varies across periods and series.
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
    result = compute_intra_period_accuracy(data, variable, metric, frequency, horizon, statistic, k, axis)
    x_col = AXIS_COLUMNS[axis]
    axis_label = AXIS_LABELS[axis]

    multiplier = 100 if convert_to_percentage else 1
    stat_labels = {"rmse": "RMSE", "mae": "MAE"}
    stat_label = stat_labels.get(statistic, statistic.upper())

    fig, ax = create_themed_figure()

    z = _z_multiplier(confidence_level) if confidence_level is not None else None

    label_col = "unique_id"
    for label in sorted(result[label_col].unique()):
        source_data = result[result[label_col] == label]
        line = ax.plot(
            source_data[x_col],
            multiplier * source_data["value"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
        )
        if z is not None and "se" in source_data.columns:
            colour = line[0].get_color()
            ax.fill_between(
                source_data[x_col],
                multiplier * (source_data["value"] - z * source_data["se"]),
                multiplier * (source_data["value"] + z * source_data["se"]),
                alpha=0.15,
                color=colour,
            )

    # Boundaries are anchored on target period ends, so only meaningful on that axis.
    if not result.empty and axis == "period_end":
        _add_period_boundaries(
            ax,
            result[x_col].min(),
            result[x_col].max(),
            frequency,
            result.attrs.get("target_dates", []),
        )

    horizon_str = f" - horizon {horizon}" if horizon is not None else ""
    ax.set_title(
        f"{stat_label} by {axis_label}\n{variable.upper()} - {metric}{horizon_str}",
        fontsize=14,
    )
    ax.set_xlabel(axis_label, fontsize=12)
    ax.set_ylabel(stat_label, fontsize=12)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(title="Forecast", loc="best")

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
    horizon: Optional[int] = None,
    k: Optional[int] = None,
    axis: Literal["period_end", "publication"] = "period_end",
    convert_to_percentage: bool = False,
    confidence_level: Optional[int] = None,
    return_plot: bool = False,
):
    """Plot forecast bias (mean error) as a function of a within-period time axis.

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
    horizon : int or None
        Forecast horizon to plot. ``None`` (default) includes all horizons.
    k : int or None
        Outturn revision index used to select the outturn. If ``None``
        (default), uses ``data.default_k`` for a ``ForecastData`` instance.
    axis : {'period_end', 'publication'}
        X-axis to plot against. ``'period_end'`` (default) is days from the
        forecast vintage to the end of the target period; ``'publication'``
        is days to the release of the selected outturn.
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
    result = compute_intra_period_bias(data, variable, metric, frequency, horizon, k, axis)
    x_col = AXIS_COLUMNS[axis]
    axis_label = AXIS_LABELS[axis]

    multiplier = 100 if convert_to_percentage else 1

    fig, ax = create_themed_figure()

    z = _z_multiplier(confidence_level) if confidence_level is not None else None

    label_col = "unique_id"
    for label in sorted(result[label_col].unique()):
        source_data = result[result[label_col] == label]
        line = ax.plot(
            source_data[x_col],
            multiplier * source_data["value"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
        )
        if z is not None and "se" in source_data.columns:
            colour = line[0].get_color()
            ax.fill_between(
                source_data[x_col],
                multiplier * (source_data["value"] - z * source_data["se"]),
                multiplier * (source_data["value"] + z * source_data["se"]),
                alpha=0.15,
                color=colour,
            )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Boundaries are anchored on target period ends, so only meaningful on that axis.
    if not result.empty and axis == "period_end":
        _add_period_boundaries(
            ax,
            result[x_col].min(),
            result[x_col].max(),
            frequency,
            result.attrs.get("target_dates", []),
        )

    horizon_str = f" - horizon {horizon}" if horizon is not None else ""
    ax.set_title(
        f"Bias by {axis_label}\n{variable.upper()} - {metric}{horizon_str}",
        fontsize=14,
    )
    ax.set_xlabel(axis_label, fontsize=12)
    ax.set_ylabel("Mean Error", fontsize=12)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(title="Forecast", loc="best")

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
