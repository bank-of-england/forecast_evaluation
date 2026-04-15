from typing import TYPE_CHECKING, Literal, Union

import matplotlib.pyplot as plt
import pandas as pd

from forecast_evaluation.tests.intra_period import compute_intra_period_accuracy, compute_intra_period_bias
from forecast_evaluation.visualisations.theme import create_themed_figure

if TYPE_CHECKING:
    from forecast_evaluation.data.ForecastData import ForecastData


def plot_intra_period_accuracy(
    data: Union[pd.DataFrame, "ForecastData"],
    variable: str,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] = "Q",
    forecast_horizon: int = 0,
    statistic: Literal["rmse", "mae"] = "rmse",
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """Plot forecast accuracy as a function of days to publication.

    Shows how forecast accuracy evolves as the publication date approaches.
    Vintages are grouped by their chronological rank within each period;
    the x-axis shows the median days-to-publication for each rank.

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
    forecast_horizon : int
        Forecast horizon to plot. Default is 0 (nowcast for current period).
    statistic : str
        Accuracy statistic to compute ('rmse' or 'mae').
    convert_to_percentage : bool
        If True, multiplies values on the y-axis by 100.
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

    for source in sorted(result["source"].unique()):
        source_data = result[result["source"] == source]
        ax.plot(
            source_data["days_to_publication"],
            multiplier * source_data["value"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=source,
        )

    ax.set_title(
        f"{stat_label} by Days to Publication\n{variable.upper()} - {metric} - horizon {forecast_horizon}",
        fontsize=14,
    )
    ax.set_xlabel("Days to Publication", fontsize=12)
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
    forecast_horizon: int = 0,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """Plot forecast bias (mean error) as a function of days to publication.

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
    forecast_horizon : int
        Forecast horizon to plot.
    convert_to_percentage : bool
        If True, multiplies values on the y-axis by 100.
    return_plot : bool
        If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
    """
    result = compute_intra_period_bias(data, variable, metric, frequency, forecast_horizon)

    multiplier = 100 if convert_to_percentage else 1

    fig, ax = create_themed_figure()

    for source in sorted(result["source"].unique()):
        source_data = result[result["source"] == source]
        ax.plot(
            source_data["days_to_publication"],
            multiplier * source_data["value"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=source,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(
        f"Bias by Days to Publication\n{variable.upper()} - {metric} - horizon {forecast_horizon}",
        fontsize=14,
    )
    ax.set_xlabel("Days to Publication", fontsize=12)
    ax.set_ylabel("Mean Error", fontsize=12)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(title="Source", loc="best")

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
