from typing import TYPE_CHECKING, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    """Plot forecast accuracy as a function of days into the vintage period.

    Shows how forecast accuracy evolves within a quarter or month as more
    data becomes available. Requires the ``days_in_period`` column, which
    can be added via :func:`~forecast_evaluation.compute_days_in_period`
    and passed as an ``extra_ids`` column.

    Parameters
    ----------
    data : ForecastData or pd.DataFrame
        A ForecastData instance (uses ``.df``) or a DataFrame with at least
        the columns: ``variable``, ``metric``, ``frequency``,
        ``forecast_horizon``, ``forecast_error``, ``source``, and
        ``days_in_period``.
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
    # Extract DataFrame from ForecastData if needed
    if hasattr(data, "df"):
        df = data.df.copy()
    else:
        df = data.copy()

    if "days_in_period" not in df.columns:
        raise ValueError(
            "Column 'days_in_period' not found. Compute it with compute_days_in_period() "
            "and pass it as an extra_ids column when adding forecasts."
        )

    # Filter
    mask = (
        (df["variable"] == variable)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
        & (df["forecast_horizon"] == forecast_horizon)
    )
    df = df.loc[mask]

    if df.empty:
        raise ValueError(
            f"No data for variable='{variable}', metric='{metric}', "
            f"frequency='{frequency}', forecast_horizon={forecast_horizon}"
        )

    # Ensure days_in_period is numeric (it may be stored as string via extra_ids)
    df["days_in_period"] = pd.to_numeric(df["days_in_period"])

    # Compute statistic per (source, days_in_period)
    sources = df["source"].unique()
    multiplier = 100 if convert_to_percentage else 1

    fig, ax = create_themed_figure()

    for source in sorted(sources):
        source_data = df[df["source"] == source]

        if statistic == "rmse":
            stats = source_data.groupby("days_in_period")["forecast_error"].apply(lambda x: np.sqrt(np.mean(x**2)))
        elif statistic == "mae":
            stats = source_data.groupby("days_in_period")["forecast_error"].apply(lambda x: np.mean(np.abs(x)))

        stats = stats.sort_index()

        ax.plot(
            stats.index,
            multiplier * stats.values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=source,
        )

    stat_labels = {"rmse": "RMSE", "mae": "MAE"}
    stat_label = stat_labels.get(statistic, statistic.upper())
    period_label = "Quarter" if frequency == "Q" else "Month"

    ax.set_title(
        f"{stat_label} by Day in {period_label}\n{variable.upper()} - {metric} - horizon {forecast_horizon}",
        fontsize=14,
    )
    ax.set_xlabel(f"Days into {period_label}", fontsize=12)
    ax.set_ylabel(stat_label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Source", loc="best")

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
