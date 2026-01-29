from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from forecast_evaluation.data import ForecastData
from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_vintage(
    data: ForecastData,
    variable: str,
    vintage_date: str | pd.Timestamp,
    forecast_source: list[str] = None,
    outturn_start_date: str | pd.Timestamp = None,
    frequency: Literal["Q", "M"] = "Q",
    metric: Literal["levels", "pop", "yoy"] = "levels",
    return_plot: bool = False,
) -> None:
    """Generate a plot comparing forecasts from different sources for a specific vintage.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data.
    variable : str
        Name of the variable to plot.
    forecast_source : list of str
        List of forecast sources to include in the plot.
    vintage_date : str or pd.Timestamp
        The vintage date to plot, either as string or pandas Timestamp.
    outturn_start_data : str or pd.Timestamp, optional
        Start date for outturn data to display (inclusive). If None, all available outturns are used.
    frequency : {"Q", "M"}, default "Q"
        Frequency of the data, either quarterly or monthly.
    metric : {"levels", "pop", "yoy"}, default "levels"
        Type of transformation to apply to the data.
    return_plot : bool, default False
        If True, returns the matplotlib figure and axis objects.

    Returns
    -------
    fig, ax : tuple or None
        If return_plot is True, returns a tuple (fig, ax) of the matplotlib figure and axis objects.
        Otherwise, returns None.
    """
    if data._forecasts is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

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
    )
    forecasts_filtered = forecasts_filtered.forecasts.copy()

    # filter outturns (they are not filtered with filter())(we select the last vintage only)
    outturns = data._outturns.copy()
    min_date = outturn_start_date if outturn_start_date is not None else outturns["date"].min()

    outturns = outturns[
        (outturns["vintage_date"] == vintage_date)
        & (outturns["variable"].isin(forecasts_filtered["variable"].unique()))
        & (outturns["metric"] == metric)
        & (outturns["date"] <= forecasts_filtered["date"].max())
        & (outturns["date"] >= min_date)
    ].copy()

    fig, ax = create_themed_figure()

    # Plot each source separately
    for forecast_id in forecasts_filtered["unique_id"].unique():
        source_df = forecasts_filtered[forecasts_filtered["unique_id"] == forecast_id].sort_values("date")
        ax.plot(source_df["date"], source_df["value"], marker="o", markersize=3, label=forecast_id, alpha=0.7)

    # Overlay the outturns series (forecast_horizon == -1)
    outturns_data = outturns.sort_values("date")
    if not outturns_data.empty:
        # Split outturns: solid before vintage_date, dashed from vintage_date onwards
        solid_outturns = outturns_data[outturns_data["date"] < vintage_date]
        dashed_outturns = outturns_data[outturns_data["date"] >= vintage_date]

        if not solid_outturns.empty:
            ax.plot(
                solid_outturns["date"],
                solid_outturns["value"],
                color="darkblue",
                marker="o",
                markersize=3,
                label="Outturns (solid)",
            )
        if not dashed_outturns.empty:
            ax.plot(
                dashed_outturns["date"],
                dashed_outturns["value"],
                color="darkblue",
                marker="o",
                markersize=3,
                linestyle="--",
                label="Outturns (dashed)",
            )

    ax.set_title(f"{variable} [{frequency}] - {metric} - Vintage: {vintage_date.date()}")
    ax.set_ylabel(f"{variable} ({metric})")
    ax.legend(title="unique_id")

    # Return or show the plot
    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
