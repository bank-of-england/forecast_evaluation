"""Visualisation functions for density forecasts."""

import warnings
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_evaluation.data import DensityForecastData
from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_density_vintage(
    data: DensityForecastData,
    variable: str,
    vintage_date: str | pd.Timestamp,
    quantiles: Optional[list[float]] = [0.16, 0.5, 0.84],
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
        print("Warning: ForecastData main table is not available. Please ensure data has been added and processed.")
        return None

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

    fig, ax = create_themed_figure()

    # Plot each source separately with quantile bands
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts_filtered["unique_id"].unique())))

    # check that the quantiles are in the dataframe otherwise select the closest available ones
    available_quantiles = forecasts_filtered["quantile"].unique()
    for i, target_quantile in enumerate(quantiles):
        if target_quantile not in available_quantiles:
            quantiles[i] = min(available_quantiles, key=lambda x: abs(x - target_quantile))
            forecasts_filtered.loc[forecasts_filtered["quantile"] == target_quantile, "quantile"] = quantiles[i]
            warnings.warn(
                f"Quantile {target_quantile} not found. Using closest available quantile: {quantiles[i]}",
                UserWarning,
                stacklevel=2,
            )

    # round quantiles
    quantiles = [round(q, 3) for q in quantiles]
    forecasts_filtered["quantile"] = forecasts_filtered["quantile"].round(3)

    for idx, forecast_id in enumerate(forecasts_filtered["unique_id"].unique()):
        source_df = forecasts_filtered[forecasts_filtered["unique_id"] == forecast_id].sort_values("date")
        color = colors[idx]

        # Group by date to get quantiles for each date
        for quantile in quantiles:
            quantile_data = source_df[source_df["quantile"] == quantile].sort_values("date")

            if not quantile_data.empty:
                if quantile == np.median(quantiles) or quantile == 0.5:  # Median - plot with solid line
                    ax.plot(
                        quantile_data["date"],
                        quantile_data["value"],
                        label=f"{forecast_id} (median)",
                        alpha=0.8,
                        color=color,
                    )
                else:
                    ax.plot(quantile_data["date"], quantile_data["value"], linestyle="--", alpha=0.1, color=color)

        # Optional: Add shaded regions between quantile pairs
        if len(quantiles) >= 3:
            lower_q = source_df[source_df["quantile"] == quantiles[0]].sort_values("date")
            upper_q = source_df[source_df["quantile"] == quantiles[-1]].sort_values("date")
            if not lower_q.empty and not upper_q.empty:
                ax.fill_between(
                    lower_q["date"],
                    lower_q["value"],
                    upper_q["value"],
                    alpha=0.1,
                    color=color,
                    label=f"{forecast_id} ({quantiles[0]}-{quantiles[-1]})",
                )

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
