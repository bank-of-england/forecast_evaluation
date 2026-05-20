import warnings
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd

from forecast_evaluation.data import ForecastData
from forecast_evaluation.utils import clean_unique_id
from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_vintage(
    data: ForecastData,
    variable: str,
    vintage_date: str | pd.Timestamp,
    forecast_source: list[str] = None,
    outturn_start_date: str | pd.Timestamp = None,
    frequency: Optional[Literal["Q", "M"]] = None,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    k: int = 12,
    convert_to_percentage: bool = False,
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
    outturn_start_date : str or pd.Timestamp, optional
        Start date for outturn data to display (inclusive). If None, all available outturns are used.
    metric : {"levels", "pop", "yoy"}, default "levels"
        Type of transformation to apply to the data.
    convert_to_percentage : bool, default False
        If True, multiplies values on the y-axis by 100.
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

    if frequency is not None:
        warnings.warn(
            "The 'frequency' argument is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    frequency = data._forecasts["frequency"].iloc[0]

    # add a check here
    vintage_date = pd.to_datetime(vintage_date)

    # filter forecasts
    filtered_data = data.copy()

    filtered_data.filter(
        variables=variable,
        metrics=metric,
        sources=forecast_source,
        start_vintage=vintage_date,
        end_vintage=vintage_date,
    )
    forecasts_filtered = filtered_data.forecasts.copy()

    # filter outturns (they are not filtered with filter())(we select the last vintage only)
    outturns = data._outturns.copy()
    min_date = outturn_start_date if outturn_start_date is not None else outturns["date"].min()

    if not data.outturn_vintages:
        # No vintage information — use the single available outturn per date
        real_time_outturns = (
            outturns[
                (outturns["variable"].isin(forecasts_filtered["variable"].unique()))
                & (outturns["metric"] == metric)
                & (outturns["date"] <= forecasts_filtered["date"].max())
                & (outturns["date"] >= min_date)
            ]
            .copy()
            .sort_values("date")
        )
        post_outturns = real_time_outturns.copy()
    else:
        real_time_outturns = (
            outturns[
                (outturns["vintage_date"] == vintage_date)
                & (outturns["variable"].isin(forecasts_filtered["variable"].unique()))
                & (outturns["metric"] == metric)
                & (outturns["date"] <= forecasts_filtered["date"].max())
                & (outturns["date"] >= min_date)
            ]
            .copy()
            .sort_values("date")
        )

        # Use -(k+1) if it exists, otherwise use max(forecast_horizon)
        post_outturns = outturns.copy()

        post_outturns["max_feasible_horizon"] = post_outturns.groupby("date")["forecast_horizon"].transform(
            lambda x: -(k + 1) if -(k + 1) in x.values else x.min()
        )

        post_outturns = (
            outturns[
                (outturns["forecast_horizon"] == post_outturns["max_feasible_horizon"])
                & (outturns["variable"].isin(forecasts_filtered["variable"].unique()))
                & (outturns["metric"] == metric)
                & (outturns["date"] <= forecasts_filtered["date"].max())
                & (outturns["date"] >= min_date)
            ]
            .copy()
            .sort_values("date")
        )

    multiplier = 100 if convert_to_percentage else 1
    forecasts_filtered = clean_unique_id(forecasts_filtered)

    fig, ax = create_themed_figure()

    # Plot each source separately
    for forecast_id in forecasts_filtered["unique_id"].unique():
        source_df = forecasts_filtered[forecasts_filtered["unique_id"] == forecast_id].sort_values("date")
        ax.plot(
            source_df["date"], multiplier * source_df["value"], marker="o", markersize=3, label=forecast_id, alpha=0.7
        )

    # Overlay the outturns series
    if not data.outturn_vintages:
        # No vintage information — plot all outturns as a single solid line
        if not real_time_outturns.empty:
            ax.plot(
                real_time_outturns["date"],
                multiplier * real_time_outturns["value"],
                color="darkblue",
                marker="o",
                markersize=3,
                label="Outturns",
            )
    elif not real_time_outturns.empty:
        # Split outturns: solid before vintage_date, dashed from vintage_date onwards
        solid_outturns = real_time_outturns[real_time_outturns["date"] < vintage_date]
        dashed_outturns = post_outturns[post_outturns["date"] >= vintage_date]

        if not solid_outturns.empty:
            ax.plot(
                solid_outturns["date"],
                multiplier * solid_outturns["value"],
                color="darkblue",
                marker="o",
                markersize=3,
                label="Outturns (at the time of forecast)",
            )
        if not dashed_outturns.empty:
            ax.plot(
                dashed_outturns["date"],
                multiplier * dashed_outturns["value"],
                color="darkblue",
                marker="o",
                markersize=3,
                linestyle="--",
                label="Outturns (post forecast)",
            )

    ax.set_title(f"{variable} [{frequency}] - {metric} - Vintage: {vintage_date.date()}")
    y_label = f"{variable} ({metric}) (p.p.)" if convert_to_percentage else f"{variable} ({metric})"
    ax.set_ylabel(y_label)
    ax.legend()

    # Return or show the plot
    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None


def plot_nowcasts(
    data: ForecastData,
    variable: str,
    target_date: str | pd.Timestamp,
    forecast_source: list[str] = None,
    frequency: Literal["Q", "M"] = "Q",
    metric: Literal["levels", "pop", "yoy"] = "levels",
    k: int = 12,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
) -> None:
    """Plot the evolution of nowcasts for a target quarter.

    Shows how forecasts for a single target quarter evolved over time, from
    the first nearcast to the last nowcast before the outturn was published.
    The x-axis is the forecast vintage date, with one line per source.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data.
    variable : str
        Name of the variable to plot.
    target_date : str or pd.Timestamp
        The target date (end of the quarter/month) to show nowcasts for.
    forecast_source : list of str, optional
        List of forecast sources to include in the plot.
    frequency : {"Q", "M"}, default "Q"
        Frequency of the data.
    metric : {"levels", "pop", "yoy"}, default "levels"
        Type of transformation to apply to the data.
    k : int, default 12
        Number of revisions used to define the outturn value.
    convert_to_percentage : bool, default False
        If True, multiplies values on the y-axis by 100.
    return_plot : bool, default False
        If True, returns the matplotlib figure and axis objects.

    Returns
    -------
    fig, ax : tuple or None
        If return_plot is True, returns (fig, ax). Otherwise None.
    """
    if data._forecasts is None or data._forecasts.empty:
        raise ValueError("ForecastData forecasts are not available.")

    target_date = pd.to_datetime(target_date)

    # Get all forecasts for the target date
    filtered_data = data.copy()
    filtered_data.filter(
        variables=variable,
        metrics=metric,
        frequencies=frequency,
        sources=forecast_source,
    )
    forecasts = filtered_data.forecasts
    forecasts = forecasts[forecasts["date"] == target_date].copy()

    if forecasts.empty:
        raise ValueError(f"No forecasts found for {variable} targeting {target_date.date()} ")

    # Get the outturn value from the main table
    from forecast_evaluation.utils import filter_k

    main_table = data._main_table
    outturn_row = filter_k(
        main_table[
            (main_table["variable"] == variable)
            & (main_table["metric"] == metric)
            & (main_table["date"] == target_date)
        ],
        k=k,
    )
    outturn_value = outturn_row["value_outturn"].iloc[0] if not outturn_row.empty else None

    multiplier = 100 if convert_to_percentage else 1

    fig, ax = create_themed_figure()

    # Plot one line per source (not unique_id, since unique_id includes days_in_period)
    for source in sorted(forecasts["source"].unique()):
        source_df = forecasts[forecasts["source"] == source].sort_values("vintage_date")
        ax.plot(
            source_df["vintage_date"],
            multiplier * source_df["value"],
            marker="o",
            markersize=3,
            label=source,
            alpha=0.7,
        )

    # Show outturn as horizontal line
    if outturn_value is not None:
        ax.axhline(
            y=multiplier * outturn_value,
            color="darkblue",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Outturn (k={k})",
        )

    ax.set_title(f"{variable} [{frequency}] - {metric} - Target: {target_date.to_period(frequency)}")
    y_label = f"{variable} ({metric}) (p.p.)" if convert_to_percentage else f"{variable} ({metric})"
    ax.set_ylabel(y_label)
    ax.set_xlabel("Forecast vintage date")
    ax.legend()

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
