from datetime import date
from typing import Literal, Union

import matplotlib.pyplot as plt

from forecast_evaluation.data import ForecastData
from forecast_evaluation.utils import filter_k
from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_hedgehog(
    data: ForecastData,
    variable: str,
    forecast_source: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"] = "Q",
    k: int = 12,
    date_start: Union[str, date, None] = None,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
) -> None:
    """Generate a hedgehog plot comparing forecasts with outturns.

    A hedgehog plot displays multiple forecast vintages as light blue lines overlaid
    with the outturn values as a darker line with markers. This visualization
    helps assess forecast accuracy over time.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data.
    variable : str
        Name of the variable to plot (e.g., 'cpisa', 'gdpkp').
    forecast_source : str
        Source of the forecasts.
    metric : {"levels", "pop", "yoy"}
        Type of transformation to apply to the data.
    frequency : {"Q", "M"}, default "Q"
        Frequency of the data, either quarterly or monthly.
    k : int, default 12
        Number of revisions used to define the outturns.
    date_start : str, date, or None, default None
        Optional start date to filter the data. If None, no filtering is applied.
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
    # Validate inputs
    if data._forecasts is None:
        raise ValueError("ForecastData forecasts are not available. Please ensure data has been added and processed.")
    if data._main_table is None:
        raise ValueError("ForecastData not available. Please ensure data has been added and processed.")

    df_forecasts = data._forecasts.copy()
    df_outturns = data._main_table.copy()
    df_outturns = filter_k(df_outturns, k)

    # Filter the data
    df_outturns_filtered = df_outturns[
        (df_outturns["variable"] == variable)
        & (df_outturns["unique_id"] == forecast_source)
        & (df_outturns["metric"] == metric)
        & (df_outturns["frequency"] == frequency)
    ]

    df_forecasts_filtered = df_forecasts[
        (df_forecasts["variable"] == variable)
        & (df_forecasts["unique_id"] == forecast_source)
        & (df_forecasts["metric"] == metric)
        & (df_forecasts["frequency"] == frequency)
        & (df_forecasts["forecast_horizon"] >= 0)
    ]

    if date_start is not None:
        df_forecasts_filtered = df_forecasts_filtered[df_forecasts_filtered["vintage_date"] >= date_start]
        df_outturns_filtered = df_outturns_filtered[df_outturns_filtered["vintage_date_forecast"] >= date_start]

    # Check if data exists
    if df_forecasts_filtered.empty:
        raise ValueError(f"No forecast data found for {variable} from {forecast_source} with metric {metric}")

    if df_outturns_filtered.empty:
        raise ValueError(f"No actuals data found for {variable} from {forecast_source} with metric {metric} and k {k}")

    # Multiply by 100 if convert_to_percentage = True
    multiplier = 100 if convert_to_percentage else 1

    # Create the hedgehog chart using object-oriented approach
    fig, ax = create_themed_figure()

    # Get unique forecast vintages
    vintage_dates = df_forecasts_filtered["vintage_date"].unique()

    # Plot a line for each forecast vintage
    for i, vintage_date in enumerate(vintage_dates):
        vintage_data = df_forecasts_filtered[df_forecasts_filtered["vintage_date"] == vintage_date].sort_values("date")
        # Only add label for the first line to avoid duplicate legend entries
        label = "Forecasts" if i == 0 else None
        ax.plot(
            vintage_data["date"],
            multiplier * vintage_data["value"],
            linewidth=1.5,
            markersize=0,
            alpha=0.7,
            color="lightblue",
            label=label,
        )

    # Overlay the actuals series (forecast_horizon == 0)
    actuals_data = df_outturns_filtered
    actuals_data = actuals_data[["date", "value_outturn"]].drop_duplicates().sort_values("date")

    if not actuals_data.empty:
        ax.plot(
            actuals_data["date"],
            multiplier * actuals_data["value_outturn"],
            linewidth=2,
            marker="o",
            markersize=3,
            alpha=1.0,
            color="darkblue",
            label="Outturns",
        )

    # Add dashed line at 2% target (only for inflation-related metrics)
    if "cpisa" in variable.lower() and metric.lower() == "yoy":
        target_value = 2 if convert_to_percentage else 0.02
        ax.axhline(y=target_value, color="red", linestyle="--", linewidth=2, alpha=0.8, label="2% Target")

    # Customize the plot
    ax.set_title(f"{variable} forecasts and actuals ({forecast_source}, k={k})", fontsize=14)

    # Update y-axis label based on whether values were multiplied
    if convert_to_percentage and metric.lower() == "levels":
        y_label = "Rate (%)"
    elif convert_to_percentage:
        y_label = f"{metric} (%)"
    else:
        y_label = f"{metric}"

    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend()

    # Return or show the plot
    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
