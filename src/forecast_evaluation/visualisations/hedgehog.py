import warnings
from datetime import date
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from forecast_evaluation.data import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData
from forecast_evaluation.utils import clean_unique_id, filter_k
from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_hedgehog(
    data: ForecastData,
    variable: str,
    forecast_source: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Optional[Literal["Q", "M"]] = None,
    k: int = 12,
    date_start: Union[str, date, None] = None,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
) -> tuple[plt.Figure, plt.Axes] | None:
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

    if frequency is not None:
        warnings.warn(
            "The 'frequency' argument is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    df_forecasts = data._forecasts.copy()
    df_outturns = data._main_table.copy()
    df_outturns = filter_k(df_outturns, k)

    # Filter the data
    df_outturns_filtered = df_outturns[
        (df_outturns["variable"] == variable)
        & (df_outturns["unique_id"] == forecast_source)
        & (df_outturns["metric"] == metric)
    ]

    df_forecasts_filtered = df_forecasts[
        (df_forecasts["variable"] == variable)
        & (df_forecasts["unique_id"] == forecast_source)
        & (df_forecasts["metric"] == metric)
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
    vintage_dates = sorted(df_forecasts_filtered["vintage_date"].unique())

    # For NowcastData there are multiple vintages per target date; within each
    # target date we colour the dots with a gradient from the earliest to the
    # latest vintage so the evolution of the nowcast across the nowcasting
    # window is visible.
    is_nowcast = isinstance(data, NowcastData)
    if is_nowcast:
        cmap_base = plt.get_cmap("YlOrRd")
        cmap = LinearSegmentedColormap.from_list(
            "nowcast_vintage_gradient",
            [cmap_base(i / 255) for i in range(64, 243)],
        )

    # Plot a line for each forecast vintage
    for i, vintage_date in enumerate(vintage_dates):
        # if there is only one available horizon dot_size = 3 otherwise 0
        if is_nowcast:
            dot_size = 0
        elif (
            df_forecasts_filtered[df_forecasts_filtered["vintage_date"] == vintage_date]["forecast_horizon"].nunique()
            == 1
        ):
            dot_size = 5
        else:
            dot_size = 0

        vintage_data = df_forecasts_filtered[df_forecasts_filtered["vintage_date"] == vintage_date].sort_values("date")
        # Only add label for the first line to avoid duplicate legend entries
        label = "Forecasts" if i == 0 else None
        ax.plot(
            vintage_data["date"],
            multiplier * vintage_data["value"],
            linewidth=1.5,
            marker="o",
            markersize=dot_size,
            alpha=0.7,
            color="lightblue",
            label=label,
        )

    # For nowcasts, overlay scatter dots coloured by vintage rank within each
    # target date so the gradient reflects nowcast evolution per outturn.
    if is_nowcast:
        df_nc = df_forecasts_filtered.sort_values(["date", "vintage_date"]).copy()
        # Rank vintages within each target date and normalise to [0, 1].
        ranks = df_nc.groupby("date")["vintage_date"].rank(method="dense") - 1
        counts = df_nc.groupby("date")["vintage_date"].transform("nunique")
        denom = (counts - 1).where(counts > 1, 1)
        df_nc["_vintage_pos"] = (ranks / denom).clip(0, 1)
        colors = cmap(df_nc["_vintage_pos"].to_numpy())
        ax.scatter(
            df_nc["date"],
            multiplier * df_nc["value"],
            c=colors,
            s=24,
            alpha=0.85,
            edgecolors="none",
            zorder=3,
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
    ax.set_title(f"{variable} forecasts and actuals ({clean_unique_id(forecast_source)}, k={k})", fontsize=14)

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

    # Colourbar showing the vintage gradient for nowcasts
    if is_nowcast:
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        sm.set_array([])
        # Use an inset axes so we don't reshape the parent axes and clash with
        # the figure's layout engine (e.g. constrained_layout in the dashboard).
        cax = ax.inset_axes([1.02, 0.0, 0.025, 1.0])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Earliest", "Latest"])
        cbar.set_label("Vintage within outturn", fontsize=10)

    # Return or show the plot
    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
