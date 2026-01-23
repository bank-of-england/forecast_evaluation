from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from forecast_evaluation.data import ForecastData
from forecast_evaluation.utils import filter_k
from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_forecast_errors(
    data: ForecastData,
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    source: str,
    vintage_date_forecast: str,
    k: int = 12,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """
    Plot average forecast errors for a specific variable/source/vintage combination

    Parameters
    -----------
    data : ForecastData
        ForecastData object containing forecast accuracy data
    variable : str
        The variable to analyze (e.g., 'gdpkp')
    metric : str
        The metric to analyze (e.g., 'yoy', 'pop', 'levels')
    frequency : str
        The frequency to analyze (e.g., 'Q', 'M')
    source : str
        The source of the forecasts (e.g., 'compass')
    vintage_date_forecast : str
        The vintage_date of the forecasts (e.g., '2022-03-31')
    k : int
        The k to analyze (e.g., 12)
    convert_to_percentage : bool, default=False
        If True, multiplies values on the y-axis by 100
    return_plot : bool, default=False
        If True, returns the matplotlib figure and axis objects

    Returns
    --------
    tuple or None
        If return_plot is True, returns (fig, ax). Otherwise, returns None.
    """
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()
    df = filter_k(df, k)

    # Filter data for the specific combination
    mask = (
        (df["variable"] == variable)
        & (df["unique_id"] == source)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
        & (df["vintage_date_forecast"] == vintage_date_forecast)
    )

    subset = df.loc[mask].copy().sort_values("date")

    if len(subset) == 0:
        raise ValueError(
            f"No data available for {variable} from {source} ({vintage_date_forecast})"
            + f" with metric {metric} and frequency {frequency}"
        )

    # Multiply forecast errors by 100 if convert_to_percentage = True
    if convert_to_percentage:
        subset["forecast_error"] = 100 * subset["forecast_error"]

    # Calculate mean error for reference line
    mean_error = subset["forecast_error"].mean()

    # Create the plot using themed figure
    fig, ax = create_themed_figure()

    ax.plot(subset["forecast_horizon"], subset["forecast_error"], "o-", label="Forecast Error")
    ax.axhline(y=0, color="r", linestyle="-", label="Zero Error")
    ax.axhline(y=mean_error, color="g", linestyle="--", label=f"Mean Error: {mean_error:.4f}")

    ax.set_title(f"Forecast Errors for {variable.upper()} ({source}, vintage={vintage_date_forecast})\n{metric}, k={k}")
    ax.set_xlabel("Forecast Horizon")

    # Update y-axis label based on whether values were multiplied
    if convert_to_percentage:
        y_label = "Forecast error (p.p.)"
    else:
        y_label = "Forecast error"

    ax.set_ylabel(y_label, fontsize=12)
    ax.legend()
    ax.grid(True)

    # Return or show the plot
    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None


def plot_forecast_errors_by_horizon(
    data: ForecastData,
    variable: str,
    source: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    k: int = 12,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """
    Plot average forecast errors by forecast horizon, averaged over all forecast vintages.

    Parameters
    -----------
    data : ForecastData
        ForecastData object containing forecast accuracy data
    variable : str
        The variable to analyze (e.g., 'gdpkp', 'cpisa', 'unemp')
    source : str
        The source of the forecasts (e.g., 'compass conditional', 'mpr')
    metric : str
        The metric to analyze (e.g., 'yoy', 'pop', 'levels')
    frequency : str
        The frequency to analyze (e.g., 'Q', 'M')
    k : int
        The k to analyze (e.g., 12)
    convert_to_percentage : bool, default=False
        If True, multiplies values on the y-axis by 100
    return_plot : bool, default=False
        If True, returns the matplotlib figure and axis objects

    Returns
    --------
    tuple or None
        If return_plot is True, returns (fig, ax). Otherwise, returns None.
    """
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()
    df = filter_k(df, k)

    # Filter data for the specific variable, source and metric
    mask = (
        (df["variable"] == variable)
        & (df["unique_id"] == source)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
    )

    subset = df.loc[mask].copy()

    if len(subset) == 0:
        raise ValueError(
            f"No data available for {variable} from {source} with metric {metric} and frequency {frequency}"
        )

    # Multiply by 100 if convert_to_percentage = True
    if convert_to_percentage:
        subset["forecast_error"] = 100 * subset["forecast_error"]

    # Calculate average forecast error by forecast horizon
    avg_errors_by_horizon = (
        subset.groupby("forecast_horizon")["forecast_error"].agg(["mean", "std", "count"]).reset_index()
    )
    avg_errors_by_horizon.columns = ["forecast_horizon", "avg_forecast_error", "std_error", "n_observations"]

    # Sort by forecast horizon for better visualization
    avg_errors_by_horizon = avg_errors_by_horizon.sort_values("forecast_horizon")

    # Create the plot using themed figure
    fig, ax = create_themed_figure()

    # Plot: Average errors by horizon as a line chart
    ax.plot(
        avg_errors_by_horizon["forecast_horizon"],
        avg_errors_by_horizon["avg_forecast_error"],
        marker="o",
        linewidth=2,
        markersize=6,
        label="Average Forecast Error",
    )

    # Add shaded error region if we have standard deviation data
    if not avg_errors_by_horizon["std_error"].isna().all():
        ax.fill_between(
            avg_errors_by_horizon["forecast_horizon"],
            avg_errors_by_horizon["avg_forecast_error"] - avg_errors_by_horizon["std_error"],
            avg_errors_by_horizon["avg_forecast_error"] + avg_errors_by_horizon["std_error"],
            alpha=0.3,
            label="± 1 Standard Deviation",
        )

    # Add zero reference line
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Zero Error")

    # Customize plot
    ax.set_title(f"Average Forecast Errors by Forecast Horizon\n{variable.upper()} - {source} ({metric})", fontsize=14)
    ax.set_xlabel("Forecast Horizon", fontsize=12)

    # Update y-axis label based on whether values were multiplied
    if convert_to_percentage:
        y_label = "Forecast error (p.p.)"
    else:
        y_label = "Forecast error"

    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Return or show the plot
    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None


def plot_forecast_error_density(
    data: ForecastData,
    variable: str,
    horizon: int,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    source: str,
    k: int = 12,
    highlight_dates: Optional[Union[str, list[str]]] = None,
    highlight_vintages: Optional[Union[str, list[str]]] = None,
    return_plot: bool = False,
):
    """
    Plot density of forecast errors for a specific variable/source/vintage combination

    Parameters
    -----------
    data : ForecastData
        ForecastData object containing forecast error data
    variable : str
        The variable to analyze (e.g., 'gdpkp')
    horizon : int
        The forecast horizon to analyze
    metric : str
        The metric to analyze (e.g., 'yoy', 'pop', 'levels')
    frequency : str
        The frequency to analyze (e.g., 'Q', 'M')
    source : str
        The source of the forecasts (e.g., 'compass')
    k : int
        The k to analyze (e.g., 12)
    highlight_dates : str or list of str, optional
        Date(s) to highlight (format: 'YYYY-MM-DD'). If None, highlights last observation.
    highlight_vintages : str or list of str, optional
        Vintage date(s) to highlight (format: 'YYYY-MM-DD'). Takes precedence over highlight_dates.
    return_plot : bool, default=False
        If True, returns the matplotlib figure and axis objects

    Returns
    --------
    tuple or None
        If return_plot is True, returns (fig, ax). Otherwise, returns None.
    """
    if data._main_table is None:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()
    df = filter_k(df, k)

    # Filter data for the specific combination
    mask = (
        (df["variable"] == variable)
        & (df["unique_id"] == source)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
        & (df["forecast_horizon"] == horizon)
    )

    subset = df.loc[mask].copy().sort_values("date")

    if len(subset) == 0:
        raise ValueError(
            f"No data available for {variable} from {source} ({horizon}) with metric {metric} and frequency {frequency}"
        )

    # Calculate statistics
    mean_error = subset["forecast_error"].mean()
    std_error = subset["forecast_error"].std()

    # Determine which observations to highlight
    highlight_obs = []

    if highlight_vintages is not None:
        # Convert to list if single value
        if isinstance(highlight_vintages, str):
            highlight_vintages = [highlight_vintages]

        highlight_vintages = pd.to_datetime(highlight_vintages)
        highlight_mask = subset["vintage_date_forecast"].isin(highlight_vintages)
        highlight_obs = subset[highlight_mask]

    if highlight_dates is not None:
        # Convert to list if single value
        if isinstance(highlight_dates, str):
            highlight_dates = [highlight_dates]

        highlight_dates = pd.to_datetime(highlight_dates)
        highlight_mask = subset["date"].isin(highlight_dates)
        highlight_obs = subset[highlight_mask]

    # Create KDE density plot using themed figure
    fig, ax = create_themed_figure()

    # Calculate kernel density estimation
    errors = subset["forecast_error"].dropna()
    kde = gaussian_kde(errors)
    x_range = np.linspace(errors.min(), errors.max(), 200)
    density = kde(x_range)

    # Plot the density
    ax.fill_between(x_range, density, alpha=0.2, color="black")

    # Add vertical lines for mean and median
    mean_density = kde(mean_error)[0]
    ax.vlines(
        x=mean_error,
        ymin=0,
        ymax=mean_density,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.9,
        label=f"Mean: {mean_error:.4f}",
    )

    # Add standard deviation bands
    std_plus_density = kde(mean_error + std_error)[0]
    std_minus_density = kde(mean_error - std_error)[0]
    ax.vlines(
        x=mean_error + std_error,
        ymin=0,
        ymax=std_plus_density,
        color="black",
        linestyle=":",
        linewidth=2,
        alpha=0.9,
        label=f"±1 SD: {std_error:.4f}",
    )
    ax.vlines(
        x=mean_error - std_error, ymin=0, ymax=std_minus_density, color="black", linestyle=":", linewidth=2, alpha=0.9
    )

    # Plot highlighted observations as points on the density curve
    if len(highlight_obs) > 0:
        colour_map = plt.cm.tab20b
        for idx, (i, row) in enumerate(highlight_obs.iterrows()):
            error_val = row["forecast_error"]
            date_val = row["date"]
            density_val = kde(error_val)[0]

            ax.plot(
                error_val,
                density_val,
                "o",
                color=colour_map(idx),
                alpha=0.7,
                markersize=10,
                label=f"{pd.Period(date_val, freq='Q').strftime('%YQ%q')}",
                zorder=5,
            )

    ax.set_title(
        f"Distribution of Forecast Errors for {variable.upper()}({source}, horizon={horizon})\n{metric}, k={k}"
    )
    ax.set_xlabel("Forecast Error")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Return or show the plot
    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
