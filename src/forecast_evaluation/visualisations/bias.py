from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from forecast_evaluation.visualisations.theme import create_themed_figure

if TYPE_CHECKING:
    from forecast_evaluation.tests.results import TestResult


def plot_bias_by_horizon(
    df: Union[pd.DataFrame, "TestResult"],
    variable: str,
    source: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """
    Plot average forecast errors by forecast horizon, averaged over all forecast vintages.

    Parameters
    -----------
    df : pd.DataFrame or TestResult
        Bias summary dataset generated with `bias_analysis()` or a BiasResults object.
    variable : str
        The variable to analyze (e.g., 'gdpkp', 'cpisa', 'unemp')
    source : str
        The source of the forecasts (e.g., 'compass conditional', 'mpr')
    metric : str
        The metric to analyze (e.g., 'yoy', 'pop', 'levels')
    frequency : str
        The frequency to analyze (e.g., 'Q', 'M')
    convert_to_percentage : bool, default=False
        If True, multiplies values on the y-axis by 100
    return_plot : bool, default=False
        If True, returns the matplotlib figure and axis objects

    Returns
    --------
    tuple or None
        If return_plot is True, returns (fig, ax). Otherwise, returns None.
    """
    # Extract DataFrame if TestResult object is passed
    if hasattr(df, "to_df"):
        df = df.to_df()

    # Filter data for the specific variable, source and metric
    mask = (
        (df["variable"] == variable)
        & (df["unique_id"] == source)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
    )

    df_filtered = df.loc[mask].copy()

    if len(df_filtered) == 0:
        raise ValueError(
            f"No data available for {variable} from {source} with metric {metric} and frequency {frequency}"
        )

    # Multiply by 100 if convert_to_percentage = True
    if convert_to_percentage:
        df_filtered["bias_estimate"] = 100 * df_filtered["bias_estimate"]
        df_filtered["ci_lower"] = 100 * df_filtered["ci_lower"]
        df_filtered["ci_upper"] = 100 * df_filtered["ci_upper"]

    # Sort by forecast horizon for better visualization
    df_filtered = df_filtered.sort_values("forecast_horizon")

    # Create the plot using themed figure
    fig, ax = create_themed_figure()

    # Plot: Average errors by horizon as a line chart
    ax.plot(
        df_filtered["forecast_horizon"],
        df_filtered["bias_estimate"],
        marker="o",
        linewidth=2,
        markersize=6,
        label="Average Forecast Error",
    )

    # Add shaded error region if we have standard deviation data
    ax.fill_between(
        df_filtered["forecast_horizon"],
        df_filtered["ci_lower"],
        df_filtered["ci_upper"],
        alpha=0.3,
        label="95% Confidence Interval",
    )

    # Add zero reference line
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Zero Error")

    # Customize plot
    ax.set_title(
        f"Bias estimate with 95% CI by Forecast Horizon\n{variable.upper()} - {source} ({metric})", fontsize=14
    )
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


def plot_rolling_bias(
    df: pd.DataFrame,
    horizons: Sequence[int],
    variable: str = None,
    source: str = None,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """
    Plot bias estimates with confidence intervals across window_end dates,
    faceted by forecast_horizon.

    If the data contains fluctuation rejection columns (reject_05, reject_10),
    the bias estimates are marked with coloured dots indicating significance levels.

    Parameters
    ----------
    df : pd.DataFrame
        Rolling bias summary dataset generated with `rolling_analysis()` with the analysis func `bias_analysis`.
    horizons : Sequence[int]
        List of forecast horizons to plot.
    variable : str, optional
        Variable to filter by. If None, uses the first variable found in the data.
    source : str, optional
        Source to filter by. If None, uses the first source found in the data.
    convert_to_percentage : bool, default False
        If True, multiplies bias estimates and confidence intervals by 100.
    return_plot : bool, default False
        If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        If return_plot is True, returns the figure and axes objects. Otherwise, displays the plot and returns None.
    """
    # Ensure window_end is datetime
    df = df.copy()
    df["window_end"] = pd.to_datetime(df["window_end"])

    # Filter by variable if provided
    if variable is not None:
        df = df[df["variable"] == variable]
    else:
        variables = df["variable"].unique()
        df = df[df["variable"] == variables[0]]

        # warning message
        print(f"No variable provided. Plotting for variable: {variables[0]}")

    # Filter by source if provided
    if source is not None:
        df = df[df["unique_id"] == source]
    else:
        sources = df["unique_id"].unique()
        df = df[df["unique_id"] == sources[0]]
        # warning message
        print(f"No source provided. Plotting for source: {sources[0]}")

    df = df[df["forecast_horizon"].isin(horizons)]
    n_horizons = df["forecast_horizon"].nunique()
    fig, axes = create_themed_figure(nrows=n_horizons, ncols=1, sharex=True)

    if n_horizons == 1:
        axes = [axes]

    colors = plt.cm.tab10(range(1))

    if convert_to_percentage:
        df["bias_estimate"] = 100 * df["bias_estimate"]
        df["ci_lower"] = 100 * df["ci_lower"]
        df["ci_upper"] = 100 * df["ci_upper"]

    for i, h in enumerate(horizons):
        ax = axes[i]
        sub = df[df["forecast_horizon"] == h]

        # Determine marker properties based on fluctuation test results if available
        if "reject_05" in sub.columns:
            marker_colors = []

            for _, row in sub.iterrows():
                if row["reject_05"]:
                    marker_colors.append("#DC143C")  # Crimson red
                elif row["reject_10"]:
                    marker_colors.append("#FF6B6B")  # Light red
                else:
                    marker_colors.append(colors[0])

            # Plot line with customized markers
            ax.plot(
                sub["window_end"],
                sub["bias_estimate"],
                marker="o",
                label="Bias",
                color=colors[0],
                linestyle="-",
                linewidth=2,
                markersize=0,
            )[0]

            # Add markers with custom colors
            for x, y, c in zip(sub["window_end"], sub["bias_estimate"], marker_colors):
                ax.plot(x, y, marker="o", color=c, markersize=6, zorder=3)
        else:
            # Standard plot without fluctuation test coloring
            ax.plot(sub["window_end"], sub["bias_estimate"], marker="o", label="Bias", color=colors[0])

        ax.fill_between(sub["window_end"], sub["ci_lower"], sub["ci_upper"], color="gray", alpha=0.3, label="95% CI")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        # Add subtitle showing the horizon
        ax.set_title(f"Forecast Horizon = {h}", loc="center")

        # Update y-axis label based on whether values were multiplied
        if i == (n_horizons - 1) // 2:  # Middle subplot
            if convert_to_percentage:
                ax.set_ylabel("Bias Coefficient in p.p.")
            else:
                ax.set_ylabel("Bias Coefficient")

        # Add custom legend handles for significance only in first subplot
        if "reject_05" in sub.columns:
            if i == 0:
                legend_elements = [
                    Line2D([0], [0], marker="o", color="w", label="Bias", markerfacecolor=colors[0], markersize=8),
                    Line2D([0], [0], color="gray", alpha=0.3, label="95% CI"),
                    Line2D([0], [0], marker="o", color="w", label="p < 0.05", markerfacecolor="#DC143C", markersize=8),
                    Line2D([0], [0], marker="o", color="w", label="p < 0.10", markerfacecolor="#FF6B6B", markersize=8),
                ]
                ax.legend(handles=legend_elements)
        else:
            if i == 0:
                ax.legend()

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Window End Date")
    fig.suptitle("Bias Estimate with 95% CI by Forecast Horizon")

    if return_plot:
        return fig, axes[0] if n_horizons == 1 else axes
    else:
        plt.show()
        return None


# Example usage:
if __name__ == "__main__":
    import pandas as pd

    from forecast_evaluation.data.ForecastData import ForecastData
    from forecast_evaluation.tests.bias import bias_analysis

    # Initialise with fer ---------
    forecast_data = ForecastData(load_fer=True)

    forecast_data.filter()

    # Generate bias analysis data
    bias_results = bias_analysis(forecast_data)

    plot_bias_by_horizon(bias_results, "aweagg", "compass conditional", "yoy", "Q")
