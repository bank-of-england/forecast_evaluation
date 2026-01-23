from typing import TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import pandas as pd

from forecast_evaluation.visualisations.theme import create_themed_figure

if TYPE_CHECKING:
    from forecast_evaluation.tests.results import TestResult


def plot_blanchard_leigh_ratios(results: Union[pd.DataFrame, "TestResult"], return_plot: bool = False):
    """Plot Blanchard-Leigh efficiency test ratios across horizons with confidence intervals.

    Parameters
    ----------
    results : DataFrame or BlanchardLeighResults
        DataFrame or BlanchardLeighResults object containing Blanchard-Leigh test results
        for multiple horizons. Expected columns include 'horizon', 'outcome_variable',
        'instrument_variable', 'ratio', 'ratio_se', 'ratio_ci_lower', 'ratio_ci_upper', 'alpha'.
    return_plot : bool, default False
        If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        If return_plot is True, returns the figure and axes objects. Otherwise, displays the plot and returns None.
    """
    # Extract DataFrame if TestResult object is passed
    if hasattr(results, "to_df"):
        results = results.to_df()

    # Filter out rows with missing data
    plot_df = results.dropna(subset=["ratio", "ratio_ci_lower", "ratio_ci_upper"]).copy()

    if plot_df.empty:
        raise ValueError("No valid data available for plotting.")

    # Rename columns for easier plotting
    plot_df = plot_df.rename(
        columns={
            "outcome_variable": "outcome variable",
            "instrument_variable": "instrument variable",
            "ratio_ci_lower": "ci_lower",
            "ratio_ci_upper": "ci_upper",
        }
    )

    # Extract data for plotting
    h_values = plot_df["horizon"].values
    outcome_variable = plot_df["outcome variable"].values[0]
    instrument_variable = plot_df["instrument variable"].values[0]
    ratios = plot_df["ratio"].values
    ci_lower = plot_df["ci_lower"].values
    ci_upper = plot_df["ci_upper"].values
    alpha = plot_df["alpha"].values[0]

    # Create the plot
    fig, ax = create_themed_figure()

    # Plot the ratio line
    ax.plot(h_values, ratios, "o-", color="blue", linewidth=2, markersize=8, label="Wald Ratio")

    # Plot confidence intervals
    ax.fill_between(
        h_values, ci_lower, ci_upper, alpha=0.3, color="blue", label=f"{int((1 - alpha) * 100)}% Confidence Interval"
    )

    # Add horizontal reference lines
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Ratio = 0")

    # Customize the plot
    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_title(
        f"Blanchard-Leigh Efficiency Test: Coefficient Ratios\n"
        f"Outcome: {outcome_variable}, Instrument: {instrument_variable}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Set x-axis to show all horizons
    ax.set_xticks(h_values)

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
