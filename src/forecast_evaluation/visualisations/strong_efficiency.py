from typing import TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import pandas as pd

from forecast_evaluation.visualisations.theme import create_themed_figure

if TYPE_CHECKING:
    from forecast_evaluation.tests.results import TestResult


def plot_strong_efficiency(results: Union[pd.DataFrame, "TestResult"], return_plot: bool = False):
    """Plot strong efficiency test coefficients across forecast horizons with confidence intervals.

    This function creates a visualization of the strong efficiency test results, showing how the
    OLS coefficient (from regressing forecast errors on instrument variables) varies across
    different forecast horizons. The plot includes confidence intervals and a reference line
    at zero to assess statistical significance.

    Parameters
    ----------
    results: DataFrame or StrongEfficiencyResults object containing strong efficiency test
        results for multiple horizons. Expected columns include 'horizon', 'outcome_variable',
        'instrument_variable', 'ols_coefficient', 'ols_se', 'coeff_ci_lower',
        'coeff_ci_upper', 'n_observations', 'alpha'.

    return_plot: If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None: If return_plot is True,
    returns the figure and axes objects. Otherwise, displays the plot and returns None.
    """
    # Extract DataFrame if TestResult object is passed
    if hasattr(results, "to_df"):
        results = results.to_df()

    # Filter out rows with missing data
    plot_df = results.dropna(subset=["ols_coefficient", "coeff_ci_lower", "coeff_ci_upper"]).copy()

    if plot_df.empty:
        raise ValueError("No valid data available for plotting.")

    # Rename columns for easier plotting
    plot_df = plot_df.rename(
        columns={
            "outcome_variable": "outcome variable",
            "instrument_variable": "instrument variable",
            "n_observations": "n_obs",
        }
    )

    # Extract data for plotting
    h_values = plot_df["horizon"].values
    outcome_variable = plot_df["outcome variable"].values[0]
    instrument_variable = plot_df["instrument variable"].values[0]
    coeffs = plot_df["ols_coefficient"].values
    ci_lower = plot_df["coeff_ci_lower"].values
    ci_upper = plot_df["coeff_ci_upper"].values
    alpha = plot_df["alpha"].values[0]

    # Create the plot
    fig, ax = create_themed_figure()

    # Plot the coefficients
    ax.plot(h_values, coeffs, "o-", color="blue", linewidth=2, markersize=8, label="Coefficient")

    # Plot confidence intervals
    ax.fill_between(
        h_values, ci_lower, ci_upper, alpha=0.3, color="blue", label=f"{int((1 - alpha) * 100)}% Confidence Interval"
    )

    # Add horizontal reference lines
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Coefficient = 0")

    # Customize the plot
    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_title(
        f"Strong Efficiency Test: Coefficient\nOutcome: {outcome_variable}, Instrument: {instrument_variable}",
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
