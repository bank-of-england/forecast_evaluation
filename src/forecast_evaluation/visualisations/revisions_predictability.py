import matplotlib.pyplot as plt

from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_average_revision_by_period(data, source, variable, metric, frequency, return_plot: bool = False):
    """Plot the average revision grouped by forecast_horizon.

    Creates a line plot showing how the average size of forecast revisions
    varies by forecast horizon (periods until final revision).

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecasts data.
    source : str
        Forecast source identifier (e.g., 'mpr').
    variable : str
        Variable to analyze (e.g., 'gdpkp', 'cpisa').
    metric : str
        Metric to analyze ('levels', 'pop', or 'yoy').
    frequency : str
        Data frequency ('Q' for quarterly or 'M' for monthly).
    return_plot : bool, default False
        If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        If return_plot is True, returns the figure and axes objects. Otherwise, displays the plot and returns None.
    """
    # Prepare revisions data
    if data._forecasts is None:
        raise ValueError("ForecastData forecasts is not available. Please ensure data has been added and processed.")

    forecasts = data._forecasts.copy()

    df = forecasts[
        (forecasts["variable"] == variable)
        & (forecasts["unique_id"] == source)
        & (forecasts["metric"] == metric)
        & (forecasts["frequency"] == frequency)
    ].copy()

    df = df.sort_values(by=["date", "vintage_date"], ascending=True).reset_index(drop=True)

    df["revision"] = df.groupby(["variable", "unique_id", "metric", "frequency", "date"])["value"].diff()

    # Calculate average revision by forecast_horizon, ignoring NAs
    avg_revision = df.groupby("forecast_horizon")["revision"].mean().dropna().reset_index()

    # Create the plot
    fig, ax = create_themed_figure()

    # Plot the average revisions
    ax.plot(-avg_revision.index, 100 * avg_revision["revision"], "o-", color="blue", linewidth=2, markersize=8)

    # Add horizontal reference line at zero
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)

    # Customize the plot
    ax.set_xlabel("Periods until last revision", fontsize=12)
    ax.set_ylabel("Average Revision (pp)", fontsize=12)
    ax.set_title("Average Revision by Forecast Horizon", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show all forecast horizons
    ax.set_xticks(-avg_revision.index)

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
