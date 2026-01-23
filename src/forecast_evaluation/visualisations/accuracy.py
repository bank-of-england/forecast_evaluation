from typing import TYPE_CHECKING, Literal, Union

import matplotlib.pyplot as plt
import pandas as pd

from forecast_evaluation.visualisations.theme import create_themed_figure

if TYPE_CHECKING:
    from forecast_evaluation.tests.results import TestResult


def plot_accuracy(
    df: Union[pd.DataFrame, "TestResult"],
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"] = "Q",
    statistic: Literal["rmse", "rmedse", "mse", "mean_abs_error"] = "rmse",
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """Plot accuracy statistic for all sources by forecast horizon for a specific variable and metric combination.

    Creates a line plot showing how the selected accuracy statistic varies across forecast horizons
    for different forecast sources. The default statistic is RMSE, but other measures like RMEDSE,
    MSE, or MAE can be selected.

    Parameters
    ----------
    df : DataFrame or AccuracyResults object
        containing accuracy statistics with columns 'variable', 'source', 'metric',
        'frequency', 'forecast_horizon', 'rmse', 'rmedse', 'mse', 'mean_abs_error',
        'n_observations', 'start_date', 'end_date'.

    variable : str
        Variable to analyze (e.g., 'gdpkp', 'cpisa', 'unemp').

    metric : str
        Metric to analyze ('levels', 'pop', or 'yoy').

    frequency : str
        Data frequency ('Q' for quarterly or 'M' for monthly).

    statistic : str
        Accuracy statistic to plot ('rmse', 'rmedse', 'mse', or 'mean_abs_error').

    convert_to_percentage : bool
        If True, multiplies values on the y-axis by 100.

    return_plot : bool
        If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        If return_plot is True, returns the figure and axes objects. Otherwise, displays the plot and returns None.
    """
    # Extract DataFrame if TestResult object is passed
    if hasattr(df, "to_df"):
        df = df.to_df()

    # Filter data for the specific combination
    mask = (df["variable"] == variable) & (df["metric"] == metric) & (df["frequency"] == frequency)

    df = df.loc[mask].copy()

    if len(df) == 0:
        raise ValueError(f"No data available for {variable}, {metric}")

    # Get the consistent start date and end date for each variable
    min_start_date = df["start_date"].min()
    min_start_date_str = f"{min_start_date.year}:Q{((min_start_date.month) // 3)}"
    max_end_date = df["end_date"].max()
    max_end_date_str = f"{max_end_date.year}:Q{((max_end_date.month) // 3)}"

    # Get unique sources
    sources = df["unique_id"].unique()

    # Multiply by 100 if convert_to_percentage = True
    multiplier = 100 if convert_to_percentage else 1

    # Create the plot
    fig, ax = create_themed_figure()

    # Plot accuracy statistic for each source
    for source in sources:
        source_data = df[df["unique_id"] == source].copy()
        source_data = source_data.sort_values("forecast_horizon")

        ax.plot(
            source_data["forecast_horizon"],
            multiplier * source_data[statistic],
            marker="o",
            linewidth=2,
            markersize=6,
            label=source,
        )

    # Relabel statistics for y-axis labels
    stat_labels = {
        "rmse": "Root Mean Square Error",
        "rmedse": "Root Median Square Error",
        "mse": "Mean Square Error",
        "mean_abs_error": "Mean Absolute Error",
    }
    stat_label = stat_labels.get(statistic.lower(), statistic)

    # Customize the plot
    ax.set_title(
        f"{stat_label} by Forecast Horizon\n{variable.upper()} - {metric} \n"
        + f"Forecasts from {min_start_date_str} to {max_end_date_str}",
        fontsize=14,
    )
    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_ylabel(f"{stat_label}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="unique_id", loc="best")

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None


def plot_compare_to_benchmark(
    df: pd.DataFrame,
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    benchmark_model: str,
    statistic: Literal["rmse", "rmedse", "mean_abs_error"] = "rmse",
    return_plot: bool = False,
):
    """Plot the ratio of each model's accuracy statistic to a benchmark model's statistic by forecast horizon.

    Parameters
    ----------
    df : DataFrame
        containing accuracy statistics with columns 'variable', 'source', 'metric',
        'frequency', 'forecast_horizon', 'rmse', 'rmedse', 'mean_abs_error',
        'n_observations', 'start_date', 'end_date'.
    variable : str
        Variable to analyze (e.g., 'gdpkp', 'cpisa', 'unemp').
    metric : str
        Metric to analyze (e.g., 'yoy', 'levels').
    frequency : str
        Data frequency ('Q' for quarterly or 'M' for monthly).
    benchmark_model : str
        The forecast source to use as the benchmark for comparison (e.g., 'mpr').
    statistic : str
        The accuracy statistic to compare ('rmse', 'rmedse', or 'mean_abs_error').
    return_plot : bool
        If True, returns (fig, ax) tuple instead of displaying the plot.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        If return_plot is True, returns the figure and axes objects. Otherwise, displays the plot and returns None.
    """
    from forecast_evaluation.tests.accuracy import compare_to_benchmark

    # Compute comparison to benchmark model
    df = compare_to_benchmark(df, benchmark_model=benchmark_model, statistic=statistic)

    # Extract the ratio column name
    ratio_col = f"{statistic}_to_benchmark"

    # Filter data for the specific combination
    mask = (
        (df["variable"] == variable)
        & (df["unique_id"] != benchmark_model)
        & (df["metric"] == metric)
        & (df["frequency"] == frequency)
    )

    df = df.loc[mask].copy()

    if len(df) == 0:
        raise ValueError(f"No data available for {variable}, {metric}")

    # Get minimum start_date
    min_start_date = df["start_date"].min()
    min_start_date_str = f"{min_start_date.year}:Q{((min_start_date.month) // 3)}"
    max_end_date = df["end_date"].max()
    max_end_date_str = f"{max_end_date.year}:Q{((max_end_date.month) // 3)}"

    # Get unique sources
    sources = df["unique_id"].unique()

    # Create the plot
    fig, ax = create_themed_figure()

    # Plot accuracy statistic ratio for each source
    for source in sources:
        source_data = df[df["unique_id"] == source].copy()
        source_data = source_data.sort_values("forecast_horizon")

        ax.plot(
            source_data["forecast_horizon"], source_data[ratio_col], marker="o", linewidth=2, markersize=6, label=source
        )

    # Relabel statistics for y-axis labels
    stat_labels = {
        "rmse": "Root Mean Square Error",
        "rmedse": "Root Median Square Error",
        "mse": "Mean Square Error",
        "mean_abs_error": "Mean Absolute Error",
    }
    stat_label = stat_labels.get(statistic.lower(), statistic)

    # Customize the plot
    ax.set_title(
        f"{stat_label} Relative to Benchmark by Forecast Horizon\n"
        + f"{variable.upper()} - {metric}\nForecasts from {min_start_date_str} to {max_end_date_str}",
        fontsize=14,
    )
    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_ylabel(f"{stat_label} Relative to Benchmark", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="unique_id", loc="best")

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None


def plot_rolling_relative_accuracy(df: pd.DataFrame, variable: str, horizons: list[int], return_plot: bool = False):
    """
    Plot RMSE ratio relative to benchmark across rolling windows,
    with significance levels indicated by dot colors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: variable, source, forecast_horizon, rmse_ratio,
        p_value, window_start, window_end
    variable : str
        Variable to plot (used for title)
    horizons : list[int]
        List of forecast horizons to include in the plot
    return_plot : bool
        If True, returns (fig, axes) instead of displaying

    Returns
    -------
    tuple or None
        (fig, axes) if return_plot=True, else None
    """
    from matplotlib.lines import Line2D

    # Ensure window_end is datetime
    df = df.copy()
    df["window_end"] = pd.to_datetime(df["window_end"])

    # Filter for specified horizons
    df = df[df["forecast_horizon"].isin(horizons)]

    # Create figure with subplots for each horizon
    n_horizons = len(horizons)
    fig, axes = create_themed_figure(nrows=n_horizons, ncols=1, sharex=True, figsize=(10, 4 * n_horizons))

    if n_horizons == 1:
        axes = [axes]

    # Get unique sources
    sources = df["unique_id"].unique()

    # Color palette for sources (lines)
    colors = plt.cm.tab10(range(len(sources)))
    source_colors = dict(zip(sources, colors))

    # Check if the fluctuation stat is available
    if "critical_value_05" in df.columns:
        df["p_value"] = 1.0  # turn off p-value results below
    else:
        # turn off fluctuation result checks
        df["reject_05"] = False
        df["reject_10"] = False

    for i, h in enumerate(horizons):
        ax = axes[i]
        sub = df[df["forecast_horizon"] == h].copy()

        for source in sources:
            source_data = sub[sub["unique_id"] == source].sort_values("window_end")

            # Plot line
            ax.plot(
                source_data["window_end"],
                source_data["rmse_ratio"],
                color=source_colors[source],
                linewidth=2,
                label=source,
                zorder=1,
            )

            # Plot dots with significance colors
            for _, row in source_data.iterrows():
                p_val = row["p_value"]
                # Determine dot color based on p-value
                if p_val < 0.01:
                    dot_color = "#8B0000"  # Dark red
                    alpha = 1
                    size = 80
                elif p_val < 0.05 or row["reject_05"]:
                    dot_color = "#DC143C"  # Crimson red
                    alpha = 1
                    size = 60
                elif p_val < 0.1 or row["reject_10"]:
                    dot_color = "#FF6B6B"  # Light red
                    alpha = 1
                    size = 40
                else:
                    dot_color = source_colors[source]
                    alpha = 0.7
                    size = 30

                ax.scatter(
                    row["window_end"],
                    row["rmse_ratio"],
                    color=dot_color,
                    s=size,
                    alpha=alpha,
                    zorder=2,
                    edgecolors="white",
                    linewidths=0.5,
                )

        # Add reference line at 1.0
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)

        # Customize subplot
        ax.set_ylabel(f"Horizon {h}")
        ax.grid(True, alpha=0.3)

        # Add legend only to first subplot
        if i == 0:
            # Create custom legend handles
            source_handles = [
                Line2D([0], [0], color=source_colors[source], linewidth=2, label=source) for source in sources
            ]

            # Add significance level handles
            sig_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#FF6B6B",
                    markersize=6,
                    label="p < 0.1",
                    linestyle="None",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#DC143C",
                    markersize=7,
                    label="p < 0.05",
                    linestyle="None",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#8B0000",
                    markersize=8,
                    label="p < 0.01",
                    linestyle="None",
                ),
            ]

            # Combine legends
            first_legend = ax.legend(handles=source_handles, title="unique_id", loc="upper left", fontsize=9)
            ax.add_artist(first_legend)

            ax.legend(handles=sig_handles, title="Significance", loc="upper right", fontsize=9)

    # Set x-label on bottom subplot
    axes[-1].set_xlabel("Window End Date")

    # Add overall title
    fig.suptitle(
        f"Rolling RMSE Ratio vs Benchmark - {variable.upper()}",
        y=0.995,
    )

    if return_plot:
        return fig, axes
    else:
        plt.show()
        return None


# Example usage:
if __name__ == "__main__":
    import pandas as pd

    import forecast_evaluation as fe

    # Initialise with fer ---------
    forecast_data = fe.ForecastData(load_fer=True)

    forecast_data.filter(variables=["gdpkp"], metrics=["yoy"], sources=["mpr", "baseline random walk model"])

    rolling_dm = fe.rolling_analysis(
        data=forecast_data,
        window_size=40,
        analysis_func=fe.diebold_mariano_table,
        analysis_args={"benchmark_model": "mpr"},
    )

    plot_rolling_relative_accuracy(df=rolling_dm, variable="gdpkp", horizons=[1, 4, 8])
