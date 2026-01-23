from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from forecast_evaluation.data import ForecastData
from forecast_evaluation.utils import filter_k
from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_errors_across_time(
    data: ForecastData,
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    error: Literal["raw", "absolute", "squared"] = "raw",
    horizons: int | list[int] = None,
    sources: str | list[str] = None,
    frequency: Literal["Q", "M"] = None,
    k: int = 12,
    ma_window: int = 1,
    show_mean: bool = True,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
    custom_labels: dict = None,
    existing_plot: tuple = None,
):
    """
    Plot average forecast errors by forecast horizon, averaged over all forecast vintages.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data.
    variable : str
        The variable to analyze (e.g., 'gdpkp', 'cpisa', 'unemp')
    metric : str
        The metric to analyze (e.g., 'yoy', 'pop', 'levels')
    error : str, default='raw'
        The type of error to plot. Options are 'raw', 'absolute', or 'squared'.
    horizons : int or list[int], default=None
        The forecast horizon(s) to analyze. If None, the minimum horizon in the data is used.
        If a list is provided, creates faceted subplots by horizon.
    sources : str or list[str], default=None
        The source(s) of the forecasts (e.g., 'compass conditional', 'mpr'). If None, all sources in the data are used.
    frequency : str, default=None
        The frequency to analyze (e.g., 'Q', 'M'). If None the most prevalent frequency in the data is used.
    k : int, default=12
        Number of revisions used to define the outturns.
    ma_window : int, default=1
        Size of moving average window to smooth the errors. By default, no smoothing is applied (ma_window=1).
    show_mean : bool, default=True
        If True, displays horizontal dashed lines showing the mean error for each source.
    convert_to_percentage : bool, default=False
        If True, multiplies values on the y-axis by 100
    return_plot : bool, default=False
        If True, returns the matplotlib figure and axis objects
    custom_labels : dict, default=None
        A dictionary mapping source names to custom labels for the legend.
    existing_plot : tuple, default=None
        A tuple (fig, axes) from a previous call to this function. If provided, new data is added to these axes.

    Returns
    -------
    tuple or None
        If return_plot is True, returns (fig, ax). Otherwise, returns None.
    """

    forecast_errors = data._main_table.copy()

    if sources is None:
        sources = forecast_errors["unique_id"].unique().tolist()
    elif isinstance(sources, str):
        sources = [sources]

    if frequency is None:
        frequency = forecast_errors["frequency"].mode()[0]

    if horizons is None:
        horizons = [forecast_errors["forecast_horizon"].min()]
    elif isinstance(horizons, int):
        horizons = [horizons]

    # filter
    data_filtered = data.copy()
    data_filtered.filter(variables=variable, metrics=metric, frequencies=frequency, sources=sources)
    forecast_errors = data_filtered._main_table.copy()
    forecast_errors = filter_k(forecast_errors, k=k)

    # Determine subplot layout
    n_horizons = len(horizons)

    if existing_plot is not None:
        fig, axes = existing_plot
        # Ensure axes is a list/array for consistent indexing
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        if len(axes) != n_horizons:
            raise ValueError(
                f"Number of horizons ({n_horizons}) does not match number of axes in existing_plot ({len(axes)})."
            )
    else:
        fig, axes = create_themed_figure(nrows=n_horizons, ncols=1, sharex=True)

        if n_horizons == 1:
            axes = [axes]

    for idx, h in enumerate(horizons):
        horizon_errors = forecast_errors[forecast_errors["forecast_horizon"] == h].copy()
        ax = axes[idx]

        if convert_to_percentage:
            horizon_errors["forecast_error"] = 100 * horizon_errors["forecast_error"]

        horizon_errors = horizon_errors[["date", "unique_id", "forecast_error"]]

        if error == "raw":
            horizon_errors["error_to_plot"] = horizon_errors["forecast_error"]
        elif error == "absolute":
            horizon_errors["error_to_plot"] = horizon_errors["forecast_error"].abs()
        elif error == "squared":
            horizon_errors["error_to_plot"] = horizon_errors["forecast_error"] ** 2

        if ma_window > 1:
            horizon_errors = (
                horizon_errors.sort_values("date")
                .groupby("unique_id")[horizon_errors.columns]
                .apply(
                    lambda x: x.assign(
                        error_to_plot=x["error_to_plot"].rolling(window=ma_window, min_periods=ma_window).mean(),
                        error_std=x["error_to_plot"].rolling(window=ma_window, min_periods=ma_window).std(),
                    )
                )
                .reset_index(drop=True)
            )
            horizon_errors["error_std"] = horizon_errors["error_std"].fillna(0)
        else:
            horizon_errors["error_std"] = 0

        linestyle = "none" if ma_window == 1 else "-"

        for source in sources:
            source_data = horizon_errors[horizon_errors["unique_id"] == source]

            display_name = source
            if custom_labels is not None and source in custom_labels:
                display_name = custom_labels[source]

            if ma_window == 1:
                label = f"{display_name} (mean {source_data['error_to_plot'].mean():.2f})"
            else:
                label = display_name

            # Only add label to first subplot to avoid duplicates in legend extraction
            plot_label = label if idx == 0 else None

            line = ax.plot(
                source_data["date"],
                source_data["error_to_plot"],
                marker="o",
                linestyle=linestyle,
                linewidth=2,
                markersize=4,
                label=plot_label,
                alpha=0.7,
            )

            color = line[0].get_color()

            if ma_window > 1:
                if error == "raw":
                    lower_bound = source_data["error_to_plot"] - source_data["error_std"]
                else:
                    lower_bound = np.maximum(source_data["error_to_plot"] - source_data["error_std"], 0.0)
                ax.fill_between(
                    source_data["date"],
                    lower_bound,
                    source_data["error_to_plot"] + source_data["error_std"],
                    alpha=0.2,
                    color=color,
                )

            if show_mean and ma_window == 1:
                mean_error = source_data["error_to_plot"].mean()
                ax.axhline(
                    y=mean_error,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )

        # Subplot title
        ax.set_title(f"Forecast Horizon = {h}")
        ax.set_xlabel("Date")

        y_label = "Forecast Error (p.p.)" if convert_to_percentage else "Forecast Error"
        if idx == (n_horizons - 1) // 2:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        # Only show legend in the first subplot
        if idx == 0:
            ax.legend()
        else:
            # Remove legend from other subplots to avoid duplication
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # Overall title
    if ma_window == 1:
        suptitle = f"Forecast errors - {variable.upper()}"
    else:
        suptitle = f"Moving average {error} error (with ±1 std dev. bands) - {variable.upper()}\nMA window={ma_window}"

    fig.suptitle(suptitle)

    if return_plot:
        return fig, axes
    else:
        plt.show()
        return None
