from datetime import date
from typing import Literal, Union

import matplotlib.pyplot as plt

from forecast_evaluation.core.outturns_revisions_table import create_outturn_revisions
from forecast_evaluation.data import ForecastData
from forecast_evaluation.utils import filter_k
from forecast_evaluation.visualisations.theme import create_themed_figure


def plot_outturn_revisions(
    data: ForecastData,
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    k: Union[int, list[int]] = 12,
    fill_k: bool = False,
    ma_window: int = 1,
    start_date: Union[date, str] = None,
    end_date: Union[date, str] = None,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """
    Create outturn revisions plot.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data.
    variable : str
        The variable to analyze (e.g., 'gdpkp', 'cpisa', 'unemp')
    metric : str
        The metric to analyze (e.g., 'yoy', 'pop', 'levels')
    frequency : str
        The frequency to analyze (e.g., 'Q', 'M')
    k : int or list of int, default=12
        Number of revisions used to define the outturns. Can be a single integer
        or a list of integers to compare multiple revision horizons on the same plot.
    fill_k : bool, default=False
        If True, uses only the latest vintage for each date when calculating revisions.
    ma_window : int, default=1
        Size of moving average window to smooth the revisions. By default, no smoothing is applied.
    start_date : date or str, default=None
        The start date for the plot. If None, uses the earliest date in the data.
    end_date : date or str, default=None
        The end date for the plot. If None, uses the latest date in the data.
    convert_to_percentage : bool, default=False
        If True, multiplies values on the y-axis by 100
    return_plot : bool, default=False
        If True, returns the matplotlib figure and axis objects

    Returns
    -------
    tuple or None
        If return_plot is True, returns (fig, ax). Otherwise, returns None.
    """
    # Normalize k to a list
    k_list = [k] if isinstance(k, int) else k

    # Create outturn revisions dataframe
    revisions_df = create_outturn_revisions(data=data)

    # Filter for the specified variable, metric, and frequency
    filtered_df = (
        revisions_df[
            (revisions_df["variable"] == variable)
            & (revisions_df["metric"] == metric)
            & (revisions_df["frequency"] == frequency)
        ]
        .copy()
        .sort_values("date")
    )

    # Apply percentage conversion if requested
    if convert_to_percentage:
        filtered_df["revision"] = 100 * filtered_df["revision"]

    # Create the plot
    fig, ax = create_themed_figure()

    # Process and plot each k value
    for k_value in k_list:
        # Filter dataframe for the current k value
        revisions_df_k = filter_k(filtered_df, k=k_value, fill_k=fill_k).copy()

        # Apply date filters if provided
        if start_date is not None:
            revisions_df_k = revisions_df_k[revisions_df_k["date"] >= start_date]
        if end_date is not None:
            revisions_df_k = revisions_df_k[revisions_df_k["date"] <= end_date]

        # Apply moving average and calculate standard deviation if ma_window > 1
        if ma_window > 1:
            revisions_df_k["revision_ma"] = (
                revisions_df_k["revision"].rolling(window=ma_window, min_periods=ma_window).mean()
            )
            revisions_df_k["revision_std"] = (
                revisions_df_k["revision"].rolling(window=ma_window, min_periods=ma_window).std()
            )
            revisions_df_k["revision_std"] = revisions_df_k["revision_std"].fillna(0)
        else:
            revisions_df_k["revision_ma"] = revisions_df_k["revision"]
            revisions_df_k["revision_std"] = 0

        # Plot the revisions (or moving average)
        label = f"k={k_value}" if len(k_list) > 1 else None
        line = ax.plot(
            revisions_df_k["date"],
            revisions_df_k["revision_ma"],
            marker="o",
            linewidth=2,
            markersize=4,
            alpha=0.7,
            label=label,
        )

        # Add shaded region for standard deviation if ma_window > 1
        if ma_window > 1:
            color = line[0].get_color()
            lower_bound = revisions_df_k["revision_ma"] - revisions_df_k["revision_std"]
            upper_bound = revisions_df_k["revision_ma"] + revisions_df_k["revision_std"]

            ax.fill_between(
                revisions_df_k["date"],
                lower_bound,
                upper_bound,
                alpha=0.2,
                color=color,
            )

    # Add a horizontal line at zero for reference
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Set labels and title
    ax.set_xlabel("Date")
    y_label = "Revision (p.p.)" if convert_to_percentage else "Revision"
    ax.set_ylabel(y_label)

    if ma_window == 1:
        title = f"Outturn Revisions - {variable.upper()}"
    else:
        title = f"Moving Average Outturn Revisions - {variable.upper()}\nMA window={ma_window}"

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add legend if multiple k values
    if len(k_list) > 1:
        ax.legend()

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None


def plot_outturns(
    data: ForecastData,
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"],
    k: Union[int, list[int]] = 12,
    fill_k: bool = True,
    start_date: Union[date, str] = None,
    end_date: Union[date, str] = None,
    convert_to_percentage: bool = False,
    return_plot: bool = False,
):
    """
    Create outturns plot.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data.
    variable : str
        The variable to analyze (e.g., 'gdpkp', 'cpisa', 'unemp')
    metric : str
        The metric to analyze (e.g., 'yoy', 'pop', 'levels')
    frequency : str
        The frequency to analyze (e.g., 'Q', 'M')
    start_date : date or str, default=None
        The start date for the plot. If None, uses the earliest date in the data.
    end_date : date or str, default=None
        The end date for the plot. If None, uses the latest date in the data.
    convert_to_percentage : bool, default=False
        If True, multiplies values on the y-axis by 100
    return_plot : bool, default=False
        If True, returns the matplotlib figure and axis objects

    Returns
    -------
    tuple or None
        If return_plot is True, returns (fig, ax). Otherwise, returns None.
    """
    # Normalize k to a list
    k_list = [k] if isinstance(k, int) else k

    # Create outturn revisions dataframe
    revisions_df = create_outturn_revisions(data=data)

    # Filter for the specified variable, metric, and frequency
    filtered_df = (
        revisions_df[
            (revisions_df["variable"] == variable)
            & (revisions_df["metric"] == metric)
            & (revisions_df["frequency"] == frequency)
        ]
        .copy()
        .sort_values("date")
    )

    # Apply percentage conversion if requested
    if convert_to_percentage:
        filtered_df["value_outturn"] = 100 * filtered_df["value_outturn"]

    # Create the plot
    fig, ax = create_themed_figure()

    # Process and plot each k value
    for k_value in k_list:
        # Filter dataframe for the current k value
        revisions_df_k = filter_k(filtered_df, k=k_value, fill_k=fill_k).copy()

        # Apply date filters if provided
        if start_date is not None:
            revisions_df_k = revisions_df_k[revisions_df_k["date"] >= start_date]
        if end_date is not None:
            revisions_df_k = revisions_df_k[revisions_df_k["date"] <= end_date]

        # Plot the revisions
        label = f"k={k_value}" if len(k_list) > 1 else None
        ax.plot(
            revisions_df_k["date"],
            revisions_df_k["value_outturn"],
            marker="o",
            linewidth=2,
            markersize=4,
            alpha=0.7,
            label=label,
        )

    # Add a horizontal line at zero for reference
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Set labels and title
    ax.set_xlabel("Date")
    # Update y-axis label based on whether values were multiplied
    if convert_to_percentage and metric.lower() != "levels":
        y_label = f"{metric} (%)"
    else:
        y_label = f"{metric}"
    ax.set_ylabel(y_label)

    ax.set_title(f"Outturn vintages - {variable.upper()}")
    ax.grid(True, alpha=0.3)

    # Add legend if multiple k values
    if len(k_list) > 1:
        ax.legend()

    if return_plot:
        return fig, ax
    else:
        plt.show()
        return None
