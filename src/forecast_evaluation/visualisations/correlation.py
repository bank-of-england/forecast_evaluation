import warnings
from typing import TYPE_CHECKING, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from forecast_evaluation.utils import clean_unique_id
from forecast_evaluation.visualisations.theme import create_themed_figure

if TYPE_CHECKING:
    from forecast_evaluation.tests.results import TestResult


def _clean_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_unique_id to both pair-id columns."""
    df = clean_unique_id(df)
    df["unique_id_b"] = df["unique_id_b"].map(clean_unique_id)
    return df


def plot_correlation_heatmap(
    df: Union[pd.DataFrame, "TestResult"],
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    horizon: int,
    frequency: Optional[Literal["Q", "M"]] = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
    return_plot: bool = False,
):
    """Plot pairwise correlation of forecast errors as a single heatmap.

    Parameters
    ----------
    df : DataFrame or TestResult
        Output of :func:`forecast_errors_correlation_analysis`. Must contain
        columns 'variable', 'metric', 'frequency', 'forecast_horizon',
        'unique_id', 'unique_id_b', 'correlation'.
    variable : str
        Variable to plot (e.g. 'gdpkp').
    metric : {"levels", "pop", "yoy"}
        Metric to plot.
    horizon : int
        Forecast horizon to plot.
    cmap : str, default 'RdBu_r'
        Matplotlib colormap. Diverging maps centred at 0 are recommended.
    annotate : bool, default True
        If True, annotate each cell with the correlation value.
    return_plot : bool, default False
        If True, returns (fig, ax) instead of displaying.

    Returns
    -------
    tuple or None
        (fig, ax) if return_plot is True, else None.
    """
    if hasattr(df, "to_df"):
        df = df.to_df()

    df = df.copy()

    if frequency is not None:
        warnings.warn(
            "The 'frequency' argument is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    df = df[(df["variable"] == variable) & (df["metric"] == metric) & (df["forecast_horizon"] == horizon)]

    if df.empty:
        raise ValueError(
            f"No correlation data available for variable='{variable}', metric='{metric}', horizon={horizon}."
        )

    df = _clean_pair_columns(df)

    sources = sorted(set(df["unique_id"]).union(df["unique_id_b"]))

    cell = max(0.55, 6.0 / max(len(sources), 1))
    width = max(6.0, cell * len(sources) + 2.0)
    height = max(4.0, cell * len(sources) + 1.5)

    # GridSpec gives both the heatmap and colorbar a real SubplotSpec, so
    # Shiny's plot-rendering layout pass doesn't trip on a None subplotspec.
    fig = plt.figure(figsize=(width, height), constrained_layout=False)
    gs = GridSpec(
        nrows=1,
        ncols=2,
        figure=fig,
        width_ratios=[1, 0.04],
        left=0.20,
        right=0.90,
        top=0.92,
        bottom=0.18,
    )

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    matrix = df.pivot_table(index="unique_id", columns="unique_id_b", values="correlation", aggfunc="mean").reindex(
        index=sources, columns=sources
    )

    im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(sources)))
    ax.set_yticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sources, fontsize=8)
    ax.grid(False)

    if annotate:
        arr = matrix.to_numpy()
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                if np.isnan(val):
                    continue
                text_color = "white" if abs(val) > 0.5 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Correlation")

    fig.suptitle(
        f"Forecast error correlation - {variable.upper()} ({metric}, horizon {horizon})",
        y=0.98,
    )

    if return_plot:
        return fig, ax
    plt.show()
    return None


def plot_rolling_correlation(
    df: Union[pd.DataFrame, "TestResult"],
    variable: str,
    anchor_source: str,
    horizons: Union[int, list[int]],
    metric: Optional[Literal["levels", "pop", "yoy"]] = None,
    frequency: Optional[Literal["Q", "M"]] = None,
    return_plot: bool = False,
):
    """Plot rolling forecast-error correlation between an anchor source and others.

    Mirrors the layout of :func:`plot_rolling_relative_accuracy`: one subplot
    per horizon, one line per partner source, x = window_end.

    Parameters
    ----------
    df : DataFrame or TestResult
        Output of ``rolling_analysis(analysis_func=forecast_errors_correlation_analysis, ...)``.
        Must contain columns 'variable', 'metric', 'frequency', 'forecast_horizon',
        'unique_id', 'unique_id_b', 'correlation', 'window_start', 'window_end'.
    variable : str
        Variable to plot.
    anchor_source : str
        Source held fixed across the lines; one line is drawn per other source
        showing its rolling correlation with the anchor.
    horizons : int or list of int
        Forecast horizon(s) to plot.
    metric : {"levels", "pop", "yoy"} or None, default None
        Metric to plot. If None, inferred from the data when unique.
    return_plot : bool, default False
        If True, returns (fig, axes) instead of displaying.

    Returns
    -------
    tuple or None
        (fig, axes) if return_plot is True, else None.
    """
    if hasattr(df, "to_df"):
        df = df.to_df()

    df = df.copy()
    df["window_end"] = pd.to_datetime(df["window_end"])

    if isinstance(horizons, int):
        horizons = [horizons]

    if metric is None:
        unique_metrics = df["metric"].unique()
        if len(unique_metrics) != 1:
            raise ValueError(
                f"Could not infer a unique metric from data; found: {list(unique_metrics)}. "
                "Please specify the 'metric' argument explicitly."
            )
        metric = unique_metrics[0]

    df = _clean_pair_columns(df)
    anchor_source_clean = clean_unique_id(anchor_source)

    if frequency is not None:
        warnings.warn(
            "The 'frequency' argument is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    df = df[
        (df["variable"] == variable)
        & (df["metric"] == metric)
        & (df["unique_id"] == anchor_source_clean)
        & (df["unique_id_b"] != anchor_source_clean)
    ]
    df = df[df["forecast_horizon"].isin(horizons)]

    if df.empty:
        raise ValueError(
            f"No rolling correlation data for variable='{variable}', anchor='{anchor_source}', horizons={horizons}."
        )

    n = len(horizons)
    fig, axes = create_themed_figure(nrows=n, ncols=1, sharex=True, figsize=(10, 4 * n))
    axes_list = [axes] if n == 1 else list(np.array(axes).flat)

    partners = sorted(df["unique_id_b"].unique())
    colors = plt.cm.tab10(range(len(partners)))
    color_map = dict(zip(partners, colors))

    for i, h in enumerate(horizons):
        ax = axes_list[i]
        sub = df[df["forecast_horizon"] == h]

        for partner in partners:
            line_data = sub[sub["unique_id_b"] == partner].sort_values("window_end")
            if line_data.empty:
                continue
            ax.plot(
                line_data["window_end"],
                line_data["correlation"],
                color=color_map[partner],
                linewidth=2,
                marker="o",
                markersize=4,
                label=partner,
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"Horizon {h}")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="best", fontsize=9)

    axes_list[-1].set_xlabel("Window end date")

    fig.suptitle(
        f"Rolling forecast-error correlation vs. {anchor_source_clean} - {variable.upper()} ({metric})",
        y=0.995,
    )

    if return_plot:
        return fig, axes
    plt.show()
    return None


# Example usage:
if __name__ == "__main__":
    import forecast_evaluation as fe

    forecast_data = fe.ForecastData(load_fer=True)
    forecast_data.filter(variables=["gdpkp"], metrics=["yoy"])

    corr_result = fe.forecast_errors_correlation_analysis(forecast_data, k=12)
    plot_correlation_heatmap(corr_result, variable="gdpkp", metric="yoy", horizons=[1, 4, 8])
