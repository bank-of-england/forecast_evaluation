import matplotlib.pyplot as plt

THEME = {
    "figure": {
        "figsize": (10, 6),
        "constrained_layout": True,
    },
    "plot": {
        "linewidth": 2,
        "marker": "o",
        "markersize": 6,
    },
    "axes": {
        "grid": True,
        "titlesize": 12,
        "labelsize": 10,
    },
    "legend": {
        "fontsize": 10,
    },
}


def apply_theme(fig, ax):
    """Apply the shared theme to a Matplotlib figure and axes."""
    # Set default plot styling that will apply to all plot calls on this axes
    ax.set_prop_cycle(
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )

    # Set default plot parameters
    plt.rcParams["lines.linewidth"] = THEME["plot"]["linewidth"]

    # Apply axes styling
    ax.grid(THEME["axes"]["grid"])
    ax.title.set_size(THEME["axes"]["titlesize"])
    ax.xaxis.label.set_size(THEME["axes"]["labelsize"])
    ax.yaxis.label.set_size(THEME["axes"]["labelsize"])


def create_themed_figure(nrows=1, ncols=1, **kwargs):
    """
    Create a matplotlib figure with theme applied from the start.

    Parameters
    ----------
    nrows : int, default=1
        Number of subplot rows
    ncols : int, default=1
        Number of subplot columns
    **kwargs : dict
        Additional arguments passed to plt.subplots() (e.g., figsize, sharex, sharey)

    Returns
    -------
    fig, ax : tuple
        Figure and axes objects with theme applied
    """

    fig_kwargs = {
        "figsize": THEME["figure"]["figsize"],
        "constrained_layout": THEME["figure"]["constrained_layout"],
    }
    fig_kwargs.update(kwargs)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, **fig_kwargs)
    apply_theme(fig, ax if nrows == 1 and ncols == 1 else ax.flat[0])

    return fig, ax
