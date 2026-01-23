"""Utility functions for the dashboard."""

import sys
import traceback
from functools import wraps
from shiny import render


# Store original render.plot
_original_render_plot = render.plot


def safe_render_plot(*outer_args, **outer_kwargs):
    """Wrapper for render.plot that handles all errors gracefully."""

    def decorator(func):
        @wraps(func)
        def error_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error for debugging
                print(f"Plot error: {traceback.format_exc()}", file=sys.stderr)

                # Return a user-friendly message figure
                from forecast_evaluation.visualisations.theme import create_themed_figure

                fig, ax = create_themed_figure()
                ax.text(
                    0.5,
                    0.5,
                    "No results available with the current settings. Click 'Update' or try a different selection.",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="#415265",
                )
                ax.axis("off")
                return fig

        # Apply the original render.plot decorator to our error wrapper
        return _original_render_plot(*outer_args, **outer_kwargs)(error_wrapper)

    # Handle both @render.plot and @render.plot() syntax
    if len(outer_args) == 1 and callable(outer_args[0]) and not outer_kwargs:
        # Called as @render.plot (without parentheses)
        return decorator(outer_args[0])
    else:
        # Called as @render.plot(...) (with arguments)
        return decorator


def patch_render_plot():
    """Monkey-patch render.plot to use safe error handling."""
    render.plot = safe_render_plot


def apply_legend_visibility(fig, ax, show_legend: bool):
    """Helper to show/hide legend on a plot without modifying original."""

    import copy

    # Create a copy to avoid modifying the original
    fig = copy.deepcopy(fig)
    ax = copy.deepcopy(ax) if ax is not None else None

    if not show_legend:
        # Handle both single axes and array of axes
        if ax is None:
            return fig

        axes_list = ax.flat if hasattr(ax, "flat") else [ax]

        # Remove axis-level legends
        for axis in axes_list:
            if axis is None:
                continue
            try:
                legend = axis.get_legend()
                if legend is not None:
                    legend.remove()
            except Exception:
                pass

        # Remove figure-level legends
        try:
            while len(fig.legends) > 0:
                fig.legends[0].remove()
        except Exception:
            pass

    return fig


def render_legend(plot, show_legend):
    """Render legend separately from the main plot."""
    import matplotlib.pyplot as plt

    if show_legend:
        return plt.figure()

    _, legend_ax = plot

    # Collect handles and labels from all axes
    handles, labels = [], []
    axes_list = legend_ax.flat if hasattr(legend_ax, "flat") else [legend_ax]

    for ax in axes_list:
        if ax is None:
            continue
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:  # Avoid duplicates
                handles.append(handle)
                labels.append(label)

    #  new figure with only the legend
    fig_legend = plt.figure()
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")

    ax_legend.legend(handles, labels, loc="upper center", frameon=False)

    return fig_legend


def remove_legend(ax):
    """Remove legend from a single axis or array of axes."""
    if ax is None:
        return

    axes_list = ax.flat if hasattr(ax, "flat") else [ax]
    for axis in axes_list:
        if axis is None:
            continue
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
