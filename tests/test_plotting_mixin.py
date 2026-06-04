"""Tests for PlottingMixin methods on ForecastData."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # non-interactive backend; must be set before any figure is created

from forecast_evaluation.data.ForecastData import ForecastData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VARIABLE = "gdpkp"
SOURCE = "mpr"
METRIC = "yoy"


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_plot_hedgehog_returns_fig_ax(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_hedgehog(
        variable=VARIABLE,
        forecast_source=SOURCE,
        metric=METRIC,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_hedgehog_no_return(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_hedgehog(
        variable=VARIABLE,
        forecast_source=SOURCE,
        metric=METRIC,
        return_plot=False,
    )
    assert result is None


def test_plot_forecast_errors_returns_fig_ax(fer_minimal_fd: ForecastData):
    # Pick a vintage date that exists in the minimal data
    vintage = fer_minimal_fd._main_table["vintage_date_forecast"].iloc[0].strftime("%Y-%m-%d")
    result = fer_minimal_fd.plot_forecast_errors(
        variable=VARIABLE,
        metric=METRIC,
        source=SOURCE,
        vintage_date_forecast=vintage,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_forecast_errors_by_horizon_returns_fig_ax(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_forecast_errors_by_horizon(
        variable=VARIABLE,
        source=SOURCE,
        metric=METRIC,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_forecast_errors_by_horizon_multiple_sources(fer_minimal_fd: ForecastData):
    sources = fer_minimal_fd._main_table["unique_id"].unique().tolist()
    result = fer_minimal_fd.plot_forecast_errors_by_horizon(
        variable=VARIABLE,
        source=sources,
        metric=METRIC,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2


def test_plot_outturn_revisions_returns_fig_ax(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_outturn_revisions(
        variable=VARIABLE,
        metric=METRIC,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_outturn_revisions_multiple_k(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_outturn_revisions(
        variable=VARIABLE,
        metric=METRIC,
        k=[4, 8],
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2


def test_plot_outturns_returns_fig_ax(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_outturns(
        variable=VARIABLE,
        metric=METRIC,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_average_revision_by_period_returns_fig_ax(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_average_revision_by_period(
        source=SOURCE,
        variable=VARIABLE,
        metric=METRIC,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_vintage_returns_fig_ax(fer_minimal_fd: ForecastData):
    vintage = fer_minimal_fd._forecasts["vintage_date"].iloc[0].strftime("%Y-%m-%d")
    result = fer_minimal_fd.plot_vintage(
        variable=VARIABLE,
        vintage_date=vintage,
        metric=METRIC,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_errors_across_time_returns_fig_ax(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_errors_across_time(
        variable=VARIABLE,
        metric=METRIC,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)


def test_plot_errors_across_time_absolute_error(fer_minimal_fd: ForecastData):
    result = fer_minimal_fd.plot_errors_across_time(
        variable=VARIABLE,
        metric=METRIC,
        error="absolute",
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2


def test_plot_errors_across_time_multiple_horizons(fer_minimal_fd: ForecastData):
    horizons = fer_minimal_fd._main_table["forecast_horizon"].unique()[:2].tolist()
    result = fer_minimal_fd.plot_errors_across_time(
        variable=VARIABLE,
        metric=METRIC,
        horizons=horizons,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2


def test_plot_forecast_error_density_returns_fig_ax(fer_minimal_fd: ForecastData):
    horizon = fer_minimal_fd._main_table["forecast_horizon"].iloc[0]
    result = fer_minimal_fd.plot_forecast_error_density(
        variable=VARIABLE,
        horizon=horizon,
        metric=METRIC,
        source=SOURCE,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
