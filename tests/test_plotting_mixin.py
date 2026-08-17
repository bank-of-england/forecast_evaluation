"""Tests for PlottingMixin methods on ForecastData."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # non-interactive backend; must be set before any figure is created

from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData
from forecast_evaluation.data.sample_data import (
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VARIABLE = "gdpkp"
SOURCE = "mpr"
METRIC = "yoy"

NOWCAST_VARIABLE = "gdp"
NOWCAST_SOURCE = "nowcast_dfm"
NOWCAST_METRIC = "levels"


@pytest.fixture
def nowcast_fd() -> NowcastData:
    """A NowcastData instance built from the sample nowcast outturns/forecasts."""
    fd = NowcastData(outturns_data=create_sample_nowcast_outturns())
    fd.add_forecasts(create_sample_nowcast_forecasts(), data_check=False)
    return fd


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
    horizons = fer_minimal_fd._main_table["target_minus_vintage"].unique()[:2].tolist()
    result = fer_minimal_fd.plot_errors_across_time(
        variable=VARIABLE,
        metric=METRIC,
        horizons=horizons,
        return_plot=True,
    )
    assert isinstance(result, tuple) and len(result) == 2


def test_plot_forecast_error_density_returns_fig_ax(fer_minimal_fd: ForecastData):
    horizon = fer_minimal_fd._main_table["target_minus_vintage"].iloc[0]
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


# ---------------------------------------------------------------------------
# plot_hedgehog releases filter
# ---------------------------------------------------------------------------


def _line_vintage_count(ax: plt.Axes) -> int:
    """Number of forecast vintage lines drawn on the hedgehog plot's axes."""
    # One line per vintage plus a line for the outturns series ("Outturns" label).
    return sum(1 for line in ax.get_lines() if line.get_label() != "Outturns")


def test_plot_hedgehog_releases_filters_to_first_release(nowcast_fd: NowcastData):
    _, ax_all = nowcast_fd.plot_hedgehog(
        variable=NOWCAST_VARIABLE,
        forecast_source=NOWCAST_SOURCE,
        metric=NOWCAST_METRIC,
        return_plot=True,
    )
    _, ax_first = nowcast_fd.plot_hedgehog(
        variable=NOWCAST_VARIABLE,
        forecast_source=NOWCAST_SOURCE,
        metric=NOWCAST_METRIC,
        releases=[1],
        return_plot=True,
    )
    assert _line_vintage_count(ax_first) < _line_vintage_count(ax_all)

    df = nowcast_fd._forecasts
    df = df[
        (df["variable"] == NOWCAST_VARIABLE)
        & (df["unique_id"] == NOWCAST_SOURCE)
        & (df["metric"] == NOWCAST_METRIC)
        & (df["forecast_horizon"] >= 0)
    ]
    expected_first_release_vintages = df.groupby("date")["vintage_date"].min().nunique()
    assert _line_vintage_count(ax_first) == expected_first_release_vintages


def test_plot_hedgehog_releases_multiple_values(nowcast_fd: NowcastData):
    _, ax_first = nowcast_fd.plot_hedgehog(
        variable=NOWCAST_VARIABLE,
        forecast_source=NOWCAST_SOURCE,
        metric=NOWCAST_METRIC,
        releases=[1],
        return_plot=True,
    )
    _, ax_multi = nowcast_fd.plot_hedgehog(
        variable=NOWCAST_VARIABLE,
        forecast_source=NOWCAST_SOURCE,
        metric=NOWCAST_METRIC,
        releases=[1, 3],
        return_plot=True,
    )
    assert _line_vintage_count(ax_multi) > _line_vintage_count(ax_first)

    df = nowcast_fd._forecasts
    df = df[
        (df["variable"] == NOWCAST_VARIABLE)
        & (df["unique_id"] == NOWCAST_SOURCE)
        & (df["metric"] == NOWCAST_METRIC)
        & (df["forecast_horizon"] >= 0)
    ].copy()
    ranks = df.groupby("date")["vintage_date"].rank(method="dense")
    expected_vintages = df.loc[ranks.isin([1, 3]), "vintage_date"].nunique()
    assert _line_vintage_count(ax_multi) == expected_vintages


def test_plot_hedgehog_releases_none_matches_default_behaviour(nowcast_fd: NowcastData):
    _, ax_default = nowcast_fd.plot_hedgehog(
        variable=NOWCAST_VARIABLE,
        forecast_source=NOWCAST_SOURCE,
        metric=NOWCAST_METRIC,
        return_plot=True,
    )
    _, ax_explicit_none = nowcast_fd.plot_hedgehog(
        variable=NOWCAST_VARIABLE,
        forecast_source=NOWCAST_SOURCE,
        metric=NOWCAST_METRIC,
        releases=None,
        return_plot=True,
    )
    assert _line_vintage_count(ax_default) == _line_vintage_count(ax_explicit_none)

    default_data = [line.get_xydata().tolist() for line in ax_default.get_lines()]
    explicit_none_data = [line.get_xydata().tolist() for line in ax_explicit_none.get_lines()]
    assert default_data == explicit_none_data


def test_plot_hedgehog_releases_raises_for_non_intra_period_data(fer_minimal_fd: ForecastData):
    with pytest.raises(ValueError):
        fer_minimal_fd.plot_hedgehog(
            variable=VARIABLE,
            forecast_source=SOURCE,
            metric=METRIC,
            releases=[1],
            return_plot=True,
        )


def test_plot_hedgehog_releases_empty_selection_raises(nowcast_fd: NowcastData):
    with pytest.raises(ValueError, match="No forecast data found"):
        nowcast_fd.plot_hedgehog(
            variable=NOWCAST_VARIABLE,
            forecast_source=NOWCAST_SOURCE,
            metric=NOWCAST_METRIC,
            releases=[999],
            return_plot=True,
        )
