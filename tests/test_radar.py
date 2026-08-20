import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before any figure is created

import pytest

from forecast_evaluation.visualisations.radar import plot_radar

# ``nowcast_fd`` comes from tests/conftest.py.


class TestPlotRadarRejectsUnsupportedNowcastAnalyses:
    """`plot_radar()` should raise early, mode-specific errors for NowcastData."""

    def test_plot_radar_tests_mode_rejects_nowcast_data(self, nowcast_fd):
        with pytest.raises(ValueError, match="mode='tests'"):
            plot_radar(
                nowcast_fd,
                mode="tests",
                variable="gdp",
                metric="levels",
                horizon=0,
                k=4,
                return_plot=True,
            )

    def test_plot_radar_variables_mode_efficiency_rejects_nowcast_data(self, nowcast_fd):
        with pytest.raises(ValueError, match="test_type='efficiency'"):
            plot_radar(
                nowcast_fd,
                mode="variables",
                metric="levels",
                horizon=0,
                test_type="efficiency",
                k=4,
                return_plot=True,
            )

    def test_plot_radar_variables_mode_bias_mz_rejects_nowcast_data(self, nowcast_fd):
        with pytest.raises(ValueError, match="bias_type='mz'"):
            plot_radar(
                nowcast_fd,
                mode="variables",
                metric="levels",
                horizon=0,
                test_type="bias",
                bias_type="mz",
                k=4,
                return_plot=True,
            )


class TestPlotRadarStillWorksForSupportedNowcastAnalyses:
    """Regression checks: the guard must not over-block supported combinations."""

    def test_plot_radar_variables_mode_accuracy_still_works_for_nowcast_data(self, nowcast_fd):
        fig, ax = plot_radar(
            nowcast_fd,
            mode="variables",
            metric="levels",
            horizon=0,
            test_type="accuracy",
            k=4,
            return_plot=True,
        )
        assert fig is not None
        assert ax is not None

    def test_plot_radar_variables_mode_correlation_still_works_for_nowcast_data(self, nowcast_fd):
        fig, ax = plot_radar(
            nowcast_fd,
            mode="variables",
            metric="levels",
            horizon=0,
            test_type="correlation",
            anchor_source="nowcast_dfm",
            k=4,
            return_plot=True,
        )
        assert fig is not None
        assert ax is not None
