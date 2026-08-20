from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from shiny import reactive, ui

from forecast_evaluation.dashboard.create_app import dashboard_app
from forecast_evaluation.dashboard.ui import create_sidebar
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData
from forecast_evaluation.data.sample_data import (
    create_sample_forecasts,
    create_sample_outturns,
)


@pytest.fixture
def sample_outturns() -> pd.DataFrame:
    return create_sample_outturns()


@pytest.fixture
def sample_forecasts() -> pd.DataFrame:
    return create_sample_forecasts()


def test_run_dashboard_non_jupyter(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    mock_app = MagicMock()
    with patch("forecast_evaluation.dashboard.create_app.dashboard_app", return_value=mock_app):
        fd.run_dashboard(from_jupyter=False)
        mock_app.run.assert_called_once_with(host="127.0.0.1", port=8000)


def test_run_dashboard_jupyter(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)
    mock_app = MagicMock()
    with (
        patch("forecast_evaluation.dashboard.create_app.dashboard_app", return_value=mock_app),
        patch("uvicorn.run") as mock_uvicorn,
        patch("IPython.display.IFrame"),
        patch("IPython.display.display"),
    ):
        fd.run_dashboard(from_jupyter=True)
        mock_uvicorn.assert_called_once()


def test_create_sidebar_includes_releases_input_for_nowcast_data(nowcast_fd: NowcastData):
    fd = nowcast_fd

    sidebar = create_sidebar(fd)
    sidebar_html = str(ui.page_fluid(ui.layout_sidebar(sidebar, ui.div())))

    assert 'id="releases"' in sidebar_html
    # The releases control must only be shown on the Hedgehog tab.
    assert 'data-display-if="input.tabs == &apos;Hedgehog&apos;"' in sidebar_html
    df = fd._forecasts
    max_rank = int(df.groupby("date")["vintage_date"].rank(method="dense").max())
    for rank in range(1, max_rank + 1):
        assert f'value="{rank}"' in sidebar_html


def test_create_sidebar_releases_hidden_default_for_forecast_data(sample_outturns, sample_forecasts):
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)

    sidebar = create_sidebar(fd)
    sidebar_html = str(ui.page_fluid(ui.layout_sidebar(sidebar, ui.div())))

    assert 'id="releases"' in sidebar_html
    # Non-nowcast data should not show a real releases selector, only a hidden default.
    assert "display: none" in sidebar_html
    assert 'data-display-if="input.tabs == &apos;Hedgehog&apos;"' not in sidebar_html


def test_dashboard_hides_correlation_and_radar_tabs_for_nowcast_data(nowcast_fd: NowcastData):
    """Correlation/Radar tabs and their handlers are unsupported for nowcast data, so both are skipped."""
    fd = nowcast_fd

    app = dashboard_app(fd)
    page_html = str(app.ui(None))

    assert 'data-value="Correlation"' not in page_html
    assert 'data-value="Radar"' not in page_html

    with (
        patch("forecast_evaluation.dashboard.create_app.correlation_heatmap") as mock_correlation_heatmap,
        patch("forecast_evaluation.dashboard.create_app.rolling_correlation") as mock_rolling_correlation,
        patch("forecast_evaluation.dashboard.create_app.radar") as mock_radar,
    ):
        app.server(MagicMock(), MagicMock(), MagicMock())

    mock_correlation_heatmap.assert_not_called()
    mock_rolling_correlation.assert_not_called()
    mock_radar.assert_not_called()


def test_dashboard_shows_correlation_and_radar_tabs_for_forecast_data(sample_outturns, sample_forecasts):
    """Regression check: the nowcast guard must not over-hide the tabs/handlers for plain forecast data."""
    fd = ForecastData(outturns_data=sample_outturns, forecasts_data=sample_forecasts)

    app = dashboard_app(fd)
    page_html = str(app.ui(None))

    assert 'data-value="Correlation"' in page_html
    assert 'data-value="Radar"' in page_html

    with (
        patch("forecast_evaluation.dashboard.create_app.correlation_heatmap") as mock_correlation_heatmap,
        patch("forecast_evaluation.dashboard.create_app.rolling_correlation") as mock_rolling_correlation,
        patch("forecast_evaluation.dashboard.create_app.radar") as mock_radar,
    ):
        app.server(MagicMock(), MagicMock(), MagicMock())

    mock_correlation_heatmap.assert_called_once()
    mock_rolling_correlation.assert_called_once()
    mock_radar.assert_called_once()


# -----------------------
# Intra-period download export content
# -----------------------
# `download_intra_accuracy`/`download_intra_bias` in dashboard/tabs/intra_period.py
# are local functions wrapped by `@render.download`; they aren't exposed on the app
# object, so `render.download` is patched with a fake decorator that captures the
# undecorated function instead of wrapping it in a `Renderer`. The captured function
# still calls the real `get_data()` (a `@reactive.calc`/`@reactive.event` closure),
# so it's invoked inside `reactive.isolate()` with a `MagicMock` `input` standing in
# for Shiny's reactive inputs, exercising the exact download callback body rather
# than just the `fe.compute_intra_period_*` helper it delegates to.
def _capture_download_fn(module, register_fn, input, data):
    captured: dict = {}

    def fake_download(*, filename=None, **kwargs):
        def decorator(fn):
            captured["fn"] = fn
            return fn

        return decorator

    with patch(f"forecast_evaluation.dashboard.tabs.{module}.render.download", fake_download):
        register_fn(input, MagicMock(), MagicMock(), data)

    return captured["fn"]


def _make_intra_period_input(**overrides):
    input = MagicMock()
    input.update = MagicMock(return_value=0)
    input.sources = MagicMock(return_value=["nowcast_dfm", "nowcast_bridge"])
    input.start_date = MagicMock(return_value=None)
    input.end_date = MagicMock(return_value=None)
    input.start_vintage = MagicMock(return_value=None)
    input.end_vintage = MagicMock(return_value=None)
    input.variable = MagicMock(return_value="gdp")
    input.transform = MagicMock(return_value="levels")
    input.covid_filter = MagicMock(return_value="No")
    input.intra_axis = MagicMock(return_value="publication")
    input.__getitem__ = MagicMock(return_value=MagicMock(return_value=[]))
    for name, value in overrides.items():
        setattr(input, name, MagicMock(return_value=value))
    return input


def test_intra_period_accuracy_download_exports_grouped_result(nowcast_fd: NowcastData):
    from forecast_evaluation.dashboard.tabs.intra_period import intra_period_accuracy

    fd = nowcast_fd
    raw_main_table = fd.df

    input = _make_intra_period_input(intra_statistic="rmse")
    download_fn = _capture_download_fn("intra_period", intra_period_accuracy, input, fd)

    with reactive.isolate():
        result = download_fn()
    exported = pd.read_csv(result, index_col=0)

    assert list(exported.columns) != list(raw_main_table.columns)
    assert len(exported) < len(raw_main_table)
    assert "value" in exported.columns
    assert "forecast_error" not in exported.columns


def test_intra_period_bias_download_exports_grouped_result(nowcast_fd: NowcastData):
    from forecast_evaluation.dashboard.tabs.intra_period import intra_period_bias

    fd = nowcast_fd
    raw_main_table = fd.df

    input = _make_intra_period_input()
    download_fn = _capture_download_fn("intra_period", intra_period_bias, input, fd)

    with reactive.isolate():
        result = download_fn()
    exported = pd.read_csv(result, index_col=0)

    assert list(exported.columns) != list(raw_main_table.columns)
    assert len(exported) < len(raw_main_table)
    assert "value" in exported.columns
    assert "forecast_error" not in exported.columns
