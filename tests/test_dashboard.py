from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from shiny import ui

from forecast_evaluation.dashboard.ui import create_sidebar
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData
from forecast_evaluation.data.sample_data import (
    create_sample_forecasts,
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
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


def test_create_sidebar_includes_releases_input_for_nowcast_data():
    fd = NowcastData(outturns_data=create_sample_nowcast_outturns())
    fd.add_forecasts(create_sample_nowcast_forecasts(), data_check=False)

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
