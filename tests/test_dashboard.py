from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.sample_data import create_sample_forecasts, create_sample_outturns


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
