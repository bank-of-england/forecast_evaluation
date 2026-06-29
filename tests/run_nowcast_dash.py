"""Run dashboard on sample nowcast data."""

from forecast_evaluation import NowcastData
from forecast_evaluation.data.sample_data import (
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
)

fd = NowcastData(
    outturns_data=create_sample_nowcast_outturns(),
    forecasts_data=create_sample_nowcast_forecasts(),
)

fd.run_dashboard()
