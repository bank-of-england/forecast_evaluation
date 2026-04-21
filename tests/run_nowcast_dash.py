"""Run dashboard on sample nowcast data."""

from forecast_evaluation import ForecastData
from forecast_evaluation.data.sample_data import (
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
)

fd = ForecastData(
    outturns_data=create_sample_nowcast_outturns(),
    forecasts_data=create_sample_nowcast_forecasts(),
    nowcasting=True,
    first_forecast_horizon=-1,
)

fd.run_dashboard()
