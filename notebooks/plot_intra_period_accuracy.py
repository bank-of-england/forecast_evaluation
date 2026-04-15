"""Plot intra-period accuracy for sample nowcasting data."""

from forecast_evaluation import ForecastData, plot_intra_period_accuracy, plot_intra_period_bias
from forecast_evaluation.data.sample_data import (
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
)

fd = ForecastData(outturns_data=create_sample_nowcast_outturns(), nowcasting=True)
fd.add_forecasts(create_sample_nowcast_forecasts(), data_check=False)

fd.run_dashboard()
plot_intra_period_accuracy(fd, variable="gdp", metric="levels", frequency="Q", forecast_horizon=0, statistic="rmse")
plot_intra_period_accuracy(fd, variable="gdp", metric="levels", frequency="Q", forecast_horizon=0, statistic="mae")
plot_intra_period_accuracy(fd, variable="cpi", metric="levels", frequency="Q", forecast_horizon=0, statistic="rmse")

plot_intra_period_bias(fd, variable="gdp", metric="levels", frequency="Q", forecast_horizon=0)
plot_intra_period_bias(fd, variable="cpi", metric="levels", frequency="Q", forecast_horizon=0)
