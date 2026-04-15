"""Plot intra-period accuracy for sample nowcasting data."""

from forecast_evaluation import (
    ForecastData,
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
    plot_intra_period_accuracy,
)

# --- Build ForecastData with nowcasting sample data ---
outturns = create_sample_nowcast_outturns()
forecasts = create_sample_nowcast_forecasts()

fd = ForecastData(outturns_data=outturns, nowcasting=True)
fd.add_forecasts(forecasts, data_check=False)

# --- Plot RMSE by day in quarter (GDP, levels, horizon 0) ---
plot_intra_period_accuracy(
    fd,
    variable="gdp",
    metric="levels",
    frequency="Q",
    forecast_horizon=0,
    statistic="rmse",
)

# --- Plot MAE by day in quarter (CPI, levels, horizon 0) ---
plot_intra_period_accuracy(
    fd,
    variable="cpi",
    metric="levels",
    frequency="Q",
    forecast_horizon=0,
    statistic="mae",
)
