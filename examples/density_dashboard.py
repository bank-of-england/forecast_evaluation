"""Run the dashboard on density forecasts merged with FER point forecasts.

run_dashboard() blocks until interrupted.
"""

import forecast_evaluation as fe
from forecast_evaluation.data.sample_data import create_sample_density_forecasts

if __name__ == "__main__":
    forecast_data = fe.ForecastData(load_fer=True)

    outturns_df = forecast_data.outturns
    outturns_df = outturns_df[outturns_df["metric"] == "levels"]

    density_data = fe.DensityForecastData(outturns_data=outturns_df, forecasts_data=create_sample_density_forecasts())
    density_data.merge(forecast_data)
    density_data.run_dashboard()
