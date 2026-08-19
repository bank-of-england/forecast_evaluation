"""Run dashboard on sample FER data."""

import forecast_evaluation as fe

fd = fe.ForecastData(load_fer=True)

fd.run_dashboard()
