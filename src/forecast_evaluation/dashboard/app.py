# Run dashboard with default data for Shiny CLI
import forecast_evaluation as fe
from forecast_evaluation.dashboard.create_app import dashboard_app

# Load data once
data = fe.ForecastData(load_fer=True)
app = dashboard_app(data)
