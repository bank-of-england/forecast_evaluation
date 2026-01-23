# forecast_evaluation/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("forecast_evaluation")
except PackageNotFoundError:
    __version__ = "unknown"

from .core import add_ar_p_forecasts, add_random_walk_forecasts, create_outturn_revisions
from .data import DensityForecastData, ForecastData, create_sample_forecasts, create_sample_outturns
from .data.utils import filter_fer_variables
from .tests import (
    bias_analysis,
    blanchard_leigh_horizon_analysis,
    compare_to_benchmark,
    compute_accuracy_statistics,
    create_comparison_table,
    diebold_mariano_table,
    fluctuation_tests,
    revision_predictability_analysis,
    revisions_errors_correlation_analysis,
    rolling_analysis,
    strong_efficiency_analysis,
    weak_efficiency_analysis,
)
from .utils import covid_filter, reconstruct_id_cols_from_unique_id
from .visualisations import (
    plot_accuracy,
    plot_average_revision_by_period,
    plot_bias_by_horizon,
    plot_blanchard_leigh_ratios,
    plot_compare_to_benchmark,
    plot_errors_across_time,
    plot_forecast_error_density,
    plot_forecast_errors,
    plot_forecast_errors_by_horizon,
    plot_hedgehog,
    plot_outturn_revisions,
    plot_outturns,
    plot_rolling_bias,
    plot_rolling_relative_accuracy,
    plot_strong_efficiency,
    plot_vintage,
)

__all__ = [
    # Core functions
    "create_outturn_revisions",
    "add_random_walk_forecasts",
    "add_ar_p_forecasts",
    # Data classes
    "ForecastData",
    "DensityForecastData",
    # Sample data functions
    "create_sample_forecasts",
    "create_sample_outturns",
    # Test/analysis functions
    "bias_analysis",
    "blanchard_leigh_horizon_analysis",
    "compare_to_benchmark",
    "compute_accuracy_statistics",
    "create_comparison_table",
    "diebold_mariano_table",
    "fluctuation_tests",
    "revisions_errors_correlation_analysis",
    "revision_predictability_analysis",
    "rolling_analysis",
    "strong_efficiency_analysis",
    "weak_efficiency_analysis",
    # Visualisation functions
    "plot_accuracy",
    "plot_average_revision_by_period",
    "plot_blanchard_leigh_ratios",
    "plot_compare_to_benchmark",
    "plot_strong_efficiency",
    "plot_forecast_error_density",
    "plot_forecast_errors",
    "plot_forecast_errors_by_horizon",
    "plot_hedgehog",
    "plot_outturn_revisions",
    "plot_outturns",
    "plot_rolling_bias",
    "plot_rolling_relative_accuracy",
    "plot_bias_by_horizon",
    "plot_vintage",
    "plot_errors_across_time",
    # Utility functions
    "covid_filter",
    "filter_fer_variables",
    "reconstruct_id_cols_from_unique_id",
]
