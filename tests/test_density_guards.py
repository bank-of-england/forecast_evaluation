"""Point-forecast analyses and plots give a clear error on objects without point forecasts."""

import matplotlib
import pytest

from forecast_evaluation.data import DensityForecastData, ForecastData
from forecast_evaluation.tests.accuracy import compute_accuracy_statistics
from forecast_evaluation.tests.bias import bias_analysis
from forecast_evaluation.tests.blanchard_leigh import blanchard_leigh_horizon_analysis
from forecast_evaluation.tests.correlation import forecast_errors_correlation_analysis
from forecast_evaluation.tests.diebold_mariano import diebold_mariano_table
from forecast_evaluation.tests.revisions_errors_correlation import revisions_errors_correlation_analysis
from forecast_evaluation.tests.revisions_predictability import revision_predictability_analysis
from forecast_evaluation.tests.rolling_analysis import rolling_analysis
from forecast_evaluation.tests.strong_efficiency import strong_efficiency_analysis
from forecast_evaluation.tests.weak_efficiency import weak_efficiency_analysis
from forecast_evaluation.visualisations.errors import plot_errors_across_time
from forecast_evaluation.visualisations.forecast import plot_nowcasts, plot_vintage
from forecast_evaluation.visualisations.forecast_errors import (
    plot_forecast_error_density,
    plot_forecast_errors,
    plot_forecast_errors_by_horizon,
)
from forecast_evaluation.visualisations.hedgehog import plot_hedgehog
from forecast_evaluation.visualisations.radar import plot_radar
from forecast_evaluation.visualisations.revisions_predictability import plot_average_revision_by_period

matplotlib.use("Agg")

_ARGS = {"source": "mpr2", "outcome_variable": "gdpkp", "instrument_variable": "gdpkp"}
_CALLS = {
    "compute_accuracy_statistics": lambda d: compute_accuracy_statistics(d),
    "bias_analysis": lambda d: bias_analysis(d),
    "diebold_mariano_table": lambda d: diebold_mariano_table(d, "mpr2"),
    "blanchard_leigh_horizon_analysis": lambda d: blanchard_leigh_horizon_analysis(
        d, outcome_metric="levels", instrument_metric="levels", **_ARGS
    ),
    "strong_efficiency_analysis": lambda d: strong_efficiency_analysis(
        d, outcome_metric="levels", instrument_metric="levels", **_ARGS
    ),
    "weak_efficiency_analysis": lambda d: weak_efficiency_analysis(d),
    "forecast_errors_correlation_analysis": lambda d: forecast_errors_correlation_analysis(d),
    "revisions_errors_correlation_analysis": lambda d: revisions_errors_correlation_analysis(d),
    "revision_predictability_analysis": lambda d: revision_predictability_analysis(d),
    "rolling_analysis": lambda d: rolling_analysis(d, 4, compute_accuracy_statistics, {}),
    "plot_forecast_errors": lambda d: plot_forecast_errors(d, "gdpkp", "levels", "mpr2", "2025-12-31"),
    "plot_forecast_errors_by_horizon": lambda d: plot_forecast_errors_by_horizon(d, "gdpkp", "mpr2", "levels"),
    "plot_forecast_error_density": lambda d: plot_forecast_error_density(d, "gdpkp", 1, "levels", "mpr2"),
    "plot_vintage": lambda d: plot_vintage(d, "gdpkp", "2025-12-31"),
    "plot_nowcasts": lambda d: plot_nowcasts(d, "gdpkp", "2025-12-31"),
    "plot_hedgehog": lambda d: plot_hedgehog(d, "gdpkp", "mpr2", "levels"),
    "plot_errors_across_time": lambda d: plot_errors_across_time(d, "gdpkp", "levels"),
    "plot_average_revision_by_period": lambda d: plot_average_revision_by_period(d, "mpr2", "gdpkp", "levels"),
    "plot_radar": lambda d: plot_radar(d, "variables", metric="levels", horizon=1),
}


@pytest.mark.parametrize("name", list(_CALLS))
def test_point_forecast_functions_point_density_only_users_to_to_point_forecast(density_only, name):
    with pytest.raises(ValueError, match=r"No point forecasts loaded.*to_point_forecast\(\)"):
        _CALLS[name](density_only)


def test_point_forecast_functions_still_work_with_point_forecasts(density_and_point):
    result = compute_accuracy_statistics(density_and_point)
    assert not result.to_df().empty

    plot_vintage(density_and_point, "gdpkp", "2025-12-31")


def test_point_forecast_function_on_object_without_any_forecasts_says_to_add_them(density_outturns_df):
    objects = (
        ForecastData(outturns_data=density_outturns_df),
        DensityForecastData(outturns_data=density_outturns_df),
    )
    for data in objects:
        with pytest.raises(ValueError, match=r"No point forecasts loaded.*add_forecasts\(\)"):
            compute_accuracy_statistics(data)


def test_require_point_forecasts_passes_with_point_forecasts(density_and_point):
    density_and_point._require_point_forecasts()


def test_require_point_forecasts_ignores_filters(density_and_point):
    density_and_point.filter(variables=["cpisa"])

    density_and_point._require_point_forecasts()


def test_filter_fer_on_density_only_object_is_a_noop(density_only):
    n_density = len(density_only.density_forecasts)

    density_only.filter_fer()

    assert len(density_only.density_forecasts) == n_density
