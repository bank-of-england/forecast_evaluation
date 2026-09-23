import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.tests.accuracy import compute_accuracy_statistics
from forecast_evaluation.tests.blanchard_leigh import blanchard_leigh_horizon_analysis
from forecast_evaluation.tests.strong_efficiency import strong_efficiency_analysis
from forecast_evaluation.visualisations.accuracy import plot_accuracy, plot_compare_to_benchmark


def _mixed_frequency_data() -> ForecastData:
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-06-30", "2022-04-30"]),
            "vintage_date": pd.to_datetime(["2022-09-30", "2022-06-30"]),
            "variable": ["quarterly_outcome", "monthly_instrument"],
            "frequency": ["Q", "M"],
            "value": [10.0, 20.0],
        }
    )
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-06-30", "2022-04-30"]),
            "vintage_date": pd.to_datetime(["2022-03-31", "2022-03-31"]),
            "variable": ["quarterly_outcome", "monthly_instrument"],
            "source": ["model", "model"],
            "frequency": ["Q", "M"],
            "forecast_horizon": [0, 0],
            "value": [9.0, 18.0],
        }
    )
    return ForecastData(outturns_data=outturns, forecasts_data=forecasts, compute_levels=False, data_check=False)


def test_strong_efficiency_requires_a_common_frequency():
    data = _mixed_frequency_data()

    with pytest.raises(ValueError, match="same frequency"):
        strong_efficiency_analysis(
            data,
            source="model",
            outcome_variable="quarterly_outcome",
            outcome_metric="levels",
            instrument_variable="monthly_instrument",
            instrument_metric="levels",
            horizons=np.array([0]),
            j=0,
        )


def test_blanchard_leigh_requires_a_common_frequency():
    data = _mixed_frequency_data()

    with pytest.raises(ValueError, match="same frequency"):
        blanchard_leigh_horizon_analysis(
            data,
            source="model",
            outcome_variable="quarterly_outcome",
            outcome_metric="levels",
            instrument_variable="monthly_instrument",
            instrument_metric="levels",
            horizons=np.array([0]),
            j=0,
        )


@pytest.mark.parametrize("analysis", [strong_efficiency_analysis, blanchard_leigh_horizon_analysis])
def test_joint_analysis_requires_both_variables(analysis):
    data = _mixed_frequency_data()

    with pytest.raises(ValueError, match="missing \['missing_instrument'\]"):
        analysis(
            data,
            source="model",
            outcome_variable="quarterly_outcome",
            outcome_metric="levels",
            instrument_variable="missing_instrument",
            instrument_metric="levels",
            horizons=np.array([0]),
            j=0,
        )


def test_random_walk_benchmark_builds_forecasts_for_mixed_frequency_variables():
    quarterly_dates = pd.date_range("2017-03-31", periods=24, freq="QE")
    monthly_dates = pd.date_range("2017-01-31", periods=72, freq="ME")
    vintage_date = pd.Timestamp("2023-03-31")
    outturns = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": quarterly_dates,
                    "vintage_date": vintage_date,
                    "variable": "quarterly",
                    "frequency": "Q",
                    "metric": "levels",
                    "value": np.arange(len(quarterly_dates), dtype=float) + 100,
                }
            ),
            pd.DataFrame(
                {
                    "date": monthly_dates,
                    "vintage_date": vintage_date,
                    "variable": "monthly",
                    "frequency": "M",
                    "metric": "levels",
                    "value": np.arange(len(monthly_dates), dtype=float) + 10,
                }
            ),
        ],
        ignore_index=True,
    )
    data = ForecastData(outturns_data=outturns, compute_levels=False)

    data.add_benchmarks(models="random_walk", forecast_periods=2)

    forecasts = data._raw_forecasts
    assert set(forecasts["frequency"]) == {"Q", "M"}
    assert forecasts.groupby("variable")["frequency"].nunique().eq(1).all()
    assert forecasts.groupby("variable").size().to_dict() == {"monthly": 2, "quarterly": 2}


def test_single_variable_accuracy_matches_single_frequency_instance():
    mixed = _mixed_frequency_data()
    variable = "monthly_instrument"
    single = ForecastData(
        outturns_data=mixed._raw_outturns[mixed._raw_outturns["variable"] == variable],
        forecasts_data=mixed._raw_forecasts[mixed._raw_forecasts["variable"] == variable],
        compute_levels=False,
        data_check=False,
    )

    mixed_accuracy = compute_accuracy_statistics(mixed, variable=variable, k=1).to_df()
    single_accuracy = compute_accuracy_statistics(single, variable=variable, k=1).to_df()
    result_columns = ["variable", "metric", "frequency", "horizon", "rmse", "mean_abs_error", "n_observations"]

    pd.testing.assert_frame_equal(
        mixed_accuracy[result_columns].sort_values(result_columns[:4]).reset_index(drop=True),
        single_accuracy[result_columns].sort_values(result_columns[:4]).reset_index(drop=True),
    )


def test_accuracy_plots_format_monthly_date_ranges():
    accuracy = pd.DataFrame(
        {
            "variable": ["monthly"] * 2,
            "unique_id": ["model", "benchmark"],
            "metric": ["levels"] * 2,
            "frequency": ["M"] * 2,
            "horizon": [0] * 2,
            "rmse": [1.0, 2.0],
            "rmedse": [1.0, 2.0],
            "mean_abs_error": [1.0, 2.0],
            "n_observations": [3, 3],
            "start_date": pd.to_datetime(["2022-01-31"] * 2),
            "end_date": pd.to_datetime(["2022-03-31"] * 2),
        }
    )

    accuracy_figure, accuracy_axis = plot_accuracy(accuracy, "monthly", "levels", return_plot=True)
    benchmark_figure, benchmark_axis = plot_compare_to_benchmark(
        accuracy,
        "monthly",
        "levels",
        benchmark_model="benchmark",
        return_plot=True,
    )

    assert "2022-01 to 2022-03" in accuracy_axis.get_title()
    assert "2022-01 to 2022-03" in benchmark_axis.get_title()
    plt.close(accuracy_figure)
    plt.close(benchmark_figure)
