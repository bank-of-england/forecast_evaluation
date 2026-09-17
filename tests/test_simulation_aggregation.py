import numpy as np
import pandas as pd
import pytest

from forecast_evaluation import SimulationData, compute_accuracy_statistics
from forecast_evaluation.tests.bias import bias_analysis
from forecast_evaluation.tests.results import TestResult as ForecastTestResult
from forecast_evaluation.tests.weak_efficiency import weak_efficiency_analysis

SIMULATION_PATHS = [(1, "baseline"), (2, "baseline"), (1, "shock"), (2, "shock")]
GENERATED_SUMMARY_NAMES = [
    "n_paths",
    "n_draws",
    "estimate_mean",
    "estimate_se",
    "estimate_sd",
    "estimate_p05",
    "estimate_median",
    "estimate_p95",
    "estimate_n",
    "estimate_n_missing",
]
COVERAGE_SUMMARY_NAMES = ["n_observations_total", "n_observations_min", "n_observations_max"]


def _simulation() -> SimulationData:
    return SimulationData(simulation_ids=("draw", "scenario"))


def _result_frame(values: list[float], *, dimensions: dict | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "draw": [path[0] for path in SIMULATION_PATHS],
            "scenario": [path[1] for path in SIMULATION_PATHS],
            "estimate": values,
        }
    )
    if dimensions:
        for column, value in dimensions.items():
            frame[column] = value
    return frame


def _accuracy_result_frame(
    *,
    values: list[float],
    rmse: list[float] | None = None,
    n_observations: list[int] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "draw": [path[0] for path in SIMULATION_PATHS],
            "scenario": [path[1] for path in SIMULATION_PATHS],
            "unique_id": "model",
            "source": "model",
            "variable": "y",
            "metric": "levels",
            "frequency": "Q",
            "horizon": 0,
            "mse": values,
            "rmse": values if rmse is None else rmse,
            "mean_abs_error": values,
            "rmedse": values,
            "n_observations": 1 if n_observations is None else n_observations,
            "start_date": pd.Timestamp("2020-01-01"),
            "end_date": pd.Timestamp("2020-01-01"),
        }
    )


def _long_simulation() -> SimulationData:
    dates = pd.date_range("2020-03-31", periods=10, freq="QE")
    errors = np.arange(1, 11, dtype=float)
    outturns = []
    forecasts = []
    for draw, scenario in SIMULATION_PATHS:
        path_offset = (draw - 1) * 100 + (50 if scenario == "shock" else 0)
        for index, date in enumerate(dates):
            outturn = 100 + path_offset + 2 * index
            outturns.append(
                {
                    "draw": draw,
                    "scenario": scenario,
                    "date": date,
                    "variable": "y",
                    "frequency": "Q",
                    "metric": "levels",
                    "value": outturn,
                }
            )
            for source, bias in (("model", 0.0), ("benchmark", -2.0)):
                forecast_date = date
                forecasts.append(
                    {
                        "draw": draw,
                        "scenario": scenario,
                        "date": date,
                        "vintage_date": forecast_date,
                        "variable": "y",
                        "frequency": "Q",
                        "metric": "levels",
                        "source": source,
                        "forecast_horizon": 0,
                        "value": outturn - errors[index] - bias,
                    }
                )
    simulation = SimulationData(
        outturns_data=pd.DataFrame(outturns),
        forecasts_data=pd.DataFrame(forecasts),
        outturn_vintages=False,
        compute_levels=False,
        data_check=False,
    )
    simulation.filter(metrics=["levels"])
    return simulation


def test_accuracy_aggregation_matches_hand_calculated_path_statistics():
    result = _accuracy_result_frame(values=[1.0, 4.0, 9.0, 16.0], rmse=[1.0, 2.0, 3.0, 4.0])

    summary = _simulation().aggregate_accuracy(result, across=["draw"])

    baseline = summary.loc[summary["scenario"].eq("baseline")].iloc[0]
    shock = summary.loc[summary["scenario"].eq("shock")].iloc[0]
    assert baseline["mse_mean"] == 2.5
    assert baseline["rmse_mean"] == 1.5
    assert baseline["mean_abs_error_mean"] == 2.5
    assert baseline["rmedse_mean"] == 2.5
    assert baseline["n_paths"] == 2
    assert baseline["n_draws"] == 2
    assert shock["mse_mean"] == 12.5


def test_accuracy_evaluation_uses_forecast_errors_and_bias_has_path_coefficients():
    simulation = _long_simulation()

    accuracy = simulation.evaluate_accuracy(source="model", k=0)
    accuracy_frame = accuracy.to_df()
    assert set(accuracy_frame[["draw", "scenario"]].itertuples(index=False, name=None)) == set(SIMULATION_PATHS)
    assert accuracy_frame["n_observations"].eq(10).all()
    assert np.allclose(accuracy_frame["mean_abs_error"], 5.5)
    assert np.allclose(accuracy_frame["mse"], np.mean(np.arange(1, 11, dtype=float) ** 2))
    assert np.allclose(accuracy_frame["rmse"], np.sqrt(np.mean(np.arange(1, 11, dtype=float) ** 2)))

    bias = simulation.evaluate(bias_analysis, source="model", variable="y", k=0, verbose=False)
    bias_frame = bias.to_df()
    assert len(bias_frame) == len(SIMULATION_PATHS)
    assert np.allclose(bias_frame["bias_estimate"], 5.5)

    summary = simulation.aggregate_results(bias, across=["draw"])
    assert np.allclose(summary["bias_estimate_mean"], 5.5)
    assert "std_error_mean" not in summary.columns
    assert "p_value_mean" not in summary.columns


def test_bias_filter_selects_requested_source():
    bias = _long_simulation().evaluate(bias_analysis, source="benchmark", variable="y", k=0, verbose=False)

    result = bias.to_df()
    assert len(result) == len(SIMULATION_PATHS)
    assert result["source"].eq("benchmark").all()
    assert np.allclose(result["bias_estimate"], 3.5)


def test_existing_regression_analysis_can_be_evaluated_per_path():
    result = _long_simulation().evaluate(weak_efficiency_analysis, source="model", variable="y", k=0, verbose=False)

    assert isinstance(result, ForecastTestResult)
    assert len(result) == len(SIMULATION_PATHS)
    assert result.to_df()["n_observations"].eq(10).all()


def test_explicit_aggregation_preserves_term_identity():
    result = pd.DataFrame(
        {
            "draw": [1, 2, 1, 2],
            "scenario": ["baseline"] * 4,
            "term": ["alpha", "alpha", "beta", "beta"],
            "estimate": [1.0, 3.0, 10.0, 14.0],
        }
    )

    summary = _simulation().aggregate_results(
        result,
        across=["draw"],
        statistics=["estimate"],
        group_by=["term"],
    )

    assert summary.set_index("term")["estimate_mean"].to_dict() == {"alpha": 2.0, "beta": 12.0}
    assert set(summary.columns) == {
        "scenario",
        "term",
        "n_paths",
        "n_draws",
        "estimate_mean",
        "estimate_se",
        "estimate_sd",
        "estimate_p05",
        "estimate_median",
        "estimate_p95",
        "estimate_n",
        "estimate_n_missing",
    }


def test_aggregation_uses_equal_path_weight_and_does_not_derive_rmse():
    result = _accuracy_result_frame(
        values=[1.0, 9.0, 1.0, 9.0],
        rmse=[1.0, 3.0, 1.0, 3.0],
        n_observations=[8, 2, 8, 2],
    )

    summary = _simulation().aggregate_accuracy(result, across=["draw"])
    baseline = summary.loc[summary["scenario"].eq("baseline")].iloc[0]
    assert baseline["mse_mean"] == 5.0
    assert baseline["rmse_mean"] == 2.0
    assert baseline["rmse_mean"] != np.sqrt(baseline["mse_mean"])
    assert baseline["n_observations_total"] == 10.0
    assert baseline["n_observations_min"] == 2.0
    assert baseline["n_observations_max"] == 8.0


def test_aggregation_without_simulation_ids_reports_paths_and_no_standard_error():
    result = _accuracy_result_frame(values=[1.0, 2.0, 3.0, 4.0])

    summary = _simulation().aggregate_accuracy(result, across=["draw", "scenario"])

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_paths"] == 4
    assert row["n_draws"] == 2
    assert row["mse_mean"] == 2.5
    assert pd.isna(row["mse_se"])
    assert summary.attrs["se_assumption"] == "descriptive only; independence not established"


@pytest.mark.parametrize(
    ("values", "expected_n", "expected_missing", "expected_mean"),
    [
        ([1.0, np.nan, 3.0, 5.0], 3, 1, 3.0),
        ([np.nan, np.nan, np.nan, np.nan], 0, 4, np.nan),
        ([np.nan, np.nan, 7.0, np.nan], 1, 3, 7.0),
    ],
)
def test_aggregation_counts_missing_zero_and_one_contributors(values, expected_n, expected_missing, expected_mean):
    result = _result_frame(values)

    summary = _simulation().aggregate_results(
        result,
        across=["draw", "scenario"],
        statistics=["estimate"],
        group_by=[],
    )

    row = summary.iloc[0]
    assert row["estimate_n"] == expected_n
    assert row["estimate_n_missing"] == expected_missing
    if np.isnan(expected_mean):
        assert pd.isna(row["estimate_mean"])
    else:
        assert row["estimate_mean"] == expected_mean
    if expected_n < 2:
        assert pd.isna(row["estimate_se"])
        assert pd.isna(row["estimate_sd"])


def test_aggregation_rejects_infinite_statistics():
    result = _result_frame([1.0, np.inf, 3.0, 4.0])

    with pytest.raises(ValueError, match="infinite"):
        _simulation().aggregate_results(result, across=["draw"], statistics=["estimate"], group_by=[])


@pytest.mark.parametrize(
    ("result", "kwargs", "message", "error_type"),
    [
        (object(), {}, "TestResult or DataFrame", TypeError),
        (
            _result_frame([1.0] * 4),
            {"across": "draw", "statistics": ["estimate"], "group_by": []},
            "sequence",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": [], "statistics": ["estimate"], "group_by": []},
            "at least one",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw", "draw"], "statistics": ["estimate"], "group_by": []},
            "unique",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["replicate"], "statistics": ["estimate"], "group_by": []},
            "unknown",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw"], "statistics": ["estimate"]},
            "both",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw"], "group_by": []},
            "both",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw"], "statistics": [], "group_by": []},
            "statistics",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw"], "statistics": ["estimate", "estimate"], "group_by": []},
            "statistics",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw"], "statistics": ["estimate"], "group_by": ["term", "term"]},
            "group_by",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw"], "statistics": "estimate", "group_by": []},
            "sequence",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw"], "statistics": ["estimate"], "group_by": "term"},
            "sequence",
            ValueError,
        ),
        (
            _result_frame([1.0] * 4),
            {"across": ["draw"], "statistics": ["estimate"], "group_by": ["term"]},
            "required",
            ValueError,
        ),
    ],
)
def test_aggregation_rejects_invalid_across_and_contract_arguments(result, kwargs, message, error_type):
    with pytest.raises(error_type, match=message):
        _simulation().aggregate_results(result, **kwargs)


def test_aggregation_rejects_invalid_result_schemas():
    simulation = _simulation()
    valid = _result_frame([1.0] * 4)
    cases = [
        (valid.drop(columns="scenario"), {"statistics": ["estimate"], "group_by": []}, "simulation identifier"),
        (valid.assign(extra=1), {"statistics": ["estimate"], "group_by": []}, "undeclared"),
        (valid, {"statistics": ["missing"], "group_by": []}, "required"),
        (valid, {"statistics": ["estimate"], "group_by": ["term"]}, "required"),
        (valid, {"statistics": ["draw"], "group_by": []}, "overlap"),
        (
            pd.concat([valid, valid.iloc[[0]]], ignore_index=True),
            {"statistics": ["estimate"], "group_by": []},
            "Duplicate result rows",
        ),
        (valid.assign(draw=[1, 2, 3.5, 4]), {"statistics": ["estimate"], "group_by": []}, "draw"),
        (
            valid.assign(scenario=["baseline", "", "baseline", "shock"]),
            {"statistics": ["estimate"], "group_by": []},
            "Invalid simulation identifier",
        ),
    ]
    for result, contract, message in cases:
        with pytest.raises(ValueError, match=message):
            simulation.aggregate_results(result, across=["draw"], **contract)


def test_aggregation_rejects_duplicate_column_names():
    result = pd.concat([_result_frame([1.0] * 4), _result_frame([1.0] * 4)[["estimate"]]], axis=1)
    result.columns = ["draw", "scenario", "estimate", "estimate"]

    with pytest.raises(ValueError, match="column names"):
        _simulation().aggregate_results(result, across=["draw"], statistics=["estimate"], group_by=[])


@pytest.mark.parametrize("collision_name", GENERATED_SUMMARY_NAMES)
@pytest.mark.parametrize("empty", [False, True])
def test_aggregation_rejects_generated_name_collisions_in_simulation_ids(collision_name, empty):
    simulation = SimulationData(simulation_ids=("draw", collision_name))
    result = _result_frame([1.0, 2.0, 3.0, 4.0]).rename(columns={"scenario": collision_name})
    if empty:
        result = result.iloc[:0].copy()
    before = result.copy()

    with pytest.raises(ValueError, match="Grouping columns conflict with generated summary columns"):
        simulation.aggregate_results(result, across=["draw"], statistics=["estimate"], group_by=[])

    pd.testing.assert_frame_equal(result, before)


@pytest.mark.parametrize("collision_name", GENERATED_SUMMARY_NAMES)
@pytest.mark.parametrize("empty", [False, True])
def test_aggregation_rejects_generated_name_collisions_in_group_by(collision_name, empty):
    simulation = _simulation()
    result = _result_frame([1.0, 2.0, 3.0, 4.0], dimensions={collision_name: "group"})
    if empty:
        result = result.iloc[:0].copy()
    before = result.copy()

    with pytest.raises(ValueError, match="Grouping columns conflict with generated summary columns"):
        simulation.aggregate_results(result, across=["draw"], statistics=["estimate"], group_by=[collision_name])

    pd.testing.assert_frame_equal(result, before)


@pytest.mark.parametrize("collision_name", COVERAGE_SUMMARY_NAMES)
def test_automatic_accuracy_aggregation_rejects_coverage_name_collisions(collision_name):
    simulation = SimulationData(simulation_ids=("draw", collision_name))
    result = _accuracy_result_frame(values=[1.0, 2.0, 3.0, 4.0]).rename(columns={"scenario": collision_name})
    before = result.copy()

    with pytest.raises(ValueError, match="Grouping columns conflict with generated summary columns"):
        simulation.aggregate_accuracy(result, across=["draw"])

    pd.testing.assert_frame_equal(result, before)


def test_aggregation_succeeds_when_collision_named_simulation_id_is_removed_by_across():
    simulation = SimulationData(simulation_ids=("draw", "estimate_mean"))
    result = _result_frame([1.0, 2.0, 3.0, 4.0]).rename(columns={"scenario": "estimate_mean"})

    summary = simulation.aggregate_results(
        result,
        across=["draw", "estimate_mean"],
        statistics=["estimate"],
        group_by=[],
    )

    assert summary["estimate_mean"].iloc[0] == 2.5
    assert summary["n_paths"].iloc[0] == 4


def test_automatic_aggregation_requires_a_known_result_contract():
    result = _result_frame([1.0] * 4)

    with pytest.raises(ValueError, match="Unknown automatic result contract"):
        _simulation().aggregate_results(result, across=["draw"])


@pytest.mark.parametrize("result_type", [pd.DataFrame, ForecastTestResult])
def test_empty_aggregation_preserves_typed_schema_and_attributes(result_type):
    simulation = _simulation()
    nonempty = _accuracy_result_frame(values=[1.0, 2.0, 3.0, 4.0])
    populated = simulation.aggregate_accuracy(result_type(nonempty), across=["draw"])
    empty_input = nonempty.iloc[:0].copy()
    empty = simulation.aggregate_accuracy(result_type(empty_input), across=["draw"])

    populated_frame = populated.to_df() if isinstance(populated, ForecastTestResult) else populated
    empty_frame = empty.to_df() if isinstance(empty, ForecastTestResult) else empty
    assert empty_frame.columns.tolist() == populated_frame.columns.tolist()
    assert empty_frame.dtypes.to_dict() == populated_frame.dtypes.to_dict()
    assert empty_frame.attrs == populated_frame.attrs
    if isinstance(empty, ForecastTestResult):
        assert empty._metadata["contract"] == populated._metadata["contract"]


def test_empty_known_evaluation_after_filter_does_not_call_analysis():
    simulation = _long_simulation()
    simulation.filter(custom_filter=lambda frame: frame.iloc[:0])
    called = False

    def should_not_run(_panel):
        nonlocal called
        called = True
        raise AssertionError("analysis should not run for an empty simulation")

    result = simulation.evaluate(compute_accuracy_statistics)
    assert isinstance(result, ForecastTestResult)
    assert result.to_df().empty
    assert not called

    result = simulation.evaluate(should_not_run)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert not called


def test_standard_errors_percentiles_and_coverage_are_hand_calculated():
    frame = _accuracy_result_frame(values=[1.0, 9.0, 4.0, 16.0])
    frame.loc[1, "start_date"] = pd.Timestamp("2019-01-01")
    frame.loc[1, "end_date"] = pd.Timestamp("2021-01-01")
    summary = _simulation().aggregate_accuracy(frame).set_index("scenario")
    baseline = summary.loc["baseline"]
    assert baseline["mse_se"] == pytest.approx(4.0)
    assert baseline["mse_sd"] == pytest.approx(np.sqrt(32.0))
    assert baseline["mse_p05"] == pytest.approx(1.4)
    assert baseline["mse_median"] == 5.0
    assert baseline["mse_p95"] == pytest.approx(8.6)
    assert baseline["start_date"] == pd.Timestamp("2019-01-01")
    assert baseline["end_date"] == pd.Timestamp("2021-01-01")


@pytest.mark.parametrize("analysis", [compute_accuracy_statistics, bias_analysis])
def test_scenario_filtered_evaluation_and_empty_summary(analysis):
    simulation = _long_simulation()
    simulation.filter(custom_filter=lambda frame: frame[frame["scenario"] == "shock"])
    result = simulation.evaluate(analysis, source="model", k=0)
    assert set(result["scenario"]) == {"shock"}
    assert len(result) == 2
    assert "date_range" not in result._metadata
    assert len(result._metadata["path_metadata"]) == 2
    populated = simulation.aggregate_results(result)
    simulation.filter(custom_filter=lambda frame: frame.iloc[:0])
    empty = simulation.aggregate_results(simulation.evaluate(analysis, source="model", k=0))
    assert empty.empty
    assert empty.dtypes.to_dict() == populated.dtypes.to_dict()
    assert empty.columns.tolist() == populated.columns.tolist()
    simulation.clear_filter()
    assert set(simulation.evaluate(analysis, source="model", k=0)["scenario"]) == {"baseline", "shock"}


def test_bias_contract_summarises_coefficients_not_diagnostics():
    result = _long_simulation().evaluate(bias_analysis, source="model", k=0)
    result._df["bias_estimate"] = [1.0, 3.0, -2.0, 4.0]
    result._df["p_value"] = [0.01, 0.99, 0.1, 0.8]
    result._df["ci_lower"] = [-10.0, -20.0, -30.0, -40.0]
    summary = _simulation().aggregate_results(result).to_df().set_index("scenario")
    assert len(summary) == 2
    assert summary.loc["baseline", "bias_estimate_mean"] == 2.0
    assert summary.loc["baseline", "bias_estimate_se"] == 1.0
    assert not {"p_value", "ci_lower", "std_error", "hac_maxlags"}.intersection(summary.columns)
