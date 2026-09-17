import pandas as pd
import pytest

from forecast_evaluation import SimulationData

from .simulation_data import create_monte_carlo_data


def _outturns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "draw": [1, 2, 1, 2],
            "scenario": ["baseline", "baseline", "shock", "shock"],
            "date": ["2020-06-30"] * 4,
            "variable": ["y"] * 4,
            "frequency": ["Q"] * 4,
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "draw": [1, 2, 1, 2],
            "scenario": ["baseline", "baseline", "shock", "shock"],
            "date": ["2020-06-30"] * 4,
            "vintage_date": ["2020-03-31"] * 4,
            "variable": ["y"] * 4,
            "frequency": ["Q"] * 4,
            "source": ["model"] * 4,
            "forecast_horizon": [0] * 4,
            "value": [9.0, 22.0, 29.0, 44.0],
        }
    )


@pytest.fixture
def simulation_data() -> SimulationData:
    return SimulationData(
        outturns_data=_outturns(),
        forecasts_data=_forecasts(),
        outturn_vintages=False,
        data_check=False,
    )


def test_simulation_data_partitions_and_tags_paths(simulation_data):
    assert simulation_data.simulation_ids == ["draw", "scenario"]
    assert set(simulation_data.panels) == {
        (1, "baseline"),
        (2, "baseline"),
        (1, "shock"),
        (2, "shock"),
    }
    assert "draw" not in simulation_data.panels[(1, "baseline")].outturns.columns
    assert set(simulation_data.df[["draw", "scenario"]].itertuples(index=False, name=None)) == set(
        simulation_data.panels
    )


def test_iter_panels_yields_non_empty_paths(simulation_data):
    panels = list(simulation_data.iter_panels())

    assert [key for key, _ in panels] == list(simulation_data.panels)


def test_clear_filter_handles_outturn_only_simulations():
    simulation_data = SimulationData(outturns_data=_outturns(), outturn_vintages=False)

    simulation_data.clear_filter()

    assert len(simulation_data.outturns) == len(_outturns())
    assert simulation_data.df.empty


def test_accuracy_is_evaluated_per_path_and_aggregation_retains_scenario(simulation_data):
    accuracy = simulation_data.evaluate_accuracy(k=0)
    result = accuracy.to_df()

    assert set(result[["draw", "scenario"]].itertuples(index=False, name=None)) == set(simulation_data.panels)
    assert result.groupby(["draw", "scenario"], dropna=False)["mse"].first().to_dict() == {
        (1, "baseline"): 1.0,
        (2, "baseline"): 4.0,
        (1, "shock"): 1.0,
        (2, "shock"): 16.0,
    }

    summary = simulation_data.aggregate_accuracy(accuracy, across=["draw"]).to_df()
    assert set(summary["scenario"]) == {"baseline", "shock"}
    assert summary.set_index("scenario").loc["baseline", "mse_mean"] == 2.5
    assert summary.set_index("scenario").loc["shock", "mse_mean"] == 8.5
    assert set(summary["n_draws"]) == {2}


def test_custom_filter_can_select_a_simulation_path(simulation_data):
    simulation_data.filter(custom_filter=lambda frame: frame[frame["scenario"] == "shock"])

    assert set(simulation_data.df["scenario"]) == {"shock"}
    simulation_data.clear_filter()
    assert set(simulation_data.df["scenario"]) == {"baseline", "shock"}


def test_evaluation_visits_only_active_paths(simulation_data):
    simulation_data.filter(custom_filter=lambda frame: frame[frame["scenario"] == "shock"])
    visited = []

    def analyse(panel):
        visited.append(panel)
        return pd.DataFrame({"estimate": [1.0]})

    result = simulation_data.evaluate(analyse)

    assert len(visited) == 2
    assert set(result["scenario"]) == {"shock"}
    simulation_data.filter(custom_filter=lambda frame: frame.iloc[:0])
    assert simulation_data.evaluate(analyse).empty
    assert len(visited) == 2


def test_evaluation_labels_active_path_failures(simulation_data):
    def analyse(panel):
        raise ValueError("invalid sample")

    with pytest.raises(RuntimeError, match="draw.*1.*scenario.*baseline.*invalid sample") as error:
        simulation_data.evaluate(analyse)

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("simulation_ids", "message"),
    [
        ([], "non-empty"),
        (["scenario"], "include 'draw'"),
        (["draw", "draw"], "unique"),
        (["draw", "date"], "reserved"),
    ],
)
def test_simulation_id_configuration_is_validated(simulation_ids, message):
    with pytest.raises(ValueError, match=message):
        SimulationData(simulation_ids=simulation_ids)


@pytest.mark.parametrize("draw", [None, -1, 1.5, "draw_1"])
def test_draw_values_are_non_negative_integers(draw):
    data = _outturns().assign(draw=draw)
    with pytest.raises(ValueError, match="draw"):
        SimulationData(outturns_data=data, outturn_vintages=False)


def test_draw_values_must_fit_in_int64():
    data = _outturns().assign(draw=2**63)

    with pytest.raises(ValueError, match="draw"):
        SimulationData(outturns_data=data, outturn_vintages=False)


def test_simulation_forecasts_require_explicit_horizon():
    simulation_data = SimulationData(
        outturns_data=_outturns(),
        outturn_vintages=False,
        first_forecast_horizon=1,
    )
    forecasts = _forecasts().drop(columns="forecast_horizon")

    with pytest.raises(ValueError, match="forecast_horizon"):
        simulation_data.add_forecasts(forecasts, data_check=False)

    assert simulation_data.forecasts.empty


def test_unknown_forecast_path_is_rejected():
    forecasts = _forecasts().assign(draw=lambda frame: frame["draw"].replace(2, 3))
    simulation_data = SimulationData(outturns_data=_outturns(), outturn_vintages=False)
    with pytest.raises(ValueError, match="has no outturns"):
        simulation_data.add_forecasts(forecasts, data_check=False)
    assert all(panel._raw_forecasts.empty for panel in simulation_data.panels.values())


def test_late_outturn_vintage_rebuilds_main_table():
    outturns = pd.DataFrame(
        {
            "draw": [1],
            "scenario": ["baseline"],
            "date": ["2020-06-30"],
            "vintage_date": ["2020-06-30"],
            "variable": ["y"],
            "frequency": ["Q"],
            "value": [10.0],
        }
    )
    forecasts = _forecasts().iloc[[0]].copy()
    simulation_data = SimulationData(outturns_data=outturns, forecasts_data=forecasts, data_check=False)
    assert len(simulation_data.df) == 1

    later_vintage = outturns.assign(vintage_date="2020-09-30", value=11.0)
    simulation_data.add_outturns(later_vintage)

    assert len(simulation_data.df) == 2


def test_aggregate_results_rejects_unknown_identifier(simulation_data):
    result = pd.DataFrame(
        {
            "draw": [1],
            "scenario": ["baseline"],
            "mse": [1.0],
        }
    )
    with pytest.raises(ValueError, match="unknown"):
        simulation_data.aggregate_results(result, across=["replicate"])


@pytest.mark.parametrize("forecast", [False, True])
@pytest.mark.parametrize("mode", ["batch", "call", "merge"])
def test_conflicting_normalised_identities_are_rejected(forecast, mode):
    outturns = _outturns().assign(metric="levels")
    forecasts = _forecasts().assign(metric="levels")
    original = forecasts if forecast else outturns
    changed = original.assign(
        date="2020-05-01",
        value=original["value"] + 0.000001,
        ignored_column="discarded by the child schema",
    )
    simulation = SimulationData(outturns_data=outturns, outturn_vintages=False)
    if forecast and mode != "batch":
        simulation.add_forecasts(forecasts, data_check=False)
    before = simulation.df.copy()
    with pytest.raises(ValueError, match="Duplicate.*different values"):
        if mode == "merge":
            other = SimulationData(outturns_data=outturns if forecast else changed, outturn_vintages=False)
            if forecast:
                other.add_forecasts(changed, data_check=False)
            simulation.merge(other)
        else:
            frame = pd.concat([original, changed], ignore_index=True) if mode == "batch" else changed
            if forecast:
                simulation.add_forecasts(frame, data_check=False)
            else:
                simulation.add_outturns(frame)
    pd.testing.assert_frame_equal(simulation.df, before)


def test_custom_identifier_order_and_forecast_labels_are_preserved():
    simulation = SimulationData(
        outturns_data=_outturns().assign(parameter_set=7),
        simulation_ids=["parameter_set", "scenario", "draw"],
        outturn_vintages=False,
    )
    simulation.add_forecasts(
        _forecasts().assign(parameter_set=7, model_label="small"), extra_ids=["model_label"], data_check=False
    )
    assert simulation.simulation_ids == ["parameter_set", "scenario", "draw"]
    assert list(simulation.forecasts.columns[:3]) == simulation.simulation_ids
    assert ("7", "baseline", 1) in simulation.panels
    accuracy = simulation.evaluate_accuracy(k=0)
    summary = simulation.aggregate_accuracy(accuracy)
    assert set(summary["parameter_set"]) == {"7"}
    assert set(summary["model_label"]) == {"small"}
    returned_ids = simulation.simulation_ids
    returned_ids.clear()
    assert simulation.simulation_ids == ["parameter_set", "scenario", "draw"]


@pytest.mark.parametrize("identifier", ["horizon", "forecast_horizon", "unique_id", "source"])
def test_reserved_identifiers_are_rejected(identifier):
    with pytest.raises(ValueError, match=identifier):
        SimulationData(simulation_ids=["draw", identifier])


@pytest.mark.parametrize(
    "identifier",
    [
        "vintage_date_forecast",
        "vintage_date_outturn",
        "value_forecast",
        "value_outturn",
        "k",
        "latest_vintage",
        "forecast_error",
    ],
)
def test_derived_column_names_are_reserved_simulation_ids(identifier):
    with pytest.raises(ValueError, match=identifier):
        SimulationData(simulation_ids=["draw", identifier])


@pytest.mark.parametrize("layout", ["overlapping", "disjoint", "empty_destination"])
def test_incompatible_default_k_merge_leaves_both_simulations_unchanged(layout):
    if layout == "overlapping":
        left_outturns = _outturns().iloc[:2]
        right_outturns = _outturns().iloc[[0, 2]]
    elif layout == "disjoint":
        left_outturns = _outturns().query("scenario == 'baseline'")
        right_outturns = _outturns().query("scenario == 'shock'")
    else:
        left_outturns = None
        right_outturns = _outturns()

    left = SimulationData(outturns_data=left_outturns, outturn_vintages=False, default_k=1)
    right = SimulationData(outturns_data=right_outturns, outturn_vintages=False, default_k=2)
    left_before = left.df.copy()
    right_before = right.df.copy()

    with pytest.raises(ValueError, match="different default_k"):
        left.merge(right)

    pd.testing.assert_frame_equal(left.df, left_before)
    pd.testing.assert_frame_equal(right.df, right_before)


def test_matching_default_k_merge_succeeds():
    left = SimulationData(
        outturns_data=_outturns().query("scenario == 'baseline'"),
        outturn_vintages=False,
        default_k=1,
    )
    right = SimulationData(
        outturns_data=_outturns().query("scenario == 'shock'"),
        outturn_vintages=False,
        default_k=1,
    )

    left.merge(right)

    assert set(left.panels) == set(zip(_outturns()["draw"], _outturns()["scenario"], strict=True))
    assert left.default_k == right.default_k == 1


def test_empty_accuracy_retains_custom_forecast_identity():
    simulation = SimulationData(outturns_data=_outturns(), outturn_vintages=False)
    simulation.add_forecasts(_forecasts().assign(model_label="small"), extra_ids=["model_label"], data_check=False)
    populated = simulation.aggregate_accuracy(simulation.evaluate_accuracy(k=0))
    simulation.filter(custom_filter=lambda frame: frame.iloc[:0])
    empty = simulation.aggregate_accuracy(simulation.evaluate_accuracy(k=0))
    assert empty.empty
    assert empty.columns.tolist() == populated.columns.tolist()
    assert empty.dtypes.to_dict() == populated.dtypes.to_dict()


def test_monte_carlo_data_is_reproducible_and_simulation_ready():
    first = create_monte_carlo_data(draws=3, scenarios=("baseline", "shock"), vintages=2, seed=7)
    second = create_monte_carlo_data(draws=3, scenarios=("baseline", "shock"), vintages=2, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["draw"]) == {0, 1, 2}
    assert set(first["scenario"]) == {"baseline", "shock"}
    assert first["vintage_date"].nunique() == 2
    simulation = SimulationData(outturns_data=first)
    assert len(simulation.panels) == 6


@pytest.mark.parametrize("scenario", [None, "", "   "])
def test_scenario_labels_must_be_present(scenario):
    with pytest.raises(ValueError, match="scenario"):
        SimulationData(outturns_data=_outturns().assign(scenario=scenario), outturn_vintages=False)


@pytest.mark.parametrize("compute_levels", [False, True])
@pytest.mark.parametrize("operation", ["add", "merge"])
def test_conflicting_compute_levels_are_rejected_without_changes(compute_levels, operation):
    simulation = SimulationData(
        outturns_data=_outturns(),
        forecasts_data=_forecasts().iloc[[0]],
        outturn_vintages=False,
        compute_levels=compute_levels,
        data_check=False,
    )
    before = simulation.df.copy()

    with pytest.raises(ValueError, match="compute_levels"):
        if operation == "add":
            simulation.add_forecasts(_forecasts().iloc[[1]], compute_levels=not compute_levels, data_check=False)
        else:
            other = SimulationData(
                outturns_data=_outturns().iloc[[1]],
                forecasts_data=_forecasts().iloc[[1]],
                outturn_vintages=False,
                compute_levels=not compute_levels,
                data_check=False,
            )
            simulation.merge(other)

    pd.testing.assert_frame_equal(simulation.df, before)


@pytest.mark.parametrize("labelled_first", [False, True])
@pytest.mark.parametrize("operation", ["add", "merge"])
def test_conflicting_forecast_identity_schemas_are_rejected_without_changes(labelled_first, operation):
    simulation = SimulationData(outturns_data=_outturns(), outturn_vintages=False)
    simulation.add_forecasts(
        _forecasts().iloc[[0]].assign(model_label="small"),
        extra_ids=["model_label"] if labelled_first else [],
        data_check=False,
    )
    before = simulation.df.copy()
    forecasts = _forecasts().iloc[[1]].assign(model_label="small")
    extra_ids = [] if labelled_first else ["model_label"]

    with pytest.raises(ValueError, match="extra_ids"):
        if operation == "add":
            simulation.add_forecasts(forecasts, extra_ids=extra_ids, data_check=False)
        else:
            other = SimulationData(outturns_data=_outturns().iloc[[1]], outturn_vintages=False)
            other.add_forecasts(forecasts, extra_ids=extra_ids, data_check=False)
            simulation.merge(other)

    pd.testing.assert_frame_equal(simulation.df, before)
