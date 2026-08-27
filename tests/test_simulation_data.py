import pandas as pd
import pytest

from forecast_evaluation import SimulationData


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


def test_omitting_first_forecast_horizon_preserves_panel_configuration():
    simulation_data = SimulationData(
        outturns_data=_outturns(),
        outturn_vintages=False,
        first_forecast_horizon=1,
    )
    forecasts = _forecasts().drop(columns="forecast_horizon")

    with pytest.warns(FutureWarning):
        simulation_data.add_forecasts(forecasts, data_check=False)

    assert simulation_data.panels[(1, "baseline")]._raw_forecasts["forecast_horizon"].iloc[0] == 0


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