import pandas as pd
import pytest

from forecast_evaluation import ForecastData, SimulationData
from forecast_evaluation.core.outturns_revisions_table import create_outturn_revisions

from .simulation_data import create_monte_carlo_data

PATHS = [(0, "baseline"), (1, "baseline"), (0, "shock"), (1, "shock")]


def _forecasts(*, source: str = "model", value_shift: float = 0) -> pd.DataFrame:
    dates = pd.to_datetime(["2022-12-31", "2023-03-31"])
    rows = []
    for draw, scenario in PATHS:
        offset = draw * 20 + (100 if scenario == "shock" else 0)
        for horizon, date in enumerate(dates):
            index = 14 + horizon
            rows.append(
                {
                    "draw": draw,
                    "scenario": scenario,
                    "date": date,
                    "vintage_date": pd.Timestamp("2022-09-30"),
                    "variable": "y",
                    "frequency": "Q",
                    "source": source,
                    "forecast_horizon": horizon,
                    "value": 100 + index * 2 + offset + 0.75 + value_shift,
                }
            )

    return pd.DataFrame(rows)


def _path_frame(frame: pd.DataFrame, key: tuple[int, str]) -> pd.DataFrame:
    draw, scenario = key
    return (
        frame.loc[(frame["draw"] == draw) & (frame["scenario"] == scenario)]
        .drop(columns=["draw", "scenario"])
        .reset_index(drop=True)
    )


def _ordinary(
    outturns: pd.DataFrame,
    forecasts: pd.DataFrame,
    key: tuple[int, str],
    *,
    outturn_vintages: bool = True,
) -> ForecastData:
    return ForecastData(
        outturns_data=_path_frame(outturns, key),
        forecasts_data=_path_frame(forecasts, key),
        outturn_vintages=outturn_vintages,
        data_check=False,
    )


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not len(frame.columns):
        return frame.reset_index(drop=True)
    return frame.sort_values(list(frame.columns), kind="stable", na_position="last").reset_index(drop=True)


def _assert_frames_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        _canonical(actual),
        _canonical(expected),
        check_dtype=False,
        check_like=True,
    )


def _assert_child_matches(actual: ForecastData, expected: ForecastData) -> None:
    for attribute in ("_raw_outturns", "_raw_forecasts", "outturns", "forecasts", "df"):
        _assert_frames_equal(getattr(actual, attribute), getattr(expected, attribute))


def _tag(frame: pd.DataFrame, key: tuple[int, str]) -> pd.DataFrame:
    tagged = frame.copy()
    tagged.insert(0, "scenario", key[1])
    tagged.insert(0, "draw", key[0])
    return tagged


@pytest.fixture
def isolated_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return create_monte_carlo_data(draws=2, periods=20, vintages=3, seed=7), _forecasts()


def test_level_transformations_are_isolated_and_match_ordinary_children(isolated_inputs):
    outturns, forecasts = isolated_inputs
    simulation = SimulationData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        data_check=False,
    )

    for key, panel in simulation.iter_panels():
        expected = _ordinary(outturns, forecasts, key)
        _assert_child_matches(panel, expected)
        assert not {"draw", "scenario"}.intersection(panel.forecasts.columns)

    target_date = pd.Timestamp("2022-12-31")
    for metric in ("pop", "yoy"):
        values = simulation.forecasts.loc[
            simulation.forecasts["date"].eq(target_date)
            & simulation.forecasts["metric"].eq(metric)
            & simulation.forecasts["source"].eq("model"),
            ["draw", "scenario", "value"],
        ]
        assert len(values) == len(PATHS)
        assert values["value"].nunique() == len(PATHS)


@pytest.mark.parametrize("metric", ["pop", "yoy"])
def test_non_level_forecasts_reconstruct_levels_per_path(isolated_inputs, metric):
    outturns, forecasts = isolated_inputs
    forecasts = forecasts.assign(metric=metric, value=0.1 if metric == "pop" else 0.2)
    simulation = SimulationData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        data_check=False,
    )

    for key, panel in simulation.iter_panels():
        expected = _ordinary(outturns, forecasts, key)
        _assert_child_matches(panel, expected)

    levels = simulation.forecasts.loc[
        simulation.forecasts["date"].eq(pd.Timestamp("2022-12-31"))
        & simulation.forecasts["metric"].eq("levels")
        & simulation.forecasts["source"].eq("model"),
        ["draw", "scenario", "value"],
    ]
    assert len(levels) == len(PATHS)
    assert levels["value"].nunique() == len(PATHS)


def test_main_tables_and_revisions_are_joined_per_path(isolated_inputs):
    outturns, forecasts = isolated_inputs
    simulation = SimulationData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        data_check=False,
    )
    expected_panels = {key: _ordinary(outturns, forecasts, key) for key in PATHS}

    expected_main_table = pd.concat(
        [_tag(expected_panels[key].df, key) for key in PATHS],
        ignore_index=True,
    )
    _assert_frames_equal(simulation.df, expected_main_table)

    identity = [
        "draw",
        "scenario",
        "date",
        "variable",
        "vintage_date_forecast",
        "vintage_date_outturn",
        "unique_id",
        "metric",
        "frequency",
        "forecast_horizon",
    ]
    assert not simulation.df.duplicated(identity).any()

    revisions = simulation.evaluate(create_outturn_revisions)
    expected_revisions = pd.concat(
        [_tag(create_outturn_revisions(expected_panels[key]), key) for key in PATHS],
        ignore_index=True,
    )
    _assert_frames_equal(revisions, expected_revisions)
    assert set(revisions[["draw", "scenario"]].itertuples(index=False, name=None)) == set(PATHS)


def test_no_vintage_synthetic_transformations_match_ordinary_children():
    outturns = create_monte_carlo_data(draws=2, periods=20, vintages=3, seed=7, include_vintages=False)
    forecasts = _forecasts()
    simulation = SimulationData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        outturn_vintages=False,
        data_check=False,
    )

    for key, panel in simulation.iter_panels():
        expected = _ordinary(outturns, forecasts, key, outturn_vintages=False)
        _assert_child_matches(panel, expected)
        assert panel.outturns["vintage_date"].isna().all()


def test_pseudo_vintages_are_created_per_path_like_ordinary_children(isolated_inputs):
    outturns, forecasts = isolated_inputs
    simulation = SimulationData(outturns_data=outturns, forecasts_data=forecasts, data_check=False)
    simulation.create_pseudo_vintages(fill_to="2021-03-31", vintage_frequency="Q")

    for key, panel in simulation.iter_panels():
        expected = _ordinary(outturns, forecasts, key)
        expected.create_pseudo_vintages(fill_to="2021-03-31", vintage_frequency="Q")
        _assert_child_matches(panel, expected)


def test_benchmarks_are_added_independently_per_path(isolated_inputs):
    outturns, forecasts = isolated_inputs
    simulation = SimulationData(outturns_data=outturns, forecasts_data=forecasts, data_check=False)
    simulation.add_benchmarks(
        models=["random_walk"],
        variables="y",
        frequency="Q",
        metric="levels",
        forecast_periods=2,
        show_progress=False,
    )

    for key, panel in simulation.iter_panels():
        expected = _ordinary(outturns, forecasts, key)
        expected.add_benchmarks(
            models=["random_walk"],
            variables="y",
            frequency="Q",
            metric="levels",
            forecast_periods=2,
            show_progress=False,
        )
        _assert_child_matches(panel, expected)

    benchmark_values = simulation.forecasts.loc[
        simulation.forecasts["source"].eq("baseline random walk model") & simulation.forecasts["metric"].eq("levels"),
        ["draw", "scenario", "value"],
    ]
    assert benchmark_values["value"].nunique() == len(PATHS)


def test_copy_and_merge_preserve_independent_paths(isolated_inputs):
    outturns, forecasts = isolated_inputs
    forecasts_a = forecasts.assign(source="model_a")
    forecasts_b = forecasts.assign(source="model_b", value=forecasts["value"] + 3)
    simulation_a = SimulationData(outturns_data=outturns, forecasts_data=forecasts_a, data_check=False)
    simulation_b = SimulationData(outturns_data=outturns, forecasts_data=forecasts_b, data_check=False)

    copied = simulation_a.copy()
    copied._panels[PATHS[0]]._raw_forecasts.loc[0, "value"] += 1000
    assert (
        copied._panels[PATHS[0]]._raw_forecasts.loc[0, "value"]
        != simulation_a.panels[PATHS[0]]._raw_forecasts.loc[0, "value"]
    )

    merged = simulation_a.copy()
    with pytest.warns(UserWarning, match="Removed .* duplicate"):
        merged.merge(simulation_b)

    combined_forecasts = pd.concat([forecasts_a, forecasts_b], ignore_index=True)
    for key, panel in merged.iter_panels():
        expected = _ordinary(outturns, combined_forecasts, key)
        _assert_child_matches(panel, expected)
    assert set(simulation_a.panels[PATHS[0]]._raw_forecasts["source"]) == {"model_a"}


def test_normalised_duplicate_identity_within_and_across_calls():
    outturns = pd.DataFrame(
        {
            "draw": [1, 1, 2, 2],
            "scenario": [10, 20, 10, 20],
            "date": pd.Timestamp("2022-06-30"),
            "variable": "y",
            "frequency": "Q",
            "value": [100.0, 110.0, 120.0, 130.0],
        }
    )
    forecasts = pd.DataFrame(
        {
            "draw": [1, 1, 2, 2],
            "scenario": [10, 20, 10, 20],
            "date": pd.Timestamp("2022-06-30"),
            "vintage_date": pd.Timestamp("2022-03-31"),
            "variable": "y",
            "frequency": "Q",
            "source": "model",
            "forecast_horizon": 0,
            "value": [99.0, 109.0, 119.0, 129.0],
        }
    )
    duplicated_outturns = pd.concat(
        [outturns, outturns.iloc[[0]].assign(draw="1", scenario="10")],
        ignore_index=True,
    )
    simulation = SimulationData(
        outturns_data=duplicated_outturns,
        outturn_vintages=False,
    )
    duplicated_forecasts = pd.concat(
        [forecasts, forecasts.iloc[[0]].assign(draw="1.0", scenario="10")],
        ignore_index=True,
    )
    simulation.add_forecasts(duplicated_forecasts, data_check=False)

    with pytest.warns(UserWarning, match="Removed .* duplicate"):
        simulation.add_outturns(outturns.iloc[[0]].assign(draw="1.0", scenario="10"))
    with pytest.warns(UserWarning, match="Removed .* duplicate"):
        simulation.add_forecasts(forecasts.iloc[[0]].assign(draw="1", scenario="10"), data_check=False)

    assert set(simulation.panels) == {(1, "10"), (1, "20"), (2, "10"), (2, "20")}
    assert all(len(panel._raw_outturns) == 1 for panel in simulation.panels.values())
    assert all(len(panel._raw_forecasts) == 1 for panel in simulation.panels.values())


def test_false_compute_levels_is_inherited_by_batched_forecast_additions(isolated_inputs):
    outturns, forecasts = isolated_inputs
    simulation = SimulationData(outturns_data=outturns, compute_levels=False, data_check=False)
    pop_forecasts = forecasts.assign(metric="pop")

    for draw in sorted(pop_forecasts["draw"].unique()):
        simulation.add_forecasts(
            pop_forecasts.loc[pop_forecasts["draw"].eq(draw)],
            data_check=False,
        )

    assert set(simulation.forecasts["metric"]) == {"pop"}
    simulation.clear_filter()
    assert set(simulation.forecasts["metric"]) == {"pop"}


def test_compatible_false_compute_levels_merge_preserves_non_level_forecasts(isolated_inputs):
    outturns, forecasts = isolated_inputs
    pop_forecasts = forecasts.assign(metric="pop")
    first = SimulationData(outturns_data=outturns, compute_levels=False, data_check=False)
    second = SimulationData(outturns_data=outturns, compute_levels=False, data_check=False)
    first.add_forecasts(pop_forecasts.assign(source="model_a"), data_check=False)
    second.add_forecasts(pop_forecasts.assign(source="model_b"), data_check=False)

    first.merge(second)
    before_clear = first.forecasts.copy()

    assert set(first.forecasts["metric"]) == {"pop"}
    assert set(first.forecasts["source"]) == {"model_a", "model_b"}
    first.clear_filter()
    _assert_frames_equal(first.forecasts, before_clear)
    assert set(first.forecasts["metric"]) == {"pop"}


def test_normalised_extra_ids_can_be_ingested_and_aggregated(isolated_inputs):
    outturns, forecasts = isolated_inputs
    simulation = SimulationData(outturns_data=outturns, data_check=False)
    first_batch = forecasts.loc[forecasts["draw"].eq(0)].assign(**{"model label": "small"})
    second_batch = forecasts.loc[forecasts["draw"].eq(1)].assign(model_label="small")

    simulation.add_forecasts(first_batch, extra_ids=["model label"], data_check=False)
    simulation.add_forecasts(second_batch, data_check=False)

    accuracy = simulation.evaluate_accuracy(k=0)
    summary = simulation.aggregate_accuracy(accuracy, across=["draw"])

    assert simulation.id_columns == ["source", "model_label"]
    assert set(accuracy.to_df()["model_label"]) == {"small"}
    assert set(summary["model_label"]) == {"small"}


def test_merge_adopts_forecast_identity_from_forecast_bearing_collection(isolated_inputs):
    outturns, forecasts = isolated_inputs
    outturn_only = SimulationData(outturns_data=outturns, data_check=False)
    forecast_bearing = SimulationData(outturns_data=outturns, data_check=False)
    forecast_bearing.add_forecasts(
        forecasts.assign(model_label="small"),
        extra_ids=["model_label"],
        data_check=False,
    )

    outturn_only.merge(forecast_bearing)
    accuracy = outturn_only.evaluate_accuracy(k=0)
    summary = outturn_only.aggregate_accuracy(accuracy, across=["draw"])

    assert outturn_only.id_columns == ["source", "model_label"]
    assert not accuracy.to_df().empty
    assert set(summary["model_label"]) == {"small"}


def test_first_forecast_add_rejects_conflicting_compute_levels(isolated_inputs):
    outturns, forecasts = isolated_inputs
    simulation = SimulationData(outturns_data=outturns, compute_levels=False, data_check=False)

    with pytest.raises(ValueError, match="compute_levels"):
        simulation.add_forecasts(forecasts, compute_levels=True, data_check=False)

    assert simulation.forecasts.empty
