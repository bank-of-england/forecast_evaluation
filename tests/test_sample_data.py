import inspect
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from forecast_evaluation.data.sample_data import (
    create_sample_density_forecasts,
    create_sample_forecasts,
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
    create_sample_outturns,
)


@pytest.mark.parametrize(
    ("generator", "facts"),
    [
        (create_sample_outturns, ("13", "26", "100", "112", "quarterly", "index", "deterministic")),
        (create_sample_forecasts, ("13", "0", "12", "100", "112", "quarterly", "index", "deterministic")),
        (
            create_sample_nowcast_outturns,
            ("24", "42", "14", "0.3", "0.45", "0.5", "quarterly", "index", "default_rng(42)"),
        ),
        (
            create_sample_nowcast_forecasts,
            ("24", "(-1, 0, 1)", "max(offset, 0)", "182", "0.25", "weekly", "index", "default_rng(42)"),
        ),
        (
            create_sample_density_forecasts,
            ("13", "12", "115", "0.5", "10,000", "50", "0.01", "0.99", "quarterly", "index", "default_rng(42)"),
        ),
    ],
)
def test_generator_documented_facts(generator, facts):
    assert not inspect.signature(generator).parameters
    documentation = inspect.getdoc(generator)
    assert documentation
    for fact in facts:
        assert fact in documentation
    if generator is create_sample_nowcast_forecasts:
        assert "plot_sample_nowcasts" not in documentation


@pytest.mark.parametrize(
    "generator",
    [create_sample_nowcast_outturns, create_sample_nowcast_forecasts, create_sample_density_forecasts],
)
def test_stochastic_generator_repeatability_and_global_rng_isolation(generator):
    initial_state = np.random.get_state()
    try:
        np.random.seed(123)
        before = np.random.get_state()
        first = generator()
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]

        np.random.seed(987)
        before_second = np.random.get_state()
        second = generator()
        after_second = np.random.get_state()
        assert before_second[0] == after_second[0]
        np.testing.assert_array_equal(before_second[1], after_second[1])
        assert before_second[2:] == after_second[2:]
        pd.testing.assert_frame_equal(first, second)
    finally:
        np.random.set_state(initial_state)


def test_point_sample_structure_and_values():
    outturns = create_sample_outturns()
    forecasts = create_sample_forecasts()
    dates = pd.date_range("2022-03-31", periods=13, freq="QE")

    assert len(outturns) == 26
    assert set(outturns["variable"]) == {"cpisa", "gdpkp"}
    assert outturns["vintage_date"].eq(pd.Timestamp("2025-09-30")).all()
    for _, group in outturns.groupby("variable"):
        np.testing.assert_array_equal(group["date"], dates)
        assert group["value"].tolist() == list(range(100, 113))
    assert len(forecasts) == 13
    np.testing.assert_array_equal(forecasts["date"], dates)
    assert forecasts["value"].tolist() == list(range(100, 113))
    assert forecasts["forecast_horizon"].tolist() == list(range(13))
    assert forecasts["variable"].eq("gdpkp").all()
    assert forecasts["source"].eq("mpr2").all()
    assert forecasts["vintage_date"].eq(pd.Timestamp("2025-03-31")).all()
    for frame, generator in [(outturns, create_sample_outturns), (forecasts, create_sample_forecasts)]:
        assert frame["frequency"].eq("Q").all()
        pd.testing.assert_frame_equal(frame, generator())


def test_nowcast_outturn_structure_and_truth():
    outturns = create_sample_nowcast_outturns()
    dates = pd.date_range("2020-03-31", periods=24, freq="QE")

    assert len(outturns) == 1200
    assert outturns.groupby("variable").size().to_dict() == {"cpi": 900, "gdp": 300}
    assert outturns["frequency"].eq("Q").all()
    assert outturns["vintage_date"].max() == pd.Timestamp("2026-04-14")
    for variable, lag, base, trend in [("gdp", 42, 99.0, 0.3), ("cpi", 14, 98.0, 0.45)]:
        variable_rows = outturns[outturns["variable"] == variable]
        np.testing.assert_array_equal(variable_rows["date"].drop_duplicates(), dates)
        for target_index, (target, group) in enumerate(variable_rows.groupby("date")):
            assert group["vintage_date"].min() == target + pd.Timedelta(days=lag)
            cycle = np.sin(target_index / 4) * 0.5 if variable == "gdp" else np.cos(target_index / 4) * 0.8
            truth = base + target_index * trend + cycle
            ages = group["vintage_date"].dt.to_period("Q").astype("int64") - target.to_period("Q").ordinal
            assert ((group["value"] - truth).abs() <= 5 * 0.5 / (1 + ages.abs()) + 0.0001).all()


def test_nowcast_forecast_targets_and_publication_cutoff():
    forecasts = create_sample_nowcast_forecasts()
    dates = pd.date_range("2020-03-31", periods=24, freq="QE")

    np.testing.assert_array_equal(forecasts["date"].drop_duplicates().sort_values(), dates)
    assert forecasts["frequency"].eq("Q").all()
    assert set(forecasts["source"]) == {"nowcast_dfm", "nowcast_bridge"}
    assert set(forecasts["variable"]) == {"gdp", "cpi"}
    weekly_dates = pd.date_range("2020-01-01", "2026-03-31", freq="7D")
    assert forecasts["vintage_date"].isin(weekly_dates).all()
    offsets = forecasts["date"].dt.to_period("Q").astype("int64") - forecasts["vintage_date"].dt.to_period(
        "Q"
    ).astype("int64")
    assert set(offsets) == {-1, 0, 1}
    np.testing.assert_array_equal(forecasts["forecast_horizon"], offsets.clip(lower=0))
    lags = forecasts["variable"].map({"gdp": 42, "cpi": 14})
    publication_dates = forecasts["date"] + pd.to_timedelta(lags, unit="D")
    assert (forecasts.loc[offsets == -1, "vintage_date"] < publication_dates[offsets == -1]).all()
    assert not forecasts.duplicated(["date", "variable", "vintage_date", "source"]).any()
    for source in ["nowcast_dfm", "nowcast_bridge"]:
        for variable, lag in [("gdp", 42), ("cpi", 14)]:
            group = forecasts[(forecasts["source"] == source) & (forecasts["variable"] == variable)]
            expected_count = sum(
                target.to_period("Q").ordinal - vintage.to_period("Q").ordinal in (-1, 0, 1)
                and (target.to_period("Q") >= vintage.to_period("Q") or vintage < target + pd.Timedelta(days=lag))
                for target in dates
                for vintage in weekly_dates
            )
            assert len(group) == expected_count


def test_density_structure_center_and_scale():
    forecasts = create_sample_density_forecasts()
    quantiles = np.linspace(0.01, 0.99, 50)
    dates = pd.date_range("2025-12-31", periods=13, freq="QE")
    normal = NormalDist()
    expected_width = 0.5 * (normal.inv_cdf(0.84) - normal.inv_cdf(0.16))

    assert len(forecasts) == 650
    assert 0.5 not in quantiles
    assert forecasts["variable"].eq("gdpkp").all()
    assert forecasts["source"].eq("mpr2").all()
    assert forecasts["frequency"].eq("Q").all()
    assert forecasts["vintage_date"].eq(pd.Timestamp("2025-12-31")).all()
    np.testing.assert_array_equal(forecasts["date"].drop_duplicates(), dates)
    for horizon, (_, group) in enumerate(forecasts.groupby("date")):
        np.testing.assert_allclose(group["quantile"], quantiles)
        assert group["forecast_horizon"].eq(horizon).all()
        assert group["value"].diff().dropna().gt(0).all()
        lower, median, upper = np.interp([0.16, 0.5, 0.84], group["quantile"], group["value"])
        assert median == pytest.approx(115 + horizon, abs=0.05)
        assert upper - lower == pytest.approx(expected_width, rel=0.05)


def test_sample_density_fan_starts_near_outturn_index_levels():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from forecast_evaluation.data import DensityForecastData

    data = DensityForecastData(
        outturns_data=create_sample_outturns().assign(metric="levels"),
        forecasts_data=create_sample_density_forecasts().assign(metric="levels"),
        data_check=False,
    )
    figure = None
    try:
        with pytest.warns(UserWarning, match="not stored") as caught:
            figure, axes = data.plot_density_vintage("gdpkp", "2025-12-31", return_plot=True)

        assert len(caught) == 3
        lines = {line.get_label(): line for line in axes.lines}
        center = lines["mpr2 (q=0.49)"]
        history = lines["Outturns (latest vintage, before forecast)"]
        center_values = np.asarray(center.get_ydata(), dtype=float)
        np.testing.assert_allclose(center_values, 115 + np.arange(13), atol=0.05)
        assert history.get_ydata()[-1] == 112
        assert 0 < center_values[0] - history.get_ydata()[-1] < 4
        assert len(center.get_xdata()) == 13

        bands = [collection for collection in axes.collections if isinstance(collection, PolyCollection)]
        assert len(bands) == 1
        assert bands[0].get_label() == "mpr2 (0.15-0.83)"
        vertices = bands[0].get_paths()[0].vertices
        lower = data.density_forecasts[np.isclose(data.density_forecasts["quantile"], 0.15)]
        upper = data.density_forecasts[np.isclose(data.density_forecasts["quantile"], 0.83)]
        assert vertices[:, 1].min() == pytest.approx(lower["value"].min())
        assert vertices[:, 1].max() == pytest.approx(upper["value"].max())
    finally:
        if figure is not None:
            plt.close(figure)