"""Tests for plot_density_vintage."""

import inspect
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

matplotlib.use("Agg")  # non-interactive backend; must be set before any figure is created

import forecast_evaluation as fe  # noqa: E402
import forecast_evaluation.data as data_package  # noqa: E402
import forecast_evaluation.visualisations as visualisations_package  # noqa: E402
from forecast_evaluation.data import DensityForecastData, create_sample_density_forecasts  # noqa: E402
from forecast_evaluation.visualisations.density_plots import plot_density_vintage  # noqa: E402
from forecast_evaluation.visualisations.forecast import plot_vintage  # noqa: E402

VINTAGE = "2025-12-31"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _plot(data, **kwargs):
    kwargs.setdefault("variable", "gdpkp")
    kwargs.setdefault("vintage_date", VINTAGE)
    kwargs.setdefault("return_plot", True)
    return plot_density_vintage(data, **kwargs)


def _density_on_grid(density_outturns_df, template, levels):
    columns = ["date", "variable", "vintage_date", "source", "frequency", "forecast_horizon", "metric"]
    base = template[columns].drop_duplicates().sort_values("date").reset_index(drop=True)
    date_order = base["date"].rank(method="dense").to_numpy()
    frames = []
    for level in levels:
        rows = base.copy()
        rows["quantile"] = float(level)
        rows["value"] = date_order + float(level)
        frames.append(rows)
    return DensityForecastData(
        outturns_data=density_outturns_df.copy(),
        forecasts_data=pd.concat(frames, ignore_index=True),
        data_check=False,
    )


def _polycollections(ax):
    return [collection for collection in ax.collections if isinstance(collection, PolyCollection)]


def test_plot_density_vintage_signature_matches_plot_vintage():
    function_parameters = inspect.signature(plot_density_vintage).parameters
    method_parameters = inspect.signature(DensityForecastData.plot_density_vintage).parameters

    assert list(function_parameters)[:7] == list(inspect.signature(plot_vintage).parameters)[:7]
    assert list(method_parameters)[1:7] == list(function_parameters)[1:7]
    for parameters in (function_parameters, method_parameters):
        assert parameters["quantiles"].default is None
        assert parameters["quantiles"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["return_plot"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "data" not in method_parameters
    assert "kwargs" not in method_parameters


def test_default_quantiles_resolved_when_none(density_only):
    with pytest.warns(UserWarning, match=r"0\.16.*0\.15"):
        _, ax = _plot(density_only, quantiles=None)

    labels = ax.get_legend_handles_labels()[1]
    assert "mpr2 (0.15-0.83)" in labels
    assert "mpr2 (q=0.49)" in labels
    assert "mpr2 (median)" not in labels


@pytest.mark.parametrize("route", ["function", "method"])
def test_callers_quantiles_not_mutated(density_only, route):
    requested = [0.05, 0.16, 0.5, 0.84, 0.95]
    with pytest.warns(UserWarning, match="not stored"):
        if route == "function":
            _plot(density_only, quantiles=requested)
        else:
            density_only.plot_density_vintage("gdpkp", VINTAGE, quantiles=requested, return_plot=True)
    assert requested == [0.05, 0.16, 0.5, 0.84, 0.95]


def test_default_quantiles_stable_across_calls(density_only, density_df, density_outturns_df):
    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        _plot(density_only)
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        _plot(density_only)
    assert len(first) == len(second) == 3

    exact = _density_on_grid(density_outturns_df, density_df, [0.16, 0.5, 0.84])
    with warnings.catch_warnings(record=True) as exact_warnings:
        warnings.simplefilter("always")
        _, ax = _plot(exact)
    assert not exact_warnings
    assert "mpr2 (0.16-0.84)" in ax.get_legend_handles_labels()[1]
    assert inspect.signature(plot_density_vintage).parameters["quantiles"].default is None


def test_no_warning_for_level_present_up_to_float_noise(density_only):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _plot(density_only, quantiles=[0.05, 0.15, 0.95])
    assert not [warning for warning in caught if "not stored" in str(warning.message)]


def test_missing_quantile_warning_names_requested_and_used(density_outturns_df, sparse_grid_density_df):
    data = _density_on_grid(density_outturns_df, sparse_grid_density_df, np.arange(0.1, 1.0, 0.1))

    with pytest.warns(UserWarning, match=r"0\.16.*0\.2"):
        _plot(data, quantiles=[0.16])


@pytest.mark.parametrize(
    "quantiles",
    [[], [float("nan")], [float("inf")], [-0.1], [1.1], [0.5, 0.5], [0.5, 0.5000000005]],
)
def test_invalid_quantile_requests_raise_value_error(density_only, quantiles):
    with pytest.raises(ValueError, match="[Qq]uantile"):
        _plot(density_only, quantiles=quantiles)


def test_distinct_requests_resolving_to_same_stored_level_raise(density_outturns_df, sparse_grid_density_df):
    data = _density_on_grid(density_outturns_df, sparse_grid_density_df, np.arange(0.1, 1.0, 0.1))

    with pytest.raises(ValueError, match="both resolve to the same stored level"):
        _plot(data, quantiles=[0.49, 0.51])


def test_unsorted_quantiles_use_sorted_outer_band(density_df, density_outturns_df):
    data = _density_on_grid(density_outturns_df, density_df, [0.05, 0.16, 0.5, 0.84, 0.95])
    _, ax = _plot(data, quantiles=[0.84, 0.5, 0.16])

    bands = _polycollections(ax)
    assert len(bands) == 1
    assert bands[0].get_label() == "mpr2 (0.16-0.84)"
    band_values = bands[0].get_paths()[0].vertices[:, 1]
    assert band_values.min() == pytest.approx(1.16)
    assert band_values.max() == pytest.approx(13.84)


def test_two_quantiles_draw_one_outer_band(density_df, density_outturns_df):
    data = _density_on_grid(density_outturns_df, density_df, [0.1, 0.9])
    _, ax = _plot(data, quantiles=[0.1, 0.9])

    bands = _polycollections(ax)
    assert len(bands) == 1
    assert bands[0].get_label() == "mpr2 (0.1-0.9)"


def test_multiple_quantiles_draw_one_outer_band(density_df, density_outturns_df):
    levels = [0.05, 0.16, 0.5, 0.84, 0.95]
    data = _density_on_grid(density_outturns_df, density_df, levels)
    _, ax = _plot(data, quantiles=levels)

    bands = _polycollections(ax)
    assert len(bands) == 1
    assert bands[0].get_alpha() == pytest.approx(0.1)
    assert bands[0].get_label() == "mpr2 (0.05-0.95)"


@pytest.mark.parametrize(
    ("levels", "expected_label"),
    [([0.05, 0.5, 0.95], "mpr2 (median)"), ([0.1, 0.3, 0.9], "mpr2 (q=0.3)"), ([0.1, 0.9], None)],
)
def test_central_line_labels_follow_requested_levels(density_df, density_outturns_df, levels, expected_label):
    data = _density_on_grid(density_outturns_df, density_df, levels)
    _, ax = _plot(data, quantiles=levels)

    labels = ax.get_legend_handles_labels()[1]
    if expected_label is None:
        assert not any("(median)" in label or "(q=" in label for label in labels)
    else:
        assert expected_label in labels


def test_requested_median_resolved_to_nearest_level_remains_solid(density_df, density_outturns_df):
    data = _density_on_grid(density_outturns_df, density_df, [0.15, 0.49, 0.83])

    with pytest.warns(UserWarning, match=r"0\.5.*0\.49"):
        _, ax = _plot(data, quantiles=[0.16, 0.5])

    labels = ax.get_legend_handles_labels()[1]
    assert "mpr2 (q=0.49)" in labels
    assert "mpr2 (median)" not in labels


def test_one_quantile_draws_a_line_without_a_band(density_df, density_outturns_df):
    data = _density_on_grid(density_outturns_df, density_df, [0.5])
    _, ax = _plot(data, quantiles=[0.5])

    assert not _polycollections(ax)
    assert "mpr2 (median)" in ax.get_legend_handles_labels()[1]


def test_outturn_labels_show_latest_vintage_and_forecast_split(density_only):
    latest = density_only._outturns.copy()
    older = latest.copy()
    older["vintage_date"] = pd.Timestamp("2025-06-30")
    older["value"] = -999.0
    future = latest[(latest["variable"] == "gdpkp") & (latest["metric"] == "levels")].iloc[[-1]].copy()
    future["date"] = pd.Timestamp("2026-03-31")
    future["value"] = 113.0
    density_only._outturns = pd.concat([latest, older, future], ignore_index=True)

    _, ax = _plot(density_only, quantiles=[0.15, 0.49, 0.83])
    labels = ax.get_legend_handles_labels()[1]
    outturn_lines = [line for line in ax.lines if line.get_label().startswith("Outturns")]
    outturn_values = np.concatenate([line.get_ydata() for line in outturn_lines])

    assert "Outturns (latest vintage, before forecast)" in labels
    assert "Outturns (latest vintage, post forecast)" in labels
    assert not any("(solid)" in label or "(dashed)" in label for label in labels)
    assert -999.0 not in outturn_values
    assert 113.0 in outturn_values


@pytest.mark.parametrize("route", ["function", "method"])
def test_legacy_positional_quantiles_warn_and_remain_supported(density_df, density_outturns_df, route):
    data = _density_on_grid(density_outturns_df, density_df, [0.05, 0.5, 0.95])

    with pytest.warns(DeprecationWarning, match="pass quantiles="):
        if route == "function":
            _, ax = plot_density_vintage(data, "gdpkp", VINTAGE, [0.05, 0.5, 0.95], return_plot=True)
        else:
            _, ax = data.plot_density_vintage("gdpkp", VINTAGE, [0.05, 0.5, 0.95], return_plot=True)

    assert "mpr2 (0.05-0.95)" in ax.get_legend_handles_labels()[1]


def test_string_forecast_sources_are_not_reinterpreted_as_quantiles(density_only):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _plot(density_only, forecast_source=["mpr2"], quantiles=[0.15, 0.49, 0.83])

    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


def test_wrapper_rejects_unknown_keywords(density_only):
    assert "kwargs" not in inspect.signature(DensityForecastData.plot_density_vintage).parameters
    with pytest.raises(TypeError, match="foo"):
        density_only.plot_density_vintage("gdpkp", VINTAGE, foo=True)


def test_density_plot_helpers_are_exported():
    assert "plot_density_vintage" in visualisations_package.__all__
    assert visualisations_package.plot_density_vintage is plot_density_vintage
    assert "plot_density_vintage" in fe.__all__
    assert fe.plot_density_vintage is plot_density_vintage
    assert "create_sample_density_forecasts" in data_package.__all__
    assert data_package.create_sample_density_forecasts is create_sample_density_forecasts
    assert "create_sample_density_forecasts" in fe.__all__
    assert fe.create_sample_density_forecasts is create_sample_density_forecasts


def test_plot_docstrings_document_quantiles():
    assert "quantiles :" in plot_density_vintage.__doc__
    assert "quantiles :" in DensityForecastData.plot_density_vintage.__doc__


def test_missing_density_data_raises_value_error(density_only):
    with pytest.raises(ValueError, match="No density forecasts found"):
        _plot(density_only, variable="cpisa")
