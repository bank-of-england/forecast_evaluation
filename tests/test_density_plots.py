"""Tests for plot_density_vintage."""

import inspect
import warnings

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # non-interactive backend; must be set before any figure is created

from forecast_evaluation.data import DensityForecastData  # noqa: E402
from forecast_evaluation.visualisations.density_plots import plot_density_vintage  # noqa: E402

VINTAGE = "2025-12-31"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _plot(data, **kwargs):
    return plot_density_vintage(data, variable="gdpkp", vintage_date=VINTAGE, return_plot=True, **kwargs)


@pytest.mark.parametrize("func", [plot_density_vintage, DensityForecastData.plot_density_vintage])
def test_quantiles_default_is_none(func):
    assert inspect.signature(func).parameters["quantiles"].default is None


def test_default_quantiles_resolved_when_none(density_only):
    _, ax = _plot(density_only, quantiles=None)
    labels = ax.get_legend_handles_labels()[1]
    assert any("(0.15-0.83)" in label for label in labels)


@pytest.mark.parametrize("route", ["function", "method"])
def test_callers_quantiles_not_mutated(density_only, route):
    requested = [0.05, 0.16, 0.5, 0.84, 0.95]
    with pytest.warns(UserWarning, match="not found"):
        if route == "function":
            _plot(density_only, quantiles=requested)
        else:
            density_only.plot_density_vintage("gdpkp", VINTAGE, quantiles=requested, return_plot=True)
    assert requested == [0.05, 0.16, 0.5, 0.84, 0.95]


def test_default_quantiles_stable_across_calls(density_only):
    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        _plot(density_only)
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        _plot(density_only)
    assert len(first) == len(second) == 3


def test_no_warning_for_level_present_up_to_float_noise(density_only):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _plot(density_only, quantiles=[0.05, 0.15, 0.95])
    assert not [w for w in caught if "not found" in str(w.message)]
