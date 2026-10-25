"""Smoke tests for density evaluation plots, TestResult.plot dispatch and package exports."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before any figure is created

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import forecast_evaluation as fe  # noqa: E402
from forecast_evaluation.tests import (  # noqa: E402
    compute_crps,
    compute_pit,
    crps_analysis,
    knuppel_test,
    rossi_sekhposyan_test,
    sharpness_analysis,
)
from forecast_evaluation.visualisations import (  # noqa: E402
    plot_crps_by_horizon,
    plot_cumulative_crps,
    plot_pit_ecdf,
    plot_pit_histogram,
    plot_sharpness_by_horizon,
)
from tests.test_density_calibration import _make_data  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def results():
    fitted = _make_data(n=40, horizons=(0, 1), sources=("a", "b")).fit_parametric_density("normal")
    return {
        "fitted": fitted,
        "pit": compute_pit(fitted),
        "rs": rossi_sekhposyan_test(fitted, seed=1),
        "sharpness": sharpness_analysis(fitted),
        "crps": crps_analysis(fitted),
        "crps_obs": compute_crps(fitted),
    }


PLOTS = {
    "pit_histogram": lambda r: plot_pit_histogram(r["pit"], source="a", return_plot=True),
    "pit_ecdf": lambda r: plot_pit_ecdf(r["pit"], source="a", return_plot=True),
    "pit_ecdf_rs_bands": lambda r: plot_pit_ecdf(r["pit"], source="a", rs_result=r["rs"], return_plot=True),
    "sharpness": lambda r: plot_sharpness_by_horizon(r["sharpness"], return_plot=True),
    "crps_by_horizon": lambda r: plot_crps_by_horizon(r["crps"], return_plot=True),
    "cumulative_crps": lambda r: plot_cumulative_crps(r["crps_obs"], horizon=0, return_plot=True),
    "cumulative_crps_vs_benchmark": lambda r: plot_cumulative_crps(
        r["crps_obs"], horizon=1, benchmark="b", return_plot=True
    ),
}


@pytest.mark.parametrize("plot", PLOTS)
def test_plot_smoke(plot, results):
    fig, ax = PLOTS[plot](results)
    assert fig is not None and ax is not None


def test_pit_plots_need_a_source(results):
    with pytest.raises(ValueError, match="Multiple sources"):
        plot_pit_histogram(results["pit"], return_plot=True)
    with pytest.raises(ValueError, match="level=5"):
        plot_pit_ecdf(results["pit"], source="a", level=10, return_plot=True)


DISPATCH = {
    "compute_pit": ("pit", {"source": "a"}),
    "compute_pit_ecdf": ("pit", {"source": "a", "kind": "ecdf"}),
    "sharpness_analysis": ("sharpness", {}),
    "crps_analysis": ("crps", {}),
    "compute_crps": ("crps_obs", {"horizon": 0}),
}


@pytest.mark.parametrize("test_name", DISPATCH)
def test_testresult_plot_dispatch(test_name, results):
    key, kwargs = DISPATCH[test_name]
    fig, _ = results[key].plot(return_plot=True, **kwargs)
    assert fig is not None


def test_testresult_plot_unsupported_density_tests(results):
    with pytest.raises(NotImplementedError, match="compute_pit"):
        knuppel_test(results["fitted"]).plot()
    with pytest.raises(ValueError, match="kind must be"):
        results["pit"].plot(kind="violin", source="a")


def test_exports_present():
    names = [
        "FittedDensityForecasts",
        "build_density_eval_table",
        "compute_pit",
        "knuppel_test",
        "mitchell_weale_test",
        "rossi_sekhposyan_test",
        "galvao_mantoan_mitchell_test",
        "compute_crps",
        "crps_analysis",
        "sharpness_analysis",
        "plot_pit_histogram",
        "plot_pit_ecdf",
        "plot_sharpness_by_horizon",
        "plot_crps_by_horizon",
        "plot_cumulative_crps",
    ]
    for name in names:
        assert name in fe.__all__ and hasattr(fe, name), name
