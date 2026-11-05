"""Numerical regression snapshots captured from Python at commit 2104be9.

Inputs and random seeds are fixed; floating-point outputs are rounded to eight
decimal places. These snapshots preserve the assumed-correct baseline, not an
independent verification of the statistics. Updates require a justified change.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from forecast_evaluation.density.fitting import fit_quantile_densities
from forecast_evaluation.tests import (
    compute_crps,
    compute_pit,
    crps_analysis,
    galvao_mantoan_mitchell_test,
    knuppel_test,
    mitchell_weale_test,
    rossi_sekhposyan_test,
    sharpness_analysis,
)
from forecast_evaluation.tests._hac import nw_hac
from forecast_evaluation.tests.density_path_calibration import (
    brownian_bridge_h1_draws,
    galvao_mantoan_mitchell_statistic,
    rossi_sekhposyan_statistic,
)
from forecast_evaluation.tests.density_scoring import density_scores, interval_widths
from tests.test_density_calibration import _make_data
from tests.test_density_fitting import KNOWN, _density_data
from tests.test_density_scoring import Y, _normal_fit


def _snapshot_values(value):
    """Convert numerical outputs to stable Python values at the snapshot precision."""
    if isinstance(value, pd.DataFrame):
        return _snapshot_values(value.to_dict(orient="records"))
    if isinstance(value, np.ndarray):
        return _snapshot_values(value.tolist())
    if isinstance(value, dict):
        return {key: _snapshot_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_values(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 8)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


@pytest.mark.parametrize("scale", [1.0, 0.75], ids=["calibrated", "narrow"])
def test_density_calibration_snapshot(scale, snapshot):
    data = _make_data(n=80, scale=scale, horizons=(0, 1), seed=13)
    fitted = fit_quantile_densities(data.density_forecasts, "normal", None, data=data)
    pit = compute_pit(fitted, k="latest").to_df().sort_values(["horizon", "date"])
    results = {"pit": pit[["horizon", "date", "outturn", "pit"]].head(4)}
    for prewhite in (False, True):
        results[f"knuppel_prewhite_{prewhite}"] = (
            knuppel_test(fitted, k="latest", lags=2, prewhite=prewhite).to_df().sort_values("horizon")
        )
        results[f"mitchell_weale_prewhite_{prewhite}"] = (
            mitchell_weale_test(fitted, k="latest", alpha=0.10, lags=2, prewhite=prewhite)
            .to_df()
            .sort_values("horizon")
        )
    assert _snapshot_values(results) == snapshot


@pytest.mark.parametrize("prewhite", [False, True])
@pytest.mark.parametrize("lags", [0, 2, -1], ids=["zero", "two", "automatic"])
def test_hac_snapshot(lags, prewhite, snapshot):
    time = np.arange(60)
    moments = np.random.default_rng(37).standard_normal((60, 2))
    moments += np.column_stack([np.linspace(-0.5, 0.5, 60), np.sin(time / 3)])
    covariance, used_lags = nw_hac(moments, lags=lags, prewhite=prewhite)
    assert _snapshot_values({"covariance": covariance, "lags": used_lags}) == snapshot


@pytest.mark.parametrize("family", ["normal", "two_piece_normal", "skew_t", "mpz"])
def test_density_fitting_snapshot(family, snapshot):
    distribution, match_levels = KNOWN["normal" if family == "mpz" else family]
    data = _density_data(distribution)
    options = {"df_grid": range(5, 13), "n_shape_grid": 21} if family == "skew_t" else {}
    fitted = fit_quantile_densities(data.density_forecasts, family, match_levels, data=data, **options)
    central_values = fitted.ppf([0.25, 0.5, 0.75])
    results = {
        "params": fitted.params.sort_values("date").head(2),
        "diagnostics": fitted.diagnostics.sort_values("date").head(2),
        "moments": fitted.moments().sort_values("date").head(2),
        "quantiles": fitted.ppf([0.05, 0.25, 0.5, 0.75, 0.95])[:2],
        "pdf": fitted.pdf(central_values)[:2],
        "cdf": fitted.cdf(central_values)[:2],
    }
    assert _snapshot_values(results) == snapshot


def test_density_scoring_snapshot(snapshot):
    fitted = _normal_fit()
    results = {
        weight: {
            "value_grid": density_scores(fitted, Y, weight, n_points=2000, tail=1e-6),
            "quantile_grid": density_scores(fitted, Y, weight, grid="quantile_999"),
        }
        for weight in ("none", "centre", "tails", "left", "right", "heavy_tails")
    }
    results["interval_widths"] = interval_widths(fitted)

    data = _make_data(n=40, scale=0.75, horizons=(0, 1), seed=13)
    fitted = fit_quantile_densities(data.density_forecasts, "normal", None, data=data)
    per_forecast = compute_crps(fitted, k="latest").to_df().sort_values(["horizon", "date"])
    results["per_forecast"] = per_forecast[["horizon", "date", "outturn", "crps"]].head(4)
    results["crps_analysis"] = crps_analysis(fitted, k="latest", weight="left").to_df().sort_values("horizon")
    results["sharpness_analysis"] = sharpness_analysis(fitted, k="latest").to_df().sort_values(["horizon", "interval"])
    assert _snapshot_values(results) == snapshot


@pytest.mark.parametrize("weights", ["unweighted", "decreasing", "increasing"])
def test_path_statistics_snapshot(weights, snapshot):
    rng = np.random.default_rng(23)
    pits = stats.norm.cdf(1.3 * rng.standard_normal((60, 3)) + 0.2)
    rvec = np.linspace(0, 1, 101)
    block_length = 3
    rs_multipliers = rng.standard_normal((60 - block_length + 1, 200))
    gmm_multipliers = rng.standard_normal((60 - block_length + 1 + 3, 200))
    h1_draws = brownian_bridge_h1_draws(rvec, n_sim=2000, seed=19)
    results = {
        "rossi_sekhposyan": rossi_sekhposyan_statistic(
            pits[:, 0], rvec=rvec, block_length=block_length, eta=rs_multipliers
        ),
        "galvao_mantoan_mitchell": galvao_mantoan_mitchell_statistic(
            pits,
            weights=weights,
            alpha=0.05,
            rvec_stats=rvec,
            rvec=rvec,
            block_length=block_length,
            eta=gmm_multipliers,
            h1_draws=h1_draws,
        ),
    }
    assert _snapshot_values(results) == snapshot


def test_path_calibration_snapshot(snapshot):
    data = _make_data(n=80, scale=0.75, horizons=(0, 1, 2), seed=13)
    fitted = fit_quantile_densities(data.density_forecasts, "normal", None, data=data)
    results = {
        "rossi_sekhposyan": rossi_sekhposyan_test(fitted, k="latest", n_boot=200, seed=17)
        .to_df()
        .sort_values("horizon"),
        "galvao_mantoan_mitchell": galvao_mantoan_mitchell_test(
            fitted, k="latest", n_boot=200, seed=17, n_sim_h1=2000
        ).to_df(),
    }
    assert _snapshot_values(results) == snapshot
