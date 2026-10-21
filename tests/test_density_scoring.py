"""Tests for density CRPS, weighted CRPS and sharpness."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from forecast_evaluation.density import FittedDensityForecasts
from forecast_evaluation.tests import compute_crps, crps_analysis, sharpness_analysis
from forecast_evaluation.tests.density_scoring import DEFAULT_INTERVALS, density_scores, interval_widths
from tests.test_density_calibration import _make_data

MU = np.array([0.0, 1.0, -0.5, 2.0, 0.0])
SIGMA = np.array([1.0, 0.5, 2.0, 1.5, 1.0])
Y = np.array([0.3, -1.0, 4.0, 2.0, 10.0])  # last outturn lies far outside the default grid


def _normal_fit():
    return FittedDensityForecasts.from_params(pd.DataFrame({"id": range(5), "loc": MU, "scale": SIGMA}), "normal")


def _normal_crps(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_normal_crps_closed_form_matches_grid():
    fitted = _normal_fit()
    exact = _normal_crps(Y, MU, SIGMA)
    np.testing.assert_allclose(density_scores(fitted, Y), exact, rtol=1e-5)
    np.testing.assert_allclose(density_scores(fitted, Y, grid="quantile_999"), exact, rtol=5e-3)
    with pytest.raises(ValueError, match="one outturn per density"):
        density_scores(fitted, Y[:2])
    with pytest.raises(ValueError, match="weight must be"):
        density_scores(fitted, Y, weight="middle")


def test_wcrps_uniform_weight_equals_crps():
    fitted = _normal_fit()
    crps = density_scores(fitted, Y)
    score = {w: density_scores(fitted, Y, w) for w in ("centre", "tails", "left", "right")}
    np.testing.assert_allclose(density_scores(fitted, Y, lambda t: np.ones_like(t)), crps, rtol=1e-12)
    # tau(1-tau) + (2tau-1)^2/4 = 1/4 and tau^2 + (1-tau)^2 + 2tau(1-tau) = 1: weighted CRPS is linear in v.
    np.testing.assert_allclose(score["centre"] + score["tails"] / 4, crps / 4, rtol=1e-10)
    np.testing.assert_allclose(score["right"] + score["left"] + 2 * score["centre"], crps, rtol=1e-10)
    # At the median of a symmetric density, left- and right-tail weights score alike.
    at_mean = density_scores(fitted, MU, "right")
    np.testing.assert_allclose(at_mean, density_scores(fitted, MU, "left"), rtol=1e-6)


def test_sharpness_widths_exact():
    widths = interval_widths(_normal_fit())
    assert list(widths.columns) == list(DEFAULT_INTERVALS)
    for name, (lo, hi) in DEFAULT_INTERVALS.items():
        np.testing.assert_allclose(widths[name], SIGMA * (stats.norm.ppf(hi) - stats.norm.ppf(lo)), rtol=1e-12)

    fitted = _make_data(n=30, scale=2.0, horizons=(0, 1)).fit_parametric_density("normal")
    result = sharpness_analysis(fitted, intervals={"central_90": (0.05, 0.95)})
    df = result.to_df()
    assert len(df) == 2 and set(df["horizon"]) == {0, 1}
    np.testing.assert_allclose(df["mean_width"], 2 * (stats.norm.ppf(0.95) - stats.norm.ppf(0.05)), rtol=1e-8)
    np.testing.assert_allclose(df["median_width"], df["mean_width"])
    assert (df["n_observations"] == 30).all() and (df["lower_level"] == 0.05).all()
    assert result._metadata["test_name"] == "sharpness_analysis"


def test_crps_analysis_grouping_and_metadata():
    fitted = _make_data(n=30, horizons=(0, 1), sources=("a", "b")).fit_parametric_density("normal")
    per_obs = compute_crps(fitted).to_df()
    assert len(per_obs) == 120
    np.testing.assert_allclose(per_obs["crps"], _normal_crps(per_obs["outturn"], 100, 1), rtol=1e-5)

    result = crps_analysis(fitted, weight="left")
    df = result.to_df()
    assert len(df) == 4 and set(df["source"]) == {"a", "b"} and (df["n_observations"] == 30).all()
    assert result._metadata["test_name"] == "crps_analysis"
    assert result._metadata["parameters"]["weight"] == "left"
