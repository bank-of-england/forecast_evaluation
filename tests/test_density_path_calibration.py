"""Tests for the Rossi-Sekhposyan and Galvao-Mantoan-Mitchell path calibration tests."""

import numpy as np
import pytest

from forecast_evaluation.tests import galvao_mantoan_mitchell_test, rossi_sekhposyan_test
from forecast_evaluation.tests.density_path_calibration import (
    brownian_bridge_h1_draws,
    galvao_mantoan_mitchell_statistic,
    gmm_weights,
    rossi_sekhposyan_statistic,
    rs_statistics,
)
from tests.test_density_calibration import _make_data


def test_rs_stat_known_case():
    # v(r) = (#{z < r} - 3r) / sqrt(3) at r = 0, .25, .5, .75, 1 is (0, .25, -.5, -.25, 0) / sqrt(3).
    ks, cvm = rs_statistics([0.1, 0.5, 0.9], rvec=[0, 0.25, 0.5, 0.75, 1])
    assert ks == pytest.approx(0.5 / np.sqrt(3))
    assert cvm == pytest.approx(0.375 / 5 / 3)

    pit = np.random.default_rng(0).uniform(size=50)
    zero = rossi_sekhposyan_statistic(pit, eta=np.zeros((50 - 3 + 1, 20)))
    assert zero["block_length"] == 3 and np.all(zero["ks_cv"] == 0) and zero["ks_reject"].all()
    a, b = (rossi_sekhposyan_statistic(pit, rng=7) for _ in range(2))
    np.testing.assert_array_equal(a["ks_cv"], b["ks_cv"])
    assert np.all(np.diff(a["ks_cv"]) <= 0)  # 1% >= 5% >= 10% critical values
    with pytest.raises(ValueError, match="eta must have"):
        rossi_sekhposyan_statistic(pit, eta=np.zeros((10, 20)))


def test_gmm_weights_sum_to_one():
    for H in range(1, 7):
        decreasing, increasing = gmm_weights(H)
        assert decreasing.sum() == pytest.approx(1) and increasing.sum() == pytest.approx(1)
        np.testing.assert_allclose(decreasing, increasing[::-1])
        assert np.all(np.diff(increasing) > 0) or H == 1


def test_h1_closed_form_cv():
    rvec = np.linspace(0, 1, 101)
    ks, cvm = brownian_bridge_h1_draws(rvec, n_sim=20000, seed=1, anchor=None)
    i95 = int(np.ceil(20000 * 0.95)) - 1
    # Asymptotic 5% values are 1.358 (KS) and 0.461 (CvM); a 101-point grid sits slightly below.
    assert 1.25 < ks[i95] < 1.40 and 0.42 < cvm[i95] < 0.50
    ks_a, cvm_a = brownian_bridge_h1_draws(rvec, n_sim=20000, seed=1)
    assert ks_a[i95] == pytest.approx(1.339) and cvm_a[i95] == pytest.approx(0.460)
    assert np.all(np.diff(ks_a) >= 0)


def test_gmm_statistic_shapes_and_weights():
    rng = np.random.default_rng(3)
    pits = rng.uniform(size=(60, 3))
    draws = brownian_bridge_h1_draws(np.linspace(0, 1, 101), n_sim=5000)
    res = galvao_mantoan_mitchell_statistic(
        pits, weights="decreasing", rvec=np.linspace(0, 1, 101), h1_draws=draws, rng=1
    )
    assert res["ks"].shape == (3,) and res["ks_cv_h"].shape == (3,) and res["ks_cv_bonf"].shape == (3,)
    np.testing.assert_allclose(res["weights"], gmm_weights(3)[0])
    assert res["ks_stat"] == pytest.approx(np.max(res["ks"] * res["weights"]))
    assert np.all(res["ks_cv_bonf"] >= res["ks_cv_h"])
    with pytest.raises(ValueError, match="weights"):
        galvao_mantoan_mitchell_statistic(pits, weights=[1, 2], h1_draws=draws)


def test_testresult_interfaces():
    fitted = _make_data(n=60, horizons=(0, 1, 2)).fit_parametric_density("normal")
    rs = rossi_sekhposyan_test(fitted, seed=1)
    df = rs.to_df()
    assert len(df) == 3 and {"ks_stat", "ks_cv_5", "cvm_reject_10", "block_length"} <= set(df.columns)
    assert rs._metadata["test_name"] == "rossi_sekhposyan_test"

    gmm = galvao_mantoan_mitchell_test(fitted, seed=1, n_sim_h1=5000)
    again = galvao_mantoan_mitchell_test(fitted, seed=1, n_sim_h1=5000)
    gdf = gmm.to_df()
    # Paths are forecast origins with a PIT at every horizon: 60 targets leave 58 complete origins.
    assert len(gdf) == 1 and gdf["horizons"].iloc[0] == (0, 1, 2) and gdf["n_observations"].iloc[0] == 58
    assert gdf["ks_cv"].iloc[0] == again.to_df()["ks_cv"].iloc[0]
    assert gmm._metadata["test_name"] == "galvao_mantoan_mitchell_test"
    bonf = galvao_mantoan_mitchell_test(fitted, seed=1, n_sim_h1=5000, bonferroni=True, weights="increasing")
    assert bonf._metadata["parameters"]["bonferroni"] is True


def test_narrow_density_rejects():
    fitted = _make_data(n=80, scale=0.5).fit_parametric_density("normal")
    df = rossi_sekhposyan_test(fitted, seed=2).to_df()
    assert df["ks_reject_5"].all() and df["cvm_reject_5"].all()
