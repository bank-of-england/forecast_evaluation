"""Tests for the Rossi-Sekhposyan path calibration tests."""

import numpy as np
import pandas as pd
import pytest

from forecast_evaluation.tests import rossi_sekhposyan_test
from forecast_evaluation.tests.density_path_calibration import (
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


def test_testresult_interfaces():
    fitted = _make_data(n=60, horizons=(0, 1, 2)).to_distribution("normal")
    rs = rossi_sekhposyan_test(fitted, seed=1)
    df = rs.to_df()
    assert len(df) == 3 and {"ks_stat", "ks_cv_5", "cvm_reject_10", "block_length"} <= set(df.columns)
    assert rs._metadata["test_name"] == "rossi_sekhposyan_test"


@pytest.mark.parametrize(
    ("test_function", "options"),
    [
        pytest.param(rossi_sekhposyan_test, {}, id="rossi-sekhposyan"),
    ],
)
def test_path_tests_are_reproducible_by_default(test_function, options):
    fitted = _make_data(n=30, horizons=(0, 1, 2)).to_distribution("normal")
    first = test_function(fitted, n_boot=50, **options)
    second = test_function(fitted, n_boot=50, **options)
    explicit_seed = test_function(fitted, n_boot=50, seed=42, **options)

    pd.testing.assert_frame_equal(first.to_df(), second.to_df())
    pd.testing.assert_frame_equal(first.to_df(), explicit_seed.to_df())
    assert first._metadata["parameters"]["seed"] == 42


@pytest.mark.parametrize(
    ("test_function", "options"),
    [
        pytest.param(rossi_sekhposyan_test, {}, id="rossi-sekhposyan"),
    ],
)
def test_path_tests_respect_random_state(test_function, options):
    fitted = _make_data(n=30, horizons=(0, 1, 2)).to_distribution("normal")
    generator = np.random.default_rng(7)
    first = test_function(fitted, n_boot=50, seed=generator, **options)
    second = test_function(fitted, n_boot=50, seed=generator, **options)
    explicit_seed = test_function(fitted, n_boot=50, seed=7, **options)
    fresh = test_function(fitted, n_boot=50, seed=None, **options)
    critical_values = [column for column in first.to_df().columns if "_cv" in column]

    pd.testing.assert_frame_equal(first.to_df(), explicit_seed.to_df())
    assert not first.to_df()[critical_values].equals(second.to_df()[critical_values])
    assert fresh._metadata["parameters"]["seed"] is None


def test_narrow_density_rejects():
    fitted = _make_data(n=80, scale=0.5).to_distribution("normal")
    df = rossi_sekhposyan_test(fitted, seed=2).to_df()
    assert df["ks_reject_5"].all() and df["cvm_reject_5"].all()
