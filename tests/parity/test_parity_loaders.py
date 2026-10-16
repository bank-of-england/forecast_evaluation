"""Tests for the MATLAB parity harness (loaders and skip logic)."""

import numpy as np
import pytest
from scipy.io import savemat

from tests.parity import _loaders
from tests.parity._loaders import load_ref, ref_available, require_ref


def test_parity_skips_when_refs_missing():
    assert not ref_available("does_not_exist_ref")
    with pytest.raises(pytest.skip.Exception):
        require_ref("does_not_exist_ref")


def test_parity_refs_load_and_have_expected_keys():
    refs = require_ref("refs_distributions")
    for key in ("skt_loc", "skt_scale", "skt_shape", "skt_df", "skt_x", "skt_tau", "skt_pdf", "skt_cdf", "skt_ppf"):
        assert key in refs
    assert not any(k.startswith("__") for k in refs)


def test_load_ref_reads_tiny_mat(tmp_path, monkeypatch):
    monkeypatch.setattr(_loaders, "PARITY_DIR", tmp_path)
    savemat(tmp_path / "tiny_ref.mat", {"a": np.arange(3.0), "b": 2.5, "s": {"x": 1.0}}, format="5")

    assert ref_available("tiny_ref")
    assert ref_available("tiny_ref.mat")
    refs = load_ref("tiny_ref")

    assert set(refs) == {"a", "b", "s"}
    np.testing.assert_array_equal(refs["a"], [0.0, 1.0, 2.0])
    assert refs["b"] == 2.5
    assert refs["s"].x == 1.0


def test_parity_marker_auto_applied(request):
    assert request.node.get_closest_marker("parity") is not None
