"""Tests for the AR(p) benchmark model parameter estimation."""

import numpy as np
import pytest

from forecast_evaluation.core.ar_p_model import fit_ar_t_mle, generate_ar_t_forecasts


def _simulate_ar(coeffs, const, sigma, n, burn_in=500, seed=0):
    """Simulate an AR(p) process with Gaussian errors.

    Parameters
    ----------
    coeffs : list of float
        AR coefficients [phi_1, phi_2, ...].
    const : float
        Constant term.
    sigma : float
        Standard deviation of the innovations.
    n : int
        Number of observations to return.
    burn_in : int, optional
        Number of initial observations to discard. Default is 500.
    seed : int, optional
        Random seed. Default is 0.

    Returns
    -------
    np.ndarray
        Simulated series of length ``n``.
    """
    rng = np.random.default_rng(seed)
    p = len(coeffs)
    total = n + burn_in
    eps = rng.normal(0.0, sigma, size=total)
    y = np.zeros(total)

    for t in range(p, total):
        y[t] = const + np.dot(coeffs, y[t - p : t][::-1]) + eps[t]

    return y[burn_in:]


def test_fit_ar1_recovers_parameters():
    """fit_ar_t_mle should recover the true AR(1) parameters from simulated data."""
    true_const = 0.5
    true_phi = 0.6
    true_sigma = 1.0

    y = _simulate_ar([true_phi], const=true_const, sigma=true_sigma, n=5000, seed=42)

    result = fit_ar_t_mle(y, p=1)

    assert result["success"]
    assert result["ar_coeffs"][0] == pytest.approx(true_phi, abs=0.05)
    assert result["constant"] == pytest.approx(true_const, abs=0.15)
    # The t-distribution scale should be close to the innovation std for near-normal errors.
    assert result["scale"] == pytest.approx(true_sigma, abs=0.15)


def test_fit_ar2_recovers_parameters():
    """fit_ar_t_mle should recover the true AR(2) parameters from simulated data."""
    true_const = 0.2
    true_phis = [0.5, 0.3]
    true_sigma = 1.0

    y = _simulate_ar(true_phis, const=true_const, sigma=true_sigma, n=8000, seed=7)

    result = fit_ar_t_mle(y, p=2)

    assert result["success"]
    assert result["ar_coeffs"][0] == pytest.approx(true_phis[0], abs=0.05)
    assert result["ar_coeffs"][1] == pytest.approx(true_phis[1], abs=0.05)
    # Implied unconditional mean: const / (1 - phi_1 - phi_2).
    implied_mean = result["constant"] / (1 - result["ar_coeffs"].sum())
    true_mean = true_const / (1 - sum(true_phis))
    assert implied_mean == pytest.approx(true_mean, abs=0.3)
    assert result["scale"] == pytest.approx(true_sigma, abs=0.15)


def test_generate_ar1_forecasts_match_recursion():
    """AR(1) forecasts should follow the deterministic conditional-mean recursion."""
    const = 0.5
    phi = 0.6
    y = np.array([1.0, 2.0, 3.0])
    model_result = {"success": True, "constant": const, "ar_coeffs": np.array([phi])}

    steps = 5
    forecasts = generate_ar_t_forecasts(y, model_result, steps=steps)

    # Manually compute the recursion: y_hat_{h} = const + phi * y_hat_{h-1}.
    expected = []
    last = y[-1]
    for _ in range(steps):
        last = const + phi * last
        expected.append(last)

    assert forecasts.shape == (steps,)
    assert np.allclose(forecasts, expected)


def test_generate_ar2_forecasts_match_recursion():
    """AR(2) forecasts should use the two most recent values in the recursion."""
    const = 0.2
    phis = np.array([0.5, 0.3])
    y = np.array([10.0, 11.0, 12.0, 13.0])
    model_result = {"success": True, "constant": const, "ar_coeffs": phis}

    steps = 4
    forecasts = generate_ar_t_forecasts(y, model_result, steps=steps)

    # Manually compute the AR(2) recursion using the last two observations.
    history = list(y[-2:])
    expected = []
    for _ in range(steps):
        nxt = const + phis[0] * history[-1] + phis[1] * history[-2]
        expected.append(nxt)
        history.append(nxt)

    assert forecasts.shape == (steps,)
    assert np.allclose(forecasts, expected)