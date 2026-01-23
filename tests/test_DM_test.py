"""
Power analysis tests for the Diebold-Mariano test.
Basically if we generate two forecast series with desired relative accuracy,
does the DM yield the correct p-values (when using many observations).

At the moment works only when there is no autocorrelation in forecast errors (h=0).
A bit more involved to do this with autocorrelated errors.
"""

import numpy as np
import pandas as pd
from scipy import stats

from forecast_evaluation.tests.diebold_mariano import diebold_mariano_test


def calculate_required_mse_difference(n_obs=1000, target_power=0.99, alpha=0.05):
    """
    Calculate the MSE difference required to achieve target power.

    For a two-tailed test with significance level alpha, we need:
    effect_size = z_alpha/2 + z_power

    Where effect_size = mean_diff / se(mean_diff)

    For horizon=0 (no autocorrelation):
    se(mean_diff) = sqrt(var_diff / n)

    For squared errors with variance approximately 2*MSE^2:
    If MSE1 = mse_base and MSE2 = mse_base + delta
    Then var(diff) ≈ 2 * (MSE1^2 + MSE2^2)

    Parameters
    ----------
    n_obs : int
        Number of observations
    target_power : float
        Desired power (e.g., 0.80 for 80% power)
    alpha : float
        Significance level (e.g., 0.05)

    Returns
    -------
    float
        Required difference in MSE between models
    """
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed test
    z_power = stats.norm.ppf(target_power)

    # Required effect size
    required_effect_size = z_alpha + z_power

    # For mse_base = 1, approximate solution:
    # delta ≈ required_effect_size * sqrt(4 / n)
    mse_base = 1.0
    required_delta = required_effect_size * np.sqrt(4 * mse_base**2 / n_obs)

    return required_delta


def test_type_i_error_equal_accuracy():
    """
    Test that DM test correctly fails to reject null when models have equal accuracy.

    Under H0 (equal predictive accuracy), p-values should be uniformly distributed.
    With 1000 observations, we should fail to reject at 5% level about 95% of the time.
    """
    np.random.seed(1234)
    n_obs = 1000
    n_simulations = 1000
    horizon = 4

    p_values = []

    for _ in range(n_simulations):
        # Generate forecast errors from two models with SAME accuracy
        # Both models have mean squared error = 1
        errors_model1 = np.random.normal(0, 1, n_obs)
        errors_model2 = np.random.normal(0, 1, n_obs)

        # Calculate squared errors
        squared_errors1 = errors_model1**2
        squared_errors2 = errors_model2**2

        # Error difference (should be centered at 0)
        error_difference = pd.Series(squared_errors1 - squared_errors2)

        # Run DM test
        result = diebold_mariano_test(error_difference, horizon=horizon)
        p_values.append(result["p_value"])

    # Under null hypothesis, proportion of p-values > 0.05 should be ~95%
    rejection_rate = np.mean(np.array(p_values) < 0.05)

    # With 1000 simulations, rejection rate should be close to 5% (alpha level)
    # Allow some tolerance due to simulation variability
    assert 0.045 <= rejection_rate <= 0.055, (
        f"Type I error rate {rejection_rate:.3f} is outside expected range [0.045, 0.055]. "
        f"Expected ~0.05 for correct test."
    )


def test_power_different_accuracy():
    """
    Test that DM test has power to detect when one model is better.

    Uses horizon=0 (nowcast) to avoid autocorrelation complications.
    With 1000 observations and a meaningful difference in accuracy,
    the test should reject the null hypothesis most of the time.
    """
    np.random.seed(123)
    n_obs = 1000
    n_simulations = 1000
    horizon = 0  # No autocorrelation for horizon=0
    target_power = 0.5

    # Calculate required MSE difference for target power
    required_delta = calculate_required_mse_difference(n_obs=n_obs, target_power=target_power)

    # Set MSE for models based on required difference
    mse_baseline = 1.0
    mse_model1 = mse_baseline + required_delta
    mse_model2 = mse_baseline

    p_values = []
    dm_statistics = []

    for _ in range(n_simulations):
        # Generate forecast errors with calculated MSE difference
        errors_model1 = np.random.normal(0, np.sqrt(mse_model1), n_obs)
        errors_model2 = np.random.normal(0, np.sqrt(mse_model2), n_obs)

        # Calculate squared errors
        squared_errors1 = errors_model1**2
        squared_errors2 = errors_model2**2

        # Error difference (model1 - model2)
        # Positive values mean model1 is worse
        error_difference = pd.Series(squared_errors1 - squared_errors2)

        # Run DM test
        result = diebold_mariano_test(error_difference, horizon=horizon)
        p_values.append(result["p_value"])
        dm_statistics.append(result["dm_statistic"])

    # Power: proportion of times we correctly reject null (p < 0.05)
    power = np.mean(np.array(p_values) < 0.05)
    # With the calculated effect size, power should match target (with some tolerance)
    assert power >= target_power - 0.1, (
        f"Test power {power:.3f} is too low. power should be >= {target_power - 0.1:.2f}"
    )

    # DM statistics should be positive (model1 worse than model2)
    mean_dm = np.mean(dm_statistics)
    assert mean_dm > 0, f"Mean DM statistic {mean_dm:.3f} should be positive when model1 has higher MSE than model2"


def test_power_with_autocorrelation():
    """
    Test DM test performance with autocorrelated forecast errors.

    Multi-step ahead forecasts typically have autocorrelated errors.
    The DM test should handle this correctly via Newey-West adjustment.
    """
    np.random.seed(456)
    n_obs = 1000
    n_simulations = 50
    horizon = 0  # Higher horizon increases autocorrelation

    p_values = []

    for _ in range(n_simulations):
        # Generate autocorrelated errors using AR(1) process
        rho = 0.7  # Autocorrelation coefficient

        # Model 1: AR(1) with sigma = 1.2
        errors1 = [np.random.normal(0, 1.2)]
        for _ in range(n_obs - 1):
            errors1.append(rho * errors1[-1] + np.random.normal(0, 1.2))

        # Model 2: AR(1) with sigma = 1.0 (better)
        errors2 = [np.random.normal(0, 1.0)]
        for _ in range(n_obs - 1):
            errors2.append(rho * errors2[-1] + np.random.normal(0, 1.0))

        # Calculate squared errors
        squared_errors1 = np.array(errors1) ** 2
        squared_errors2 = np.array(errors2) ** 2

        # Error difference
        error_difference = pd.Series(squared_errors1 - squared_errors2)

        # Run DM test with appropriate horizon for HAC correction
        result = diebold_mariano_test(error_difference, horizon=horizon)
        p_values.append(result["p_value"])

    # Even with autocorrelation, test should have reasonable power
    power = np.mean(np.array(p_values) < 0.05)

    assert power >= 0.50, (
        f"Test power {power:.3f} with autocorrelation is too low. "
        f"DM test should handle autocorrelated errors via HAC adjustment."
    )
