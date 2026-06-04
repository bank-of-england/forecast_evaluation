"""Tests for forecast_errors_correlation_analysis."""

import forecast_evaluation as fe


def test_forecast_errors_correlation_snapshot(fer_minimal_fd, snapshot):
    """Check that pairwise forecast-error correlations are computed correctly."""
    fd = fer_minimal_fd

    result = fe.forecast_errors_correlation_analysis(fd, k=12, min_observations=2)

    # Take a deterministic slice of the result so the snapshot stays small.
    df = result.to_df().sample(n=10, random_state=123)

    # Round floats to 10 decimal places to account for numerical precision differences
    df = df.round(10)

    assert df.to_dict() == snapshot
