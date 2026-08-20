"""Tests for the ``horizon`` analysis dimension.

Analyses group, filter and report on ``target_minus_vintage`` — the calendar
distance between the forecast target date and the forecast vintage — under the
output name ``horizon``. ``forecast_horizon`` (the information horizon) is used
only to set serial-correlation lag lengths. The main table itself is unchanged
and carries both raw columns.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import acovf

import forecast_evaluation as fe
from forecast_evaluation.core.revisions_table import create_revision_dataframe
from forecast_evaluation.data.ForecastData import ForecastData

N_VINTAGES = 60
HORIZONS = (0, 1, 2)

# The publication lag lengthens from one period to two halfway through the sample, so
# every calendar horizon is observed at two different information horizons.
LAG_BEFORE_SWITCH = 1
LAG_AFTER_SWITCH = 2
SWITCH_AT = N_VINTAGES // 2


def _mixed_lag_forecast_data(sources=("model",), source_offsets=(0,)) -> ForecastData:
    """Build data where ``forecast_horizon`` and ``target_minus_vintage`` differ.

    Each source forecasts every target in ``HORIZONS`` at every quarterly vintage.
    Its information horizon exceeds the calendar horizon by the publication lag,
    which lengthens partway through the sample, plus a per-source offset so that
    two sources can disagree about how much outturn data they used at one vintage.
    """
    rng = np.random.default_rng(0)

    outturn_dates = pd.date_range("1999-03-31", periods=80, freq="QE")
    level = pd.Series(100 + np.arange(80) + rng.normal(0, 1, 80), index=outturn_dates)
    outturns = pd.DataFrame({"date": outturn_dates, "variable": "gdp", "frequency": "Q", "value": level.to_numpy()})

    vintages = pd.date_range("2000-03-31", periods=N_VINTAGES, freq="QE")
    rows = []
    for source, offset in zip(sources, source_offsets):
        for i, vintage in enumerate(vintages):
            publication_lag = LAG_BEFORE_SWITCH if i < SWITCH_AT else LAG_AFTER_SWITCH
            for horizon in HORIZONS:
                target = vintage + pd.offsets.QuarterEnd(horizon)
                rows.append(
                    {
                        "date": target,
                        "vintage_date": vintage,
                        "variable": "gdp",
                        "frequency": "Q",
                        "source": source,
                        "forecast_horizon": horizon + publication_lag + offset,
                        "value": level.get(target, np.nan) + rng.normal(0, 1),
                    }
                )

    forecasts = pd.DataFrame(rows).dropna(subset=["value"])

    return ForecastData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        outturn_vintages=False,
        compute_levels=False,
        data_check=False,
    )


@pytest.fixture
def mixed_lag_data() -> ForecastData:
    return _mixed_lag_forecast_data()


@pytest.fixture
def mixed_lag_two_sources() -> ForecastData:
    """Two sources that used different amounts of outturn data at the same vintage."""
    return _mixed_lag_forecast_data(sources=("model", "benchmark"), source_offsets=(0, 1))


# --- Inputs -------------------------------------------------------------------


def test_main_table_carries_both_horizon_columns(fer_minimal_fd: ForecastData):
    main_table = fer_minimal_fd._main_table

    assert "target_minus_vintage" in main_table.columns
    assert "forecast_horizon" in main_table.columns


def test_create_revision_dataframe_preserves_target_minus_vintage(fer_minimal_fd: ForecastData):
    revisions = create_revision_dataframe(fer_minimal_fd._main_table, fer_minimal_fd._forecasts, k=12)

    assert "target_minus_vintage" in revisions.columns
    assert "forecast_horizon" in revisions.columns


def test_fixture_separates_the_two_horizons(mixed_lag_data: ForecastData):
    """Guard the fixture itself: the two horizon columns must genuinely disagree."""
    pairs = mixed_lag_data._main_table[["target_minus_vintage", "forecast_horizon"]].drop_duplicates()

    assert set(pairs["target_minus_vintage"]) == set(HORIZONS)
    for horizon in HORIZONS:
        information_horizons = set(pairs.loc[pairs["target_minus_vintage"] == horizon, "forecast_horizon"])
        assert information_horizons == {horizon + LAG_BEFORE_SWITCH, horizon + LAG_AFTER_SWITCH}


# --- Results carry `horizon`, measured as the calendar distance ----------------


def _run(analysis, data):
    if analysis is fe.diebold_mariano_table:
        return analysis(data, benchmark_model="benchmark")
    return analysis(data)


ANALYSES = [
    fe.compute_accuracy_statistics,
    fe.bias_analysis,
    fe.weak_efficiency_analysis,
    fe.forecast_errors_correlation_analysis,
    fe.diebold_mariano_table,
]


@pytest.mark.parametrize("analysis", ANALYSES, ids=lambda a: a.__name__)
def test_results_emit_horizon_column(analysis, mixed_lag_two_sources: ForecastData):
    df = _run(analysis, mixed_lag_two_sources).to_df()

    assert "horizon" in df.columns
    assert "forecast_horizon" not in df.columns


@pytest.mark.parametrize("analysis", ANALYSES, ids=lambda a: a.__name__)
def test_results_group_by_calendar_horizon(analysis, mixed_lag_two_sources: ForecastData):
    """The reported horizons are the calendar ones, not the information horizons."""
    df = _run(analysis, mixed_lag_two_sources).to_df()

    assert set(df["horizon"]) == set(HORIZONS)


@pytest.mark.parametrize("analysis", ANALYSES, ids=lambda a: a.__name__)
def test_metadata_records_horizon_measure(analysis, mixed_lag_two_sources: ForecastData):
    result = _run(analysis, mixed_lag_two_sources)

    assert result._metadata["horizon_measure"] == "target_minus_vintage"


def test_mixed_information_horizons_pool_into_one_row(mixed_lag_data: ForecastData):
    """A publication-lag change must not split one calendar horizon into two rows."""
    df = fe.bias_analysis(mixed_lag_data, k=0, verbose=False).to_df()
    levels = df[df["metric"] == "levels"]

    assert levels["horizon"].tolist() == list(HORIZONS)
    assert (levels["n_observations"] == N_VINTAGES).all()


# --- Lag lengths follow the information horizon -------------------------------


@pytest.mark.parametrize("horizon", HORIZONS)
def test_bias_lag_is_the_groups_longest_information_horizon(horizon, mixed_lag_data: ForecastData):
    df = fe.bias_analysis(mixed_lag_data, k=0, verbose=False).to_df()
    row = df[(df["metric"] == "levels") & (df["horizon"] == horizon)].iloc[0]

    assert row["hac_maxlags"] == horizon + LAG_AFTER_SWITCH


@pytest.mark.parametrize("horizon", HORIZONS)
def test_weak_efficiency_lag_is_the_groups_longest_information_horizon(horizon, mixed_lag_data: ForecastData):
    df = fe.weak_efficiency_analysis(mixed_lag_data, k=0, verbose=False).to_df()
    row = df[(df["metric"] == "levels") & (df["horizon"] == horizon)].iloc[0]

    assert row["hac_maxlags"] == horizon + LAG_AFTER_SWITCH


def test_bias_standard_error_matches_independent_newey_west(mixed_lag_data: ForecastData):
    """Recompute the HAC standard error from scratch at the intended truncation.

    ``hac_maxlags`` is written by the same code that sets the lag, so checking it
    against itself proves nothing. This rebuilds the Newey-West variance of the mean
    error directly from the autocovariances and compares it to the reported standard
    error, which pins down the lag actually used.
    """
    horizon = 2
    result = fe.bias_analysis(mixed_lag_data, k=0, verbose=False).to_df()
    row = result[(result["metric"] == "levels") & (result["horizon"] == horizon)].iloc[0]

    sample = mixed_lag_data._main_table.assign(horizon=lambda d: d["target_minus_vintage"].astype(int))
    sample = sample[(sample["metric"] == "levels") & (sample["horizon"] == horizon)].sort_values("date")

    errors = sample["forecast_error"].to_numpy()
    n = len(errors)
    expected_lag = horizon + LAG_AFTER_SWITCH

    # Bartlett-weighted long-run variance of the sample mean, the estimator the bias
    # regression on a constant reduces to.
    demeaned = errors - errors.mean()
    autocovariances = acovf(demeaned, nlag=expected_lag, fft=False, demean=False)
    long_run_variance = autocovariances[0] + 2 * sum(
        (1 - lag / (expected_lag + 1)) * autocovariances[lag] for lag in range(1, expected_lag + 1)
    )

    assert row["std_error"] == pytest.approx(np.sqrt(long_run_variance / n))

    # And the shorter lag the calendar horizon would have implied gives a different answer,
    # so the test genuinely discriminates between the two rules.
    shorter = acovf(demeaned, nlag=horizon, fft=False, demean=False)
    shorter_variance = autocovariances[0] + 2 * sum(
        (1 - lag / (horizon + 1)) * shorter[lag] for lag in range(1, horizon + 1)
    )
    assert row["std_error"] != pytest.approx(np.sqrt(shorter_variance / n))


def test_diebold_mariano_pairs_sources_with_different_information_sets(mixed_lag_two_sources: ForecastData):
    """Dropping forecast_horizon from the pair key lets differently-informed sources compare."""
    result = fe.diebold_mariano_table(mixed_lag_two_sources, benchmark_model="benchmark").to_df()
    levels = result[result["metric"] == "levels"]

    assert levels["horizon"].tolist() == list(HORIZONS)
    assert (levels["n_observations"] > 0).all()

    # The benchmark used one period less outturn data, so its information horizon is the
    # longer of the pair and is what sets the truncation lag.
    for horizon in HORIZONS:
        row = levels[levels["horizon"] == horizon].iloc[0]
        assert row["hac_maxlags"] == horizon + LAG_AFTER_SWITCH + 1
