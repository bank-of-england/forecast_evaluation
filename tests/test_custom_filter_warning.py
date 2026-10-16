"""Tests for the warning emitted when a custom_filter column exists in none of the filtered tables."""

import warnings

from forecast_evaluation.data import ForecastData
from forecast_evaluation.data.sample_data import create_sample_forecasts, create_sample_outturns

_SKIPPED = "custom_filter was not applied"


def _skipped_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if _SKIPPED in str(w.message)]


def _typo_filter(df):
    return df[df["valuee"] > 0]


def test_custom_filter_typo_warns_per_table_forecast_data():
    data = ForecastData(outturns_data=create_sample_outturns(), forecasts_data=create_sample_forecasts())
    sizes = (len(data.forecasts), len(data.outturns), len(data._main_table))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data.filter(custom_filter=_typo_filter)

    messages = _skipped_warnings(caught)
    assert len(messages) == 3
    assert all("valuee" in m for m in messages)
    for table in ("'forecasts'", "'outturns'", "'main table'"):
        assert sum(table in m for m in messages) == 1
    assert (len(data.forecasts), len(data.outturns), len(data._main_table)) == sizes


def test_custom_filter_typo_warns_per_table_density(density_and_point):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        density_and_point.filter(custom_filter=_typo_filter)

    messages = _skipped_warnings(caught)
    assert len(messages) == 4
    for table in ("'forecasts'", "'outturns'", "'main table'", "'density forecasts'"):
        assert sum(table in m for m in messages) == 1
    assert len(density_and_point.density_forecasts) == 650


def test_custom_filter_typo_warns_density_only(density_only):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        density_only.filter(custom_filter=_typo_filter)

    messages = _skipped_warnings(caught)
    assert len(messages) == 2
    for table in ("'outturns'", "'density forecasts'"):
        assert sum(table in m for m in messages) == 1


def test_custom_filter_table_specific_column_is_silent(density_and_point):
    n_forecasts, n_outturns = len(density_and_point.forecasts), len(density_and_point.outturns)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        density_and_point.filter(custom_filter=lambda df: df[df["quantile"] >= 0.5])

    assert len(density_and_point.density_forecasts) == 325
    assert len(density_and_point.forecasts) == n_forecasts
    assert len(density_and_point.outturns) == n_outturns


def test_custom_filter_real_column_is_silent_and_filters():
    data = ForecastData(outturns_data=create_sample_outturns(), forecasts_data=create_sample_forecasts())
    n_forecasts = len(data.forecasts)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data.filter(custom_filter=lambda df: df[df["value"] > 105])

    assert len(data.forecasts) < n_forecasts
