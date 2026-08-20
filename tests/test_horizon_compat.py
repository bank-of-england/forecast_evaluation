"""Tests for the temporary ``forecast_horizon`` compatibility shims.

Covers both the deprecated ``forecast_horizon=`` keyword on analysis functions and the
deprecated ``forecast_horizon`` column lookup on ``TestResult``.

Revert this file together with ``src/forecast_evaluation/_compat.py`` when the
compatibility layer is dropped.
"""

import pandas as pd
import pytest

from forecast_evaluation.tests.bias import bias_analysis, evaluate_bias
from forecast_evaluation.tests.weak_efficiency import weak_efficiency_test

from .test_horizon_column import _mixed_lag_forecast_data


@pytest.fixture
def sample():
    data = _mixed_lag_forecast_data()
    df = data._main_table.assign(horizon=lambda d: d["target_minus_vintage"].astype(int))
    return df[df["metric"] == "levels"]


@pytest.fixture
def result():
    return bias_analysis(_mixed_lag_forecast_data(), verbose=False)


ARGS = ("gdp", "model", "levels")


def test_deprecated_kwarg_still_selects_a_horizon(sample):
    with pytest.warns(FutureWarning, match="forecast_horizon"):
        deprecated = evaluate_bias(sample, *ARGS, forecast_horizon=1, verbose=False)

    current = evaluate_bias(sample, *ARGS, horizon=1, verbose=False)

    assert deprecated.params.iloc[0] == current.params.iloc[0]


def test_deprecated_kwarg_warns_about_the_changed_meaning(sample):
    with pytest.warns(FutureWarning, match="target_minus_vintage"):
        evaluate_bias(sample, *ARGS, forecast_horizon=1, verbose=False)


def test_passing_both_arguments_is_rejected(sample):
    with pytest.raises(TypeError, match="both 'horizon' and the deprecated"):
        evaluate_bias(sample, *ARGS, horizon=1, forecast_horizon=1, verbose=False)


def test_current_keyword_does_not_warn(sample, recwarn):
    evaluate_bias(sample, *ARGS, horizon=1, verbose=False)

    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


def test_shim_covers_the_other_renamed_functions(sample):
    with pytest.warns(FutureWarning, match="weak_efficiency_test"):
        weak_efficiency_test(sample, *ARGS, forecast_horizon=1, verbose=False)


def test_deprecated_column_lookup_returns_the_horizon_column(result):
    with pytest.warns(FutureWarning, match="no longer a column"):
        deprecated = result["forecast_horizon"]

    pd.testing.assert_series_equal(deprecated, result["horizon"], check_names=False)


def test_deprecated_column_lookup_works_inside_a_list_of_columns(result):
    with pytest.warns(FutureWarning, match="no longer a column"):
        selected = result[["variable", "forecast_horizon"]]

    assert list(selected.columns) == ["variable", "horizon"]


def test_deprecated_attribute_access_returns_the_horizon_column(result):
    with pytest.warns(FutureWarning, match="no longer a column"):
        deprecated = result.forecast_horizon

    pd.testing.assert_series_equal(deprecated, result["horizon"], check_names=False)


def test_deprecated_column_warns_about_hac_maxlags(result):
    with pytest.warns(FutureWarning, match="hac_maxlags"):
        result["forecast_horizon"]


def test_deprecated_filter_keyword_selects_the_same_rows(result):
    with pytest.warns(FutureWarning, match="no longer a column"):
        deprecated = result.filter(forecast_horizon=1)

    pd.testing.assert_frame_equal(deprecated.to_df(), result.filter(horizon=1).to_df())
    assert deprecated._metadata["filters"]["horizon"] == 1


def test_filtering_on_both_horizon_names_is_rejected(result):
    with pytest.raises(TypeError, match="both 'horizon' and the deprecated"):
        result.filter(horizon=1, forecast_horizon=1)


def test_exports_never_carry_the_deprecated_column(result):
    assert "forecast_horizon" not in result.to_df().columns
    assert "forecast_horizon" not in result.to_csv()


def test_current_column_name_does_not_warn(result, recwarn):
    result["horizon"]
    result.filter(horizon=1)

    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


def test_unknown_column_still_raises(result):
    with pytest.raises(KeyError):
        result["not_a_column"]
