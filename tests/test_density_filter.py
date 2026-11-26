"""Tests for DensityForecastData.filter and to_point_forecast under an active filter."""

import pandas as pd
import pytest

from forecast_evaluation.data import DensityForecastData


def _a_quantile(data: DensityForecastData) -> str:
    """A quantile level present in the grid, as a ``to_point_forecast`` method string."""
    return str(data.density_forecasts["quantile"].iloc[0])


def _levels(data: DensityForecastData) -> pd.DataFrame:
    return data.forecasts[data.forecasts["metric"] == "levels"]


def test_filter_variables_filters_outturns_density_only(density_only):
    assert sorted(density_only.outturns["variable"].unique()) == ["cpisa", "gdpkp"]
    assert len(density_only.outturns) == 68

    density_only.filter(variables=["gdpkp"])

    assert density_only.outturns["variable"].unique().tolist() == ["gdpkp"]
    assert len(density_only.outturns) == 34
    assert len(density_only.density_forecasts) == 650


def test_filter_start_date_filters_outturns_density_only(density_only):
    assert density_only.outturns["date"].min() < pd.Timestamp("2024-01-01")

    density_only.filter(start_date="2024-01-01")

    assert density_only.outturns["date"].min() >= pd.Timestamp("2024-01-01")
    assert len(density_only.density_forecasts) == 650


def test_filter_empty_density_still_filters_outturns(density_only):
    density_only.filter(variables=["cpisa"])

    assert density_only.density_forecasts.empty
    assert density_only.outturns["variable"].unique().tolist() == ["cpisa"]


def test_filter_point_forecasts_false_leaves_outturns(density_only):
    density_only.filter(variables=["gdpkp"], filter_point_forecasts=False)

    assert len(density_only.outturns) == 68


def test_filter_with_point_forecasts_filters_outturns(density_and_point):
    density_and_point.filter(variables=["gdpkp"])

    assert density_and_point.outturns["variable"].unique().tolist() == ["gdpkp"]
    assert len(density_and_point.density_forecasts) == 650


def test_clear_filter_restores_outturns_and_density(density_only):
    snapshot = density_only._density_df.copy()

    density_only.filter(variables=["gdpkp"], start_date="2024-01-01")
    pd.testing.assert_frame_equal(density_only._density_df, snapshot)
    density_only.clear_filter()

    assert len(density_only.outturns) == 68
    assert len(density_only.density_forecasts) == 650
    pd.testing.assert_frame_equal(density_only._density_df, snapshot)


def test_to_point_forecast_respects_filter(density_only):
    method = _a_quantile(density_only)
    density_only.filter(start_date="2027-01-01")

    density_only.to_point_forecast(method)

    levels = _levels(density_only)
    assert levels["date"].nunique() == 8
    assert levels["date"].min() >= pd.Timestamp("2027-01-01")


def test_to_point_forecast_warns_when_filter_empties_density(density_only):
    method = _a_quantile(density_only)
    density_only.filter(variables=["cpisa"])

    with pytest.warns(UserWarning, match="No density forecasts remain"):
        density_only.to_point_forecast(method)

    assert density_only.forecasts.empty


def test_to_point_forecast_without_density_raises(density_outturns_df):
    data = DensityForecastData(outturns_data=density_outturns_df)

    with pytest.raises(ValueError, match="No density forecasts loaded"):
        data.to_point_forecast()


def test_filter_is_atomic_on_failure(density_and_point):
    def failing(df):
        if "quantile" in df.columns:
            raise RuntimeError("boom")
        return df

    with pytest.raises(RuntimeError, match="boom"):
        density_and_point.filter(variables=["gdpkp"], custom_filter=failing)

    assert len(density_and_point.outturns) == 68
    assert len(density_and_point.density_forecasts) == 650


def test_merge_ignores_other_filter(density_only, density_outturns_df):
    density_only.filter(start_date="2027-01-01")
    target = DensityForecastData(outturns_data=density_outturns_df)

    target.merge(density_only)

    assert len(target._density_df) == 650
