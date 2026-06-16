"""Basic tests for the random walk benchmark model."""

import numpy as np
import pandas as pd

from forecast_evaluation.core.random_walk_model import build_random_walk_model
from forecast_evaluation.data.ForecastData import ForecastData


def _build_outturns(variable="x", frequency="Q", n=40, vintage_date=pd.Timestamp("2011-03-31")):
    """Create a simple single-vintage outturns DataFrame for testing."""
    dates = pd.date_range(start="2000-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(0)
    values = np.cumsum(rng.normal(0.0, 1.0, size=n)) + 100.0

    return pd.DataFrame(
        {
            "date": dates,
            "vintage_date": vintage_date,
            "variable": variable,
            "frequency": frequency,
            "forecast_horizon": -1,
            "value": values,
            "metric": "levels",
        }
    )


def test_random_walk_forecasts_equal_latest_outturn():
    """Random walk forecasts in levels should equal the latest observed outturn value."""
    outturns = _build_outturns()
    fd = ForecastData(outturns_data=outturns)

    forecast_periods = 13
    result = build_random_walk_model(
        data=fd,
        variable="x",
        metric="levels",
        frequency="Q",
        forecast_periods=forecast_periods,
        first_forecast_horizon=0,
    )

    assert not result.empty

    latest_value = outturns.sort_values("date")["value"].iloc[-1]

    # Every forecast row (all horizons) should reproduce the latest outturn value.
    assert np.allclose(result["value"].to_numpy(), latest_value)
    assert (result["source"] == "baseline random walk model").all()


def test_random_walk_forecast_dates_and_horizons():
    """Random walk should produce the expected number of forecast horizons and dates."""
    outturns = _build_outturns()
    fd = ForecastData(outturns_data=outturns)

    forecast_periods = 13
    result = build_random_walk_model(
        data=fd,
        variable="x",
        metric="levels",
        frequency="Q",
        forecast_periods=forecast_periods,
        first_forecast_horizon=0,
    )

    # One anchor row (horizon -1) plus ``forecast_periods`` forecast rows.
    assert len(result) == forecast_periods + 1
    assert sorted(result["forecast_horizon"].tolist()) == list(range(-1, forecast_periods))

    # Forecast dates should be consecutive quarter-ends after the last outturn date.
    last_outturn_date = outturns["date"].max()
    expected_future = pd.date_range(
        start=last_outturn_date + pd.offsets.QuarterEnd(), periods=forecast_periods, freq="QE"
    )
    future_dates = result[result["forecast_horizon"] >= 0]["date"].sort_values().to_numpy()
    assert np.array_equal(future_dates, expected_future.to_numpy())
