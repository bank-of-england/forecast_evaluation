"""Shared pytest fixtures for forecast_evaluation tests."""

import functools

import pytest

from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.loader import load_fer_forecasts, load_fer_outturns
from forecast_evaluation.data.NowcastData import NowcastData
from forecast_evaluation.data.sample_data import (
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
)


@pytest.fixture
def fer_minimal_fd() -> ForecastData:
    """ForecastData loaded with minimal FER outturns and forecasts."""
    outturns = load_fer_outturns(minimal=True)
    forecasts = load_fer_forecasts(minimal=True)

    return ForecastData(outturns_data=outturns, forecasts_data=forecasts, compute_levels=False)


@functools.lru_cache(maxsize=1)
def _nowcast_fd_template() -> NowcastData:
    """Build the sample nowcast dataset once per test session.

    Ingesting the sample nowcast forecasts is by far the most expensive operation in the
    test suite (~1.8s), and dozens of tests need exactly this object. Building it once and
    handing out deep copies keeps every test fully independent while paying that cost only
    once. Do not use this function directly in tests -- use the ``nowcast_fd`` fixture,
    which copies it.
    """
    fd = NowcastData(outturns_data=create_sample_nowcast_outturns())
    fd.add_forecasts(create_sample_nowcast_forecasts(), data_check=False)

    return fd


@pytest.fixture
def nowcast_fd() -> NowcastData:
    """A fresh NowcastData built from the sample nowcast outturns and forecasts.

    Each test gets its own deep copy, so filtering, adding forecasts or otherwise
    mutating it cannot leak into another test.
    """
    return _nowcast_fd_template().copy()


@pytest.mark.skip(reason="Data generation helper")
def produce_fer_minimal_fd() -> ForecastData:
    """Helper function to produce a minimal ForecastData object for testing."""
    outturns = load_fer_outturns(minimal=False)
    forecasts = load_fer_forecasts(minimal=False)

    # select date 2018 and 2019
    outturns = outturns[outturns["date"].dt.year.isin([2018, 2019])]
    forecasts = forecasts[forecasts["date"].dt.year.isin([2018, 2019])]

    # select vintage 2018 and 2019
    outturns = outturns[outturns["vintage_date"].dt.year.isin([2018, 2019])]
    forecasts = forecasts[forecasts["vintage_date"].dt.year.isin([2018, 2019])]

    # select 2 variables
    forecasts = forecasts[forecasts["variable"].isin(["gdpkp", "cpisa"])]
    outturns = outturns[outturns["variable"].isin(["gdpkp", "cpisa"])]

    # select 2 sources
    forecasts = forecasts[forecasts["source"].isin(["mpr", "compass unconditional"])]
    forecasts = forecasts[forecasts["source"].isin(["mpr", "compass unconditional"])]

    # save
    outturns.to_parquet("src/forecast_evaluation/data/files/fer_outturns_minimal.parquet", index=False)
    forecasts.to_parquet("src/forecast_evaluation/data/files/fer_forecasts_minimal.parquet", index=False)
