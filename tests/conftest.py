"""Shared pytest fixtures for forecast_evaluation tests."""

import functools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from forecast_evaluation.data.DensityForecastData import DensityForecastData
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.loader import load_fer_forecasts, load_fer_outturns
from forecast_evaluation.data.NowcastData import NowcastData
from forecast_evaluation.data.sample_data import (
    create_sample_density_forecasts,
    create_sample_forecasts,
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
    create_sample_outturns,
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


# -----------------------
# Density fixtures
# -----------------------
def _synthetic_density(grid, source: str) -> pd.DataFrame:
    """Normal quantiles (mean 115 + i, sd 0.5) on ``grid`` for the 13 sample density dates."""
    dates = pd.date_range(start="2025-12-31", periods=13, freq="QE")
    rows = [
        {
            "date": date,
            "variable": "gdpkp",
            "vintage_date": pd.Timestamp("2025-12-31"),
            "source": source,
            "frequency": "Q",
            "quantile": float(q),
            "value": float(norm.ppf(q, loc=115 + i, scale=0.5)),
            "forecast_horizon": i,
            "metric": "levels",
        }
        for i, date in enumerate(dates)
        for q in grid
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def density_outturns_df() -> pd.DataFrame:
    """Sample outturns with an explicit ``metric`` column."""
    return create_sample_outturns().assign(metric="levels")


@pytest.fixture
def density_df() -> pd.DataFrame:
    """Sample density forecasts (source 'mpr2', 13 dates x 50 quantiles) with ``metric``."""
    return create_sample_density_forecasts().assign(metric="levels")


@pytest.fixture
def density_only(density_outturns_df, density_df) -> DensityForecastData:
    """DensityForecastData with density forecasts only (no point forecasts)."""
    return DensityForecastData(outturns_data=density_outturns_df, forecasts_data=density_df)


@pytest.fixture
def density_and_point(density_only) -> DensityForecastData:
    """``density_only`` plus the sample point forecasts (source 'mpr2')."""
    density_only.add_forecasts(create_sample_forecasts().assign(metric="levels"))
    return density_only


@pytest.fixture
def density_two_sources(density_outturns_df, density_df) -> DensityForecastData:
    """Density forecasts from 'mpr2' and a shifted copy (+1) labelled 'toy'."""
    toy = density_df.assign(source="toy", value=density_df["value"] + 1)
    return DensityForecastData(
        outturns_data=density_outturns_df, forecasts_data=pd.concat([density_df, toy], ignore_index=True)
    )


@pytest.fixture
def density_extra_ids(density_outturns_df, density_df) -> DensityForecastData:
    """Density forecasts with a single extra id ``region='UK'``."""
    return DensityForecastData(
        outturns_data=density_outturns_df, forecasts_data=density_df.assign(region="UK"), extra_ids=["region"]
    )


@pytest.fixture
def density_two_regions(density_outturns_df, density_df) -> DensityForecastData:
    """Density forecasts for regions 'UK' and 'EW' (EW values +1), ``extra_ids=['region']``."""
    uk = density_df.assign(region="UK")
    ew = density_df.assign(region="EW", value=density_df["value"] + 1)
    return DensityForecastData(
        outturns_data=density_outturns_df,
        forecasts_data=pd.concat([uk, ew], ignore_index=True),
        extra_ids=["region"],
    )


@pytest.fixture
def crossed_density_df(density_df) -> pd.DataFrame:
    """``density_df`` with the ~0.15 and ~0.83 quantile values swapped at 2027-03-31."""
    on_date = density_df["date"] == pd.Timestamp("2027-03-31")
    lo = on_date & np.isclose(density_df["quantile"], 0.15)
    hi = on_date & np.isclose(density_df["quantile"], 0.83)
    assert lo.sum() == 1 and hi.sum() == 1
    crossed = density_df.copy()
    crossed.loc[lo, "value"] = density_df.loc[hi, "value"].iloc[0]
    crossed.loc[hi, "value"] = density_df.loc[lo, "value"].iloc[0]
    return crossed


@pytest.fixture
def sparse_grid_density_df() -> pd.DataFrame:
    """Synthetic normal densities on a 9-point grid [0.1, ..., 0.9], source 'toy_sparse'."""
    return _synthetic_density(np.round(np.arange(0.1, 0.91, 0.1), 10), "toy_sparse")


@pytest.fixture
def bvar_grid_density_df() -> pd.DataFrame:
    """Synthetic normal densities on the BVAR-style grid [0.16, 0.5, 0.84], source 'toy_bvar'."""
    return _synthetic_density([0.16, 0.5, 0.84], "toy_bvar")


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
