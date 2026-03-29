"""Test ForecastData.summary() with extra ID columns."""

import pandas as pd
import pytest

from forecast_evaluation.data.ForecastData import ForecastData


def make_outturns(variables, dates):
    rows = []
    for var in variables:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "variable": var,
                    "date": date,
                    "vintage_date": date + pd.offsets.MonthEnd(i % 3 + 1),
                    "frequency": "Q",
                    "forecast_horizon": 0,
                    "value": float(i),
                }
            )
    return pd.DataFrame(rows)


def make_forecasts(variables, regions, sources, dates):
    rows = []
    for var in variables:
        for region in regions:
            for source in sources:
                max_h = 8 if source == "short_source" else 13
                for date in dates:
                    for h in range(1, max_h + 1):
                        rows.append(
                            {
                                "variable": var,
                                "date": date,
                                "vintage_date": date - pd.offsets.QuarterEnd(1),
                                "frequency": "Q",
                                "forecast_horizon": h,
                                "source": source,
                                "value": float(h),
                                "region": region,
                            }
                        )
    return pd.DataFrame(rows)


@pytest.fixture
def fd_with_extra_ids():
    variables = ["gdp", "inflation_long_name"]
    regions = ["UK", "EU"]
    sources = ["short_source", "a_longer_source_name"]
    dates = pd.date_range("2015-03-31", periods=6, freq="QE").tolist()

    outturns = make_outturns(variables, dates)
    forecasts = make_forecasts(variables, regions, sources, dates)

    return ForecastData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        extra_ids=["region"],
        compute_levels=False,
    )


@pytest.fixture
def fd_no_extra_ids():
    variables = ["gdp", "inflation_long_name"]
    regions = ["UK", "EU"]
    sources = ["short_source", "a_longer_source_name"]
    dates = pd.date_range("2015-03-31", periods=6, freq="QE").tolist()

    outturns = make_outturns(variables, dates)
    forecasts = make_forecasts(variables, regions, sources, dates)

    return ForecastData(
        outturns_data=outturns,
        forecasts_data=forecasts,
        compute_levels=False,
    )


def test_summary_with_extra_ids_snapshot(fd_with_extra_ids, snapshot, capsys):
    fd_with_extra_ids.summary()
    captured = capsys.readouterr()
    assert captured.out == snapshot


def test_summary_with_no_extra_ids_snapshot(fd_no_extra_ids, snapshot, capsys):
    fd_no_extra_ids.summary()
    captured = capsys.readouterr()
    assert captured.out == snapshot
