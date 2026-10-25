"""Test ForecastData.summary() with extra ID columns."""

import numpy as np
import pandas as pd
import pytest

from forecast_evaluation.data.DensityForecastData import (
    DensityForecastData,
    _describe_quantile_grid,
    _describe_quantile_grids,
)
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.sample_data import create_sample_forecasts


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


# -----------------------
# Density summary
# -----------------------
def _summary_text(data, capsys) -> str:
    data.summary()
    return capsys.readouterr().out


def test_summary_density_only_block_content(density_only, capsys):
    out = _summary_text(density_only, capsys)

    assert "[FORECASTS] No point forecasts loaded (density forecasts are loaded: see [DENSITY FORECASTS])" in out
    assert "[FORECASTS] No data loaded" not in out

    lines = out[out.index("[DENSITY FORECASTS]") :].splitlines()
    assert "  Number of variables: 1" in lines
    assert "  Variables: gdpkp" in lines
    assert "  Sources: mpr2" in lines
    assert "  mpr2" in lines
    assert "  Variable | Metric | Freq | Date                     | Vintage                  | Horizon " in lines
    assert "  gdpkp    | levels | Q    | 2025-12-31 to 2028-12-31 | 2025-12-31 to 2025-12-31 | 0 to 12 " in lines
    assert "  " + "-" * 89 in lines
    assert "  Quantile grid:" in lines
    assert "    gdpkp, levels: 0.01-0.99 step 0.02 (n=50); no exact 0.5" in lines


def test_summary_density_and_point_lists_both_blocks(density_and_point, density_outturns_df, capsys):
    out = _summary_text(density_and_point, capsys)

    assert "No point forecasts loaded" not in out
    assert out.index("\n[FORECASTS]\n") < out.index("\n[DENSITY FORECASTS]\n")

    plain = ForecastData(
        outturns_data=density_outturns_df, forecasts_data=create_sample_forecasts().assign(metric="levels")
    )
    plain_out = _summary_text(plain, capsys)
    plain_block = plain_out.split("[FORECASTS]")[1].split("=" * 80)[0]
    point_block = out.split("[FORECASTS]")[1].split("\n[DENSITY FORECASTS]")[0]
    assert point_block == plain_block
    assert "Metric" not in point_block


def test_summary_without_forecasts_unchanged(density_outturns_df, capsys):
    out = _summary_text(DensityForecastData(outturns_data=density_outturns_df), capsys)

    assert "[FORECASTS] No data loaded" in out
    assert "DENSITY FORECASTS" not in out


def test_summary_density_groups_by_source_and_extra_id(density_two_regions, capsys):
    out = _summary_text(density_two_regions, capsys)
    lines = out[out.index("[DENSITY FORECASTS]") :].splitlines()

    assert "  region: EW, UK" in lines
    assert "  mpr2  [region: EW]" in lines
    assert "  mpr2  [region: UK]" in lines
    assert lines.count("  Quantile grid:") == 2


def test_summary_density_reflects_filter(density_two_sources, capsys):
    assert "  Sources: mpr2, toy" in _summary_text(density_two_sources, capsys)

    density_two_sources.filter(sources=["toy"])
    out = _summary_text(density_two_sources, capsys)
    block = out[out.index("[DENSITY FORECASTS]") :]

    assert "  Sources: toy" in block
    assert "mpr2" not in block


def test_summary_density_filtered_to_nothing_is_not_reported_as_unloaded(density_only, capsys):
    density_only.filter(variables=["cpisa"])
    out = _summary_text(density_only, capsys)

    assert "[DENSITY FORECASTS] No data after filtering" in out
    assert "[FORECASTS] No data loaded" not in out


def test_summary_density_na_extra_id_is_listed(density_extra_ids, density_df, capsys):
    density_extra_ids.add_density_forecasts(density_df.assign(source="toy"))
    out = _summary_text(density_extra_ids, capsys)
    block = out[out.index("[DENSITY FORECASTS]") :]

    assert "  Sources: mpr2, toy" in block
    assert "  mpr2  [region: UK]" in block
    assert "  toy  [region: " in block


def test_summary_density_lists_each_grid_per_variable_and_metric(density_only, density_df, capsys):
    odd_tenths = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    sparse = density_df[density_df["quantile"].round(2).isin(odd_tenths)].assign(variable="cpisa", metric="yoy")
    density_only.add_density_forecasts(sparse)
    out = _summary_text(density_only, capsys)

    assert "    cpisa, yoy: 0.05-0.95 step 0.1 (n=10); no exact 0.5" in out
    assert "    gdpkp, levels: 0.01-0.99 step 0.02 (n=50); no exact 0.5" in out
    assert "  cpisa    | yoy    | Q    |" in out
    assert "  gdpkp    | levels | Q    |" in out


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        (np.round(np.arange(0.01, 0.995, 0.02), 10), "0.01-0.99 step 0.02 (n=50)"),
        (np.round(np.arange(0.1, 0.95, 0.1), 10), "0.1-0.9 step 0.1 (n=9)"),
        ([0.05, 0.16, 0.5, 0.84, 0.95], "{0.05, 0.16, 0.5, 0.84, 0.95}"),
        ([0.9500000000000001, 0.05, 0.5, 0.16, 0.84], "{0.05, 0.16, 0.5, 0.84, 0.95}"),
        (
            [0.01, 0.025, 0.05, 0.1, 0.16, 0.25, 0.5, 0.75, 0.84, 0.9, 0.95, 0.99],
            "0.01-0.99, n=12, irregular",
        ),
        ([0.5], "{0.5}"),
        ([0.1, 0.9], "{0.1, 0.9}"),
    ],
)
def test_describe_quantile_grid(levels, expected):
    assert _describe_quantile_grid(levels) == expected


def test_describe_quantile_grids_multiple_grids():
    narrow = (0.05, 0.16, 0.5, 0.84, 0.95)
    other = (0.05, 0.1, 0.5, 0.84, 0.95)

    assert _describe_quantile_grids([narrow, narrow]) == "{0.05, 0.16, 0.5, 0.84, 0.95}"
    assert _describe_quantile_grids([narrow, other]) == "2 distinct grids; common: {0.05, 0.5, 0.84, 0.95}"
    assert (
        _describe_quantile_grids([(0.1, 0.2), (0.3, 0.4)])
        == "2 distinct grids; common: none; 0.5 missing in 2 of 2 grids"
    )
    assert _describe_quantile_grids([(0.25, 0.5, 0.75, 0.95), (0.25, 0.75, 0.95, 0.99)]).endswith(
        "; 0.5 missing in 1 of 2 grids"
    )
