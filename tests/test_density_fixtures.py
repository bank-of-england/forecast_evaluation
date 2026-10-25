"""Smoke tests for the shared density fixtures in conftest.py."""

import warnings

import numpy as np
import pandas as pd
import pytest

OBJECT_FIXTURES = {
    # name: (expected density rows, sources, id_columns contain 'region')
    "density_only": (650, {"mpr2"}, False),
    "density_and_point": (650, {"mpr2"}, False),
    "density_two_sources": (1300, {"mpr2", "toy"}, False),
    "density_extra_ids": (650, {"mpr2"}, True),
    "density_two_regions": (1300, {"mpr2"}, True),
}


def _build(request, name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = request.getfixturevalue(name)
    assert not [w for w in caught if "metric" in str(w.message).lower()]
    return obj


@pytest.mark.parametrize("name", OBJECT_FIXTURES)
def test_density_object_fixtures(request, name):
    rows, sources, has_region = OBJECT_FIXTURES[name]
    obj = _build(request, name)
    assert len(obj.density_forecasts) == rows
    assert set(obj.density_forecasts["source"]) == sources
    assert ("region" in obj.id_columns) is has_region
    assert obj.forecasts.empty is (name != "density_and_point")


def test_density_two_regions_ids(density_two_regions):
    assert set(density_two_regions.density_forecasts["region"]) == {"UK", "EW"}


def test_crossed_density_df(density_df, crossed_density_df):
    diff = crossed_density_df["value"] != density_df["value"]
    assert diff.sum() == 2
    assert (crossed_density_df.loc[diff, "date"] == pd.Timestamp("2027-03-31")).all()
    crossed = crossed_density_df[crossed_density_df["date"] == pd.Timestamp("2027-03-31")].sort_values("quantile")
    assert (np.diff(crossed["value"]) < 0).any()


@pytest.mark.parametrize(
    "name, source, grid",
    [
        ("sparse_grid_density_df", "toy_sparse", np.arange(0.1, 0.91, 0.1)),
        ("bvar_grid_density_df", "toy_bvar", [0.16, 0.5, 0.84]),
    ],
)
def test_synthetic_grid_fixtures(request, density_df, name, source, grid):
    df = request.getfixturevalue(name)
    assert list(df.columns) == list(density_df.columns)
    assert len(df) == 13 * len(grid)
    assert set(df["source"]) == {source}
    np.testing.assert_allclose(np.sort(df["quantile"].unique()), grid)
    medians = df[np.isclose(df["quantile"], 0.5)].sort_values("date")["value"]
    np.testing.assert_allclose(medians, 115 + np.arange(13))
