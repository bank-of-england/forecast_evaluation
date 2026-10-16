"""Outturns that already contain pop/yoy rows must not be duplicated (A24)."""

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from forecast_evaluation.core.transformations import transform_series
from forecast_evaluation.data.DensityForecastData import DensityForecastData
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData
from forecast_evaluation.data.sample_data import (
    create_sample_forecasts,
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
    create_sample_outturns,
)
from forecast_evaluation.tests.accuracy import compute_accuracy_statistics

KEY = ["variable", "frequency", "date", "vintage_date", "metric"]
DIFFER_WARNING = "supplied pop/yoy outturn rows differ"
REMOVED_WARNING = "Removed .* duplicate records with identical values"


def _nowcast_forecasts() -> pd.DataFrame:
    forecasts = create_sample_nowcast_forecasts()
    in_2023 = (forecasts["vintage_date"] >= "2023-01-01") & (forecasts["vintage_date"] < "2024-01-01")
    return forecasts[in_2023].reset_index(drop=True)


@dataclass
class Case:
    cls: type
    outturns: Callable[[], pd.DataFrame]
    forecasts: Callable[[], pd.DataFrame]
    k: int


CASES = {
    "forecast": Case(ForecastData, create_sample_outturns, create_sample_forecasts, k=8),
    "nowcast": Case(NowcastData, create_sample_nowcast_outturns, _nowcast_forecasts, k=0),
    "density": Case(DensityForecastData, create_sample_outturns, create_sample_forecasts, k=8),
}


@pytest.fixture(params=list(CASES))
def case(request) -> Case:
    return CASES[request.param]


@pytest.fixture
def levels(case) -> pd.DataFrame:
    return case.outturns().assign(metric="levels")


@pytest.fixture
def reference(case, levels):
    """Object built from levels-only outturns (holds levels, pop and yoy)."""
    return case.cls(outturns_data=levels)


def _modified(outturns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Copy of ``outturns`` with three pop values shifted (largest shift 0.5)."""
    modified = outturns.copy()
    rows = modified.index[modified["metric"] == "pop"][:3]
    modified.loc[rows, "value"] += [0.5, 0.1, 0.2]
    return modified, modified.loc[rows].set_index(KEY)["value"]


def _differ_warnings(record) -> list:
    return [w for w in record if DIFFER_WARNING in str(w.message)]


def _n_observations(fd, k: int) -> pd.Series:
    cols = ["variable", "unique_id", "metric", "frequency", "horizon"]
    return compute_accuracy_statistics(data=fd, k=k).to_df().set_index(cols)["n_observations"].sort_index()


def _assert_no_duplicates(outturns: pd.DataFrame) -> None:
    assert not outturns.duplicated(subset=KEY).any()


def test_reference_has_derived_rows(reference):
    """Guard: the scenarios below are only meaningful if pop and yoy rows exist."""
    assert {"levels", "pop", "yoy"} <= set(reference.outturns["metric"])


def test_rewrapping_outturns_does_not_duplicate(case, reference):
    """Old code re-derived pop/yoy and concatenated them with the supplied pop/yoy rows."""
    rewrapped = case.cls(outturns_data=reference.outturns)

    assert len(rewrapped.outturns) == len(reference.outturns)
    _assert_no_duplicates(rewrapped.outturns)
    pd.testing.assert_frame_equal(
        rewrapped.outturns.sort_values(KEY).reset_index(drop=True)[KEY + ["value"]],
        reference.outturns.sort_values(KEY).reset_index(drop=True)[KEY + ["value"]],
        check_dtype=False,
    )


def test_levels_only_input_is_unchanged(levels, reference):
    """Regression guard (passes on old code): levels rows first, in input order, then derived rows."""
    outturns = reference.outturns
    n_levels = len(levels)

    assert (outturns["metric"].iloc[:n_levels] == "levels").all()
    np.testing.assert_allclose(outturns["value"].iloc[:n_levels].to_numpy(), levels["value"].to_numpy())
    for metric in ("pop", "yoy"):
        expected = len(transform_series(levels.copy(), transform=metric, frequency="Q"))
        assert (outturns["metric"] == metric).sum() == expected
    assert set(outturns["metric"]) == {"levels", "pop", "yoy"}
    _assert_no_duplicates(outturns)


def test_n_observations_matches_levels_only_case(case, levels, reference):
    """Old code doubled n_observations for the pop and yoy metrics."""
    forecasts = case.forecasts().assign(metric="levels")

    expected_fd = case.cls(outturns_data=levels)
    expected_fd.add_forecasts(forecasts, data_check=False)
    rewrapped_fd = case.cls(outturns_data=reference.outturns)
    rewrapped_fd.add_forecasts(forecasts, data_check=False)

    expected = _n_observations(expected_fd, case.k)
    assert {"pop", "yoy"} & set(expected.index.get_level_values("metric"))
    pd.testing.assert_series_equal(_n_observations(rewrapped_fd, case.k), expected)


def test_matching_values_do_not_warn(case, reference):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        case.cls(outturns_data=reference.outturns)

    assert not _differ_warnings(record)


def test_differing_supplied_value_is_kept_and_warns(case, reference):
    """Old code kept both the supplied and the derived row and never warned."""
    modified, expected_values = _modified(reference.outturns)

    with pytest.warns(UserWarning, match=rf"3 {DIFFER_WARNING}.*largest absolute difference 0\.5\b") as record:
        fd = case.cls(outturns_data=modified)

    assert len(_differ_warnings(record)) == 1
    assert len(fd.outturns) == len(reference.outturns)
    _assert_no_duplicates(fd.outturns)
    kept = fd.outturns.set_index(KEY)["value"].loc[expected_values.index]
    np.testing.assert_allclose(kept.to_numpy(), expected_values.to_numpy())


def test_nowcast_aligned_snapshots_do_not_warn_about_differences():
    """Aligned pop/yoy snapshots may differ from values re-derived from the latest levels; that is not an error."""
    case = CASES["nowcast"]
    levels = case.outturns().assign(metric="levels")
    rewrapped = NowcastData(outturns_data=NowcastData(outturns_data=levels).outturns)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        rewrapped.add_forecasts(case.forecasts().assign(metric="levels"), data_check=False)

    raw = rewrapped._raw_outturns
    assert (raw["_aligned"] & raw["metric"].isin(["pop", "yoy"])).any()
    assert not _differ_warnings(record)


def test_add_outturns_to_empty_object(case, reference):
    fd = case.cls()
    fd.add_outturns(reference.outturns)

    assert len(fd.outturns) == len(reference.outturns)
    _assert_no_duplicates(fd.outturns)


def test_add_outturns_overlapping_existing_levels(case, levels, reference):
    """Old code appended the supplied pop/yoy rows to the pop/yoy rows already derived from levels."""
    fd = case.cls(outturns_data=levels)

    with pytest.warns(UserWarning, match=REMOVED_WARNING):
        fd.add_outturns(reference.outturns)

    assert len(fd.outturns) == len(reference.outturns)
    _assert_no_duplicates(fd.outturns)


def test_add_differing_supplied_rows_to_existing_levels(case, levels, reference):
    """The supplied value replaces the stored derived row and the difference is reported once."""
    modified, expected_values = _modified(reference.outturns)
    supplied = modified[modified["metric"] != "levels"]
    fd = case.cls(outturns_data=levels)

    with pytest.warns(UserWarning, match=rf"3 {DIFFER_WARNING}.*largest absolute difference 0\.5\b") as record:
        fd.add_outturns(supplied)

    assert len(_differ_warnings(record)) == 1
    assert len(fd.outturns) == len(reference.outturns)
    _assert_no_duplicates(fd.outturns)
    kept = fd.outturns.set_index(KEY)["value"].loc[expected_values.index]
    np.testing.assert_allclose(kept.to_numpy(), expected_values.to_numpy())


def test_merge_does_not_duplicate_outturns(case, levels, reference):
    """Merging an object whose raw outturns hold pop/yoy rows into a levels-only object."""
    levels_fd = case.cls(outturns_data=levels)
    rewrapped = case.cls(outturns_data=reference.outturns)

    with pytest.warns(UserWarning, match=REMOVED_WARNING) as record:
        levels_fd.merge(rewrapped)

    assert not _differ_warnings(record)
    assert len(levels_fd.outturns) == len(reference.outturns)
    _assert_no_duplicates(levels_fd.outturns)

    with pytest.warns(UserWarning, match=REMOVED_WARNING):
        rewrapped.merge(case.cls(outturns_data=levels))

    assert len(rewrapped.outturns) == len(reference.outturns)
    _assert_no_duplicates(rewrapped.outturns)


def test_levels_only_merge_still_reports_removed_duplicates(case, levels):
    """The existing 'Removed N duplicate records' behaviour is unchanged for levels-only merges."""
    first = case.cls(outturns_data=levels)
    second = case.cls(outturns_data=levels)
    n_before = len(first.outturns)

    with pytest.warns(UserWarning, match=REMOVED_WARNING):
        first.merge(second)

    assert len(first.outturns) == n_before


def test_outturns_without_vintages_can_be_rewrapped():
    """Null vintage_date keys must match each other, so nothing is dropped or duplicated."""
    levels = create_sample_outturns().drop(columns="vintage_date").assign(metric="levels")
    levels = levels[levels["variable"] == "gdpkp"]
    reference = ForecastData(outturns_data=levels, outturn_vintages=False)
    assert reference.outturns["vintage_date"].isna().all()

    rewrapped = ForecastData(outturns_data=reference.outturns, outturn_vintages=False)

    assert len(rewrapped.outturns) == len(reference.outturns)
    assert {"pop", "yoy"} <= set(rewrapped.outturns["metric"])
    _assert_no_duplicates(rewrapped.outturns)
