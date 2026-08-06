"""Tests driven by a simulated real-time release calendar.

These cover the *timing* semantics of the revision index ``k``: series published
at different points relative to the period end, revisions, mixed frequencies and
pre-period-end (flash) releases.  Each test here targets a class of bug that
previously slipped through because the synthetic sample data used only
quarter-end-aligned vintage dates.
"""

import pandas as pd
import pytest

from forecast_evaluation.core.main_table import compute_k
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData

from .realtime_dataset import (
    MONTHLY_RELEASE_OFFSETS,
    QUARTERLY_RELEASE_OFFSETS,
    create_realtime_forecasts,
    create_realtime_outturns,
    release_calendar,
)

# Columns that must uniquely identify a row of the main table. `metric` matters:
# monthly series automatically gain a derived 'pop' metric alongside 'levels'.
K_KEY = ["unique_id", "variable", "metric", "date", "vintage_date_forecast", "k"]


@pytest.fixture(scope="module")
def realtime_outturns() -> pd.DataFrame:
    return create_realtime_outturns()


@pytest.fixture(scope="module")
def realtime_forecasts(realtime_outturns) -> pd.DataFrame:
    return create_realtime_forecasts(realtime_outturns)


@pytest.fixture(scope="module")
def quarterly_nowcast_data(realtime_outturns, realtime_forecasts) -> NowcastData:
    """NowcastData built from the quarterly slice of the release calendar."""
    nd = NowcastData(compute_levels=False)
    nd.add_outturns(realtime_outturns[realtime_outturns["frequency"] == "Q"])
    nd.add_forecasts(realtime_forecasts[realtime_forecasts["frequency"] == "Q"], data_check=False)
    return nd


class TestReleaseCalendarFixture:
    """The fixture itself must have the structure the other tests rely on."""

    def test_all_series_present(self, realtime_outturns):
        expected = set(QUARTERLY_RELEASE_OFFSETS) | set(MONTHLY_RELEASE_OFFSETS)
        assert set(realtime_outturns["variable"]) == expected

    def test_contains_a_pre_period_end_release(self, realtime_outturns):
        """q_preflash must publish before the period ends; nothing else should."""
        early = realtime_outturns[realtime_outturns["vintage_date"] < realtime_outturns["date"]]
        assert not early.empty
        assert set(early["variable"]) == {"q_preflash"}

    def test_releases_are_staggered_within_the_quarter(self, realtime_outturns):
        """Each quarterly series lands on a distinct offset from period end."""
        q = realtime_outturns[realtime_outturns["frequency"] == "Q"]
        offsets = q.assign(offset=(q["vintage_date"] - q["date"]).dt.days).groupby("variable")["offset"].apply(set)
        for variable, expected in QUARTERLY_RELEASE_OFFSETS.items():
            assert offsets[variable] == set(expected)

    def test_calendar_has_non_quarter_aligned_dates(self, realtime_outturns):
        """The bug class under test only appears off quarter-end alignment."""
        calendar = release_calendar(realtime_outturns)
        assert any(d.month % 3 != 0 for d in calendar)


class TestNowcastRevisionIndex:
    """NowcastData's dense revision index over a staggered release calendar."""

    def test_pre_release_gets_negative_k(self, quarterly_nowcast_data):
        """Regression: the documented negative-k branch must be reachable."""
        df = quarterly_nowcast_data.df
        pre = df[df["vintage_date_outturn"] < df["date"]]
        assert not pre.empty, "pre-period-end outturns were dropped from the main table"
        assert (pre["k"] < 0).all()

    def test_only_pre_release_rows_are_negative(self, quarterly_nowcast_data):
        df = quarterly_nowcast_data.df
        assert (df.loc[df["k"] < 0, "vintage_date_outturn"] < df.loc[df["k"] < 0, "date"]).all()
        assert set(df.loc[df["k"] < 0, "variable"]) == {"q_preflash"}

    def test_first_post_period_release_is_k_zero(self, quarterly_nowcast_data):
        """k=0 must be the first release on or after the period end."""
        df = quarterly_nowcast_data.df
        post = df[df["vintage_date_outturn"] >= df["date"]]
        first = post.groupby(["unique_id", "variable", "metric", "date"])["vintage_date_outturn"].transform("min")
        assert (post.loc[post["vintage_date_outturn"] == first, "k"] == 0).all()

    def test_k_is_unique_per_forecast_vintage(self, quarterly_nowcast_data):
        """Guarantee stated in _set_revision_index_k's docstring."""
        assert not quarterly_nowcast_data.df.duplicated(subset=K_KEY).any()

    def test_k_ordering_follows_release_order(self, quarterly_nowcast_data):
        """Ranking by k must reproduce ranking by publication date."""
        df = quarterly_nowcast_data.df
        for _, group in df.groupby(["unique_id", "variable", "metric", "date", "vintage_date_forecast"]):
            ordered = group.sort_values("vintage_date_outturn")
            assert ordered["k"].is_monotonic_increasing

    def test_k_matches_number_of_releases(self, quarterly_nowcast_data):
        """Each series' k range reflects its own release schedule."""
        df = quarterly_nowcast_data.df
        for variable, offsets in QUARTERLY_RELEASE_OFFSETS.items():
            n_post = sum(o >= 0 for o in offsets)
            assert df.loc[df["variable"] == variable, "k"].max() == n_post - 1


class TestCalendarKArithmetic:
    """compute_k must place releases in the right calendar bucket."""

    @staticmethod
    def _frame(date: str, vintages: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime([date] * len(vintages)),
                "vintage_date_outturn": pd.to_datetime(vintages),
            }
        )

    def test_quarterly_buckets(self):
        """Pre-quarter-end -> -1, next quarter -> 0, following quarter -> 1.

        Regression for the ``month // 3`` bucketing, which put a March release
        and an April release in the same bucket.
        """
        df = self._frame("2022-03-31", ["2022-03-21", "2022-04-20", "2022-07-05"])
        assert compute_k(df, "Q")["k"].tolist() == [-1, 0, 1]

    def test_quarter_aligned_vintages_unchanged(self):
        """Quarter-end-aligned vintages keep their historical k values."""
        df = self._frame("2022-03-31", ["2022-06-30", "2022-09-30", "2022-12-31"])
        assert compute_k(df, "Q")["k"].tolist() == [0, 1, 2]

    def test_monthly_lags(self):
        """Monthly releases at 1, 2 and 3 month lags map to k = 0, 1, 2."""
        df = self._frame("2022-01-31", ["2022-02-05", "2022-03-07", "2022-04-16"])
        assert compute_k(df, "M")["k"].tolist() == [0, 1, 2]

    def test_releases_before_period_start_are_dropped(self):
        """Anything earlier than the target period is a forecast, not an outturn."""
        df = self._frame("2022-03-31", ["2021-11-15", "2022-03-21"])
        assert compute_k(df, "Q")["k"].tolist() == [-1]


class TestMixedFrequency:
    """Monthly series must not be bucketed with quarterly arithmetic."""

    def test_nowcast_monthly_k_counts_releases(self, realtime_outturns, realtime_forecasts):
        """NowcastData ranks releases directly, so it is frequency-agnostic."""
        nd = NowcastData(compute_levels=False)
        nd.add_outturns(realtime_outturns[realtime_outturns["frequency"] == "M"])
        nd.add_forecasts(realtime_forecasts[realtime_forecasts["frequency"] == "M"], data_check=False)
        df = nd.df

        assert not df.duplicated(subset=K_KEY).any()
        for variable, offsets in MONTHLY_RELEASE_OFFSETS.items():
            assert df.loc[df["variable"] == variable, "k"].max() == len(offsets) - 1

    def test_calendar_k_distinguishes_monthly_releases(self, realtime_outturns, realtime_forecasts):
        """Regression: a monthly instance must use monthly, not quarterly, k arithmetic.

        Unlike NowcastData's dense rank, this ``k`` is a *calendar* index, so its
        value tracks the publication lag in months rather than the release ordinal.
        Under the quarterly arithmetic this used to get, monthly releases fell into
        negative buckets and distinct releases collapsed onto a single k.
        """
        fd = ForecastData(compute_levels=False)
        fd.add_outturns(realtime_outturns[realtime_outturns["frequency"] == "M"])
        fd.add_forecasts(realtime_forecasts[realtime_forecasts["frequency"] == "M"], data_check=False)
        df = fd.df

        for variable, offsets in MONTHLY_RELEASE_OFFSETS.items():
            k = df.loc[df["variable"] == variable, "k"]
            # Every monthly release lands after its period ends, so k is never negative.
            assert k.min() >= 0
            # Distinct releases must stay in distinct calendar buckets.
            assert k.nunique() == len(offsets)
