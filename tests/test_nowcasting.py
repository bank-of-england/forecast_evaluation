import pandas as pd
import pytest

import forecast_evaluation as fe
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.NowcastData import NowcastData
from forecast_evaluation.data.sample_data import (
    create_sample_forecasts,
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
    create_sample_outturns,
)
from forecast_evaluation.data.utils import compute_days_in_period
from forecast_evaluation.tests.intra_period import (
    compute_intra_period_accuracy,
    compute_intra_period_bias,
)
from forecast_evaluation.visualisations.forecast import plot_nowcasts
from forecast_evaluation.visualisations.intra_period import (
    _z_multiplier,
    plot_intra_period_accuracy,
    plot_intra_period_bias,
)


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def nowcast_outturns() -> pd.DataFrame:
    return create_sample_nowcast_outturns()


@pytest.fixture
def nowcast_forecasts() -> pd.DataFrame:
    return create_sample_nowcast_forecasts()


def _with_forecast_horizon(forecasts: pd.DataFrame, horizon: int = 0) -> pd.DataFrame:
    return forecasts.assign(forecast_horizon=horizon)


# -----------------------
# compute_days_in_period Tests
# -----------------------
class TestComputeDaysInPeriod:
    """Tests for the compute_days_in_period helper function."""

    def test_quarterly_boundary_values(self):
        """First day = 0, mid-quarter correct, last day = 90."""
        dates = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-14"), pd.Timestamp("2024-03-31")])
        freqs = pd.Series(["Q", "Q", "Q"])
        result = compute_days_in_period(dates, freqs)
        assert result.iloc[0] == 0
        assert result.iloc[1] == 44
        assert result.iloc[2] == 90

    def test_monthly(self):
        """Feb 14 is day 13 of February."""
        dates = pd.Series([pd.Timestamp("2024-02-14")])
        freqs = pd.Series(["M"])
        result = compute_days_in_period(dates, freqs)
        assert result.iloc[0] == 13

    def test_quarterly_full_year_range(self):
        """All quarterly values should be between 0 and 91."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        freqs = pd.Series(["Q"] * len(dates))
        result = compute_days_in_period(dates.to_series().reset_index(drop=True), freqs)
        assert result.min() >= 0
        assert result.max() <= 91


# -----------------------
# days_in_period in ForecastData
# -----------------------
class TestDaysInPeriod:
    """Test that days_in_period is no longer injected into extra_ids for nowcasting data."""

    def test_days_in_period_not_in_unique_id(self, nowcast_outturns, nowcast_forecasts):
        """With NowcastData, unique_id should just be 'source' (no days_in_period)."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert fd.id_columns == ["source"]
        # No '+' separator with days_in_period in unique_id
        assert all("+" not in uid for uid in fd._raw_forecasts["unique_id"].unique())

    def test_days_in_period_not_added_without_nowcasting(self):
        """Standard forecasts (nowcasting=False) should NOT get days_in_period."""
        fd = ForecastData(outturns_data=create_sample_outturns(), forecasts_data=create_sample_forecasts())
        assert "days_in_period" not in fd._raw_forecasts.columns

    def test_sample_data_does_not_have_days_in_period(self, nowcast_forecasts):
        """The sample nowcast data no longer includes days_in_period (it is computed on ingestion)."""
        assert "days_in_period" not in nowcast_forecasts.columns


# -----------------------
# Core Nowcasting Flow Tests
# -----------------------
class TestNowcastingFlow:
    """Test the core add/filter/evaluate flow with nowcasting data."""

    def test_uses_intra_period_vintages(self, nowcast_outturns):
        """Only nowcast data reports intra-period forecast vintages."""
        assert not ForecastData().uses_intra_period_vintages
        assert NowcastData(outturns_data=nowcast_outturns).uses_intra_period_vintages

    def test_supports_outturn_revision_analysis_tracks_outturn_vintages(self, nowcast_outturns):
        """Outturn revision support depends on outturn vintages, not on the class."""
        assert ForecastData().supports_outturn_revision_analysis
        assert not ForecastData(outturn_vintages=False).supports_outturn_revision_analysis
        assert NowcastData(outturns_data=nowcast_outturns).supports_outturn_revision_analysis

    def test_nowcast_data_properties(self, nowcast_outturns, nowcast_forecasts):
        """Nowcast data should preserve row count, vintages, sources, variables, and horizons."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        # Row count preserved
        assert len(fd._raw_forecasts) == len(nowcast_forecasts)

        # Vintage dates not snapped to quarter-end
        vintages = fd._raw_forecasts["vintage_date"]
        has_non_quarter_end = (vintages.dt.month % 3 != 0) | (vintages != vintages + pd.offsets.MonthEnd(0))
        assert has_non_quarter_end.any()

        # Target dates snapped to quarter-end
        dates = fd._raw_forecasts["date"]
        pd.testing.assert_series_equal(dates, dates + pd.offsets.QuarterEnd(0), check_names=False)

        # Both models and variables present
        assert set(fd._raw_forecasts["source"].unique()) == {"nowcast_dfm", "nowcast_bridge"}
        assert set(fd._raw_forecasts["variable"].unique()) == {"gdp", "cpi"}

        # forecast_horizon is the information horizon (0 nowcast, 1 one-quarter-ahead)
        fh = fd._raw_forecasts["forecast_horizon"]
        assert pd.api.types.is_integer_dtype(fh)
        assert (fh >= 0).all()
        # Horizons are small integers, not days.
        assert fh.max() <= 10
        assert (fd._raw_forecasts["target_minus_vintage"] == -1).any()
        # Multiple weekly vintages per (source, date, horizon) group
        assert len(fd._raw_forecasts) > len(fh.unique()) * 2

        # k in main table is revision index (0, 1, 2, ...), not calendar-quarter distance
        if not fd.df.empty:
            assert fd.df["k"].min() >= -1

    def test_outturns_omit_forecast_horizon(self, nowcast_outturns, nowcast_forecasts):
        """Outturns do not carry the forecast-only information horizon."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert "forecast_horizon" not in fd._raw_outturns.columns
        assert fd._raw_forecasts["forecast_horizon"].equals(nowcast_forecasts["forecast_horizon"])

    def test_mixed_weekly_and_quarterly_vintages(self, nowcast_outturns, nowcast_forecasts):
        """Can add both quarterly-vintage and weekly-vintage forecasts."""
        outturns = pd.concat([nowcast_outturns, create_sample_outturns()], ignore_index=True)
        outturns = outturns.drop_duplicates(subset=[c for c in outturns.columns if c != "value"])

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(create_sample_forecasts(), data_check=False)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        sources = set(fd._raw_forecasts["source"].unique())
        assert "mpr2" in sources
        assert "nowcast_dfm" in sources

    def test_k_is_revision_index_unique_outturn_per_date_and_k(self, nowcast_outturns, nowcast_forecasts):
        """For nowcasting, each (variable, metric, frequency, date, k) should map to one outturn vintage."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        mt = fd.df.copy()
        assert not mt.empty

        grouped = (
            mt.groupby(["variable", "metric", "frequency", "date", "k"])["vintage_date_outturn"]
            .nunique()
            .reset_index(name="n_vintages")
        )

        assert (grouped["n_vintages"] <= 1).all()

    def test_filter_by_source(self, nowcast_outturns, nowcast_forecasts):
        """Filtering by source should work with nowcast data."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        fd.filter(sources="nowcast_dfm")
        assert set(fd.forecasts["source"].unique()) == {"nowcast_dfm"}

    def test_filter_by_variable(self, nowcast_outturns, nowcast_forecasts):
        """Filtering by variable should work with nowcast data."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        fd.filter(variables="gdp")
        assert set(fd.forecasts["variable"].unique()) == {"gdp"}

    def test_clear_filter_preserves_nowcast_k_and_days_to_publication(self, nowcast_outturns, nowcast_forecasts):
        """clear_filter() should restore the nowcast-specific k index and days_to_publication column."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert "days_to_publication" in fd.df.columns
        original = fd.df.sort_values(list(fd.df.columns)).reset_index(drop=True)

        # Apply a filter, then clear it
        fd.filter(variables="gdp")
        fd.clear_filter()

        assert "days_to_publication" in fd.df.columns
        # k should remain a dense revision index (0, 1, 2, ...), not calendar-quarter distance
        assert fd.df["k"].min() >= -1

        restored = fd.df.sort_values(list(fd.df.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(original, restored)


# -----------------------
# `_aligned` column regression tests
# -----------------------
class TestAlignedColumnStaysNonNull:
    """Regression tests: outturns -> forecasts -> more outturns -> more forecasts.

    Additional outturns added after alignment must not introduce NaNs into the
    internal ``_aligned`` marker column, otherwise ``~outturns["_aligned"]``
    expressions used for filtering (main table, outturn revisions) break.
    """

    def test_aligned_column_has_no_nulls_after_more_outturns_and_forecasts(self):
        outturns_1 = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-03-31", "2020-06-30"]),
                "variable": ["gdp", "gdp"],
                "vintage_date": pd.to_datetime(["2020-04-30", "2020-04-30"]),
                "frequency": ["Q", "Q"],
                "value": [100.0, 101.0],
            }
        )

        forecasts_1 = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-06-30", "2020-09-30"]),
                "variable": ["gdp", "gdp"],
                "vintage_date": pd.to_datetime(["2020-07-15", "2020-07-15"]),
                "source": ["modelA", "modelA"],
                "frequency": ["Q", "Q"],
                "value": [101.5, 102.0],
            }
        )

        fd = NowcastData(outturns_data=outturns_1)
        # First round of forecasts triggers outturn-vintage alignment (`_aligned` rows added).
        fd.add_forecasts(_with_forecast_horizon(forecasts_1), data_check=False)

        assert "_aligned" in fd._raw_outturns.columns
        assert fd._raw_outturns["_aligned"].isna().sum() == 0

        # A genuine new outturn release, added via the plain (non-aligning) code path.
        outturns_2 = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-06-30"]),
                "variable": ["gdp"],
                "vintage_date": pd.to_datetime(["2020-07-31"]),
                "frequency": ["Q"],
                "value": [101.2],
            }
        )
        fd.add_outturns(outturns_2)

        # No NaNs should be introduced by concatenating rows without `_aligned`.
        assert fd._raw_outturns["_aligned"].isna().sum() == 0
        assert fd._raw_outturns["_aligned"].dtype == bool
        assert fd._outturns["_aligned"].isna().sum() == 0
        assert fd._outturns["_aligned"].dtype == bool

        # `~outturns["_aligned"]` style expressions must not raise or misbehave.
        not_aligned = ~fd._raw_outturns["_aligned"]
        assert not_aligned.notna().all()

        # More forecasts at a new vintage triggers alignment again; should stay well-formed.
        forecasts_2 = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-09-30"]),
                "variable": ["gdp"],
                "vintage_date": pd.to_datetime(["2020-10-15"]),
                "source": ["modelA"],
                "frequency": ["Q"],
                "value": [102.3],
            }
        )
        fd.add_forecasts(_with_forecast_horizon(forecasts_2), data_check=False)

        assert fd._raw_outturns["_aligned"].isna().sum() == 0
        assert fd._raw_outturns["_aligned"].dtype == bool
        assert fd._outturns["_aligned"].isna().sum() == 0

        # Downstream filtering (main table / outturn revisions) should not crash.
        from forecast_evaluation.core.outturns_revisions_table import create_outturn_revisions

        revisions = create_outturn_revisions(fd)
        assert "_aligned" not in revisions.columns


# -----------------------
# merge() regression tests
# -----------------------
class TestMergeExcludesSyntheticOutturns:
    """`merge()` must not turn another instance's synthetic snapshots into genuine releases.

    The parent implementation passes ``other._raw_outturns`` straight through
    ``add_outturns``; strict schema filtering drops the ``_aligned`` marker, so
    forward-filled snapshots become indistinguishable from official releases.
    """

    @staticmethod
    def _build(source: str, outturn_vintage: str, forecast_vintage: str) -> NowcastData:
        outturns = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-03-31", "2020-06-30"]),
                "variable": ["gdp", "gdp"],
                "vintage_date": pd.to_datetime([outturn_vintage, outturn_vintage]),
                "frequency": ["Q", "Q"],
                "value": [100.0, 101.0],
            }
        )
        forecasts = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-06-30"]),
                "variable": ["gdp"],
                "vintage_date": pd.to_datetime([forecast_vintage]),
                "source": [source],
                "frequency": ["Q"],
                "value": [101.5],
            }
        )
        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False)
        return fd

    def test_merge_keeps_other_snapshots_marked_as_aligned(self):
        fd_a = self._build("modelA", outturn_vintage="2020-07-31", forecast_vintage="2020-08-15")
        fd_b = self._build("modelB", outturn_vintage="2020-08-31", forecast_vintage="2020-09-15")

        # `fd_b` holds a synthetic snapshot at its own forecast vintage.
        b_snapshots = fd_b._raw_outturns.loc[fd_b._raw_outturns["_aligned"], "vintage_date"].unique()
        assert pd.Timestamp("2020-09-15") in set(b_snapshots)

        fd_a.merge(fd_b)

        outturns = fd_a._raw_outturns
        genuine_vintages = set(outturns.loc[~outturns["_aligned"], "vintage_date"].unique())

        # Only real releases from both instances may count as genuine.
        assert genuine_vintages == {pd.Timestamp("2020-07-31"), pd.Timestamp("2020-08-31")}

        # `fd_b`'s snapshot vintage is rebuilt locally and must stay flagged synthetic.
        snapshot_rows = outturns[outturns["vintage_date"] == pd.Timestamp("2020-09-15")]
        assert not snapshot_rows.empty
        assert snapshot_rows["_aligned"].all()

        # Merged forecasts from both sources are present.
        assert set(fd_a.df["source"].unique()) == {"modelA", "modelB"}

    def test_merge_does_not_inflate_outturn_revisions(self):
        fd_a = self._build("modelA", outturn_vintage="2020-07-31", forecast_vintage="2020-08-15")
        fd_b = self._build("modelB", outturn_vintage="2020-08-31", forecast_vintage="2020-09-15")

        fd_a.merge(fd_b)

        from forecast_evaluation.core.outturns_revisions_table import create_outturn_revisions

        revisions = create_outturn_revisions(fd_a)
        if not revisions.empty:
            assert set(revisions["vintage_date_outturn"].unique()) <= {
                pd.Timestamp("2020-07-31"),
                pd.Timestamp("2020-08-31"),
            }


# -----------------------
# Multi-metric alignment regression tests
# -----------------------
def _multi_metric_outturns() -> pd.DataFrame:
    """One variable ('gdp') with two metrics on *different* vintage calendars.

    - levels: released 2020-04-30 (Q1) and 2020-07-15 (Q1+Q2)
    - pop:    released 2020-04-30 (Q1) only -- no 2020-07-15 vintage
    """
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-03-31", "2020-03-31", "2020-06-30", "2020-03-31"],
            ),
            "variable": ["gdp"] * 4,
            "metric": ["levels", "levels", "levels", "pop"],
            "vintage_date": pd.to_datetime(
                ["2020-04-30", "2020-07-15", "2020-07-15", "2020-04-30"],
            ),
            "frequency": ["Q"] * 4,
            "value": [100.0, 100.2, 101.0, 1.0],
        }
    )


class TestAlignmentIsMetricAware:
    """``_align_outturn_vintages`` must key series by ``(variable, metric)``.

    Grouping by ``variable`` alone lets a vintage that only exists for one
    metric be treated as existing for every metric, and lets the
    latest-release lookup (one row per ``date``) drop or swap rows across
    metrics.
    """

    def test_vintage_of_one_metric_does_not_mask_another(self):
        """2020-07-15 exists for levels but not pop; pop must still be aligned."""
        forecasts = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-09-30", "2020-09-30"]),
                "variable": ["gdp", "gdp"],
                "metric": ["levels", "pop"],
                "vintage_date": pd.to_datetime(["2020-07-15", "2020-07-15"]),
                "source": ["modelA", "modelA"],
                "frequency": ["Q", "Q"],
                "value": [102.0, 1.2],
            }
        )

        fd = NowcastData(outturns_data=_multi_metric_outturns())
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        aligned_pop = raw[
            (raw["variable"] == "gdp") & (raw["metric"] == "pop") & (raw["vintage_date"] == pd.Timestamp("2020-07-15"))
        ]
        # Without metric-aware grouping the 'levels' 2020-07-15 vintage makes
        # this vintage look like it already exists for 'pop', so no snapshot
        # is built and this is empty.
        assert not aligned_pop.empty
        assert aligned_pop["_aligned"].all()

    def test_snapshot_keeps_every_metric_for_a_date(self):
        """The latest-release lookup must not collapse metrics sharing a date."""
        forecasts = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-09-30", "2020-09-30"]),
                "variable": ["gdp", "gdp"],
                "metric": ["levels", "pop"],
                "vintage_date": pd.to_datetime(["2020-07-20", "2020-07-20"]),
                "source": ["modelA", "modelA"],
                "frequency": ["Q", "Q"],
                "value": [102.0, 1.2],
            }
        )

        fd = NowcastData(outturns_data=_multi_metric_outturns())
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        snapshot = raw[raw["vintage_date"] == pd.Timestamp("2020-07-20")]

        # Both metrics must appear in the snapshot. Grouping by 'date' alone
        # keeps only one row per date, so the 'pop' observation for 2020-03-31
        # is silently replaced by the 'levels' one.
        assert set(snapshot["metric"].unique()) == {"levels", "pop"}

        # And the values must come from the correct series, not be swapped.
        pop_q1 = snapshot[(snapshot["metric"] == "pop") & (snapshot["date"] == pd.Timestamp("2020-03-31"))]
        assert pop_q1["value"].tolist() == [1.0]

    def test_every_metric_of_a_forecast_variable_is_aligned(self):
        """Eligibility is decided by ``variable``, not ``(variable, metric)``.

        With ``compute_levels=True`` a 'pop' forecast is converted to levels
        using the *levels* outturn history, so the levels series must be
        aligned even though no forecast row carries ``metric='levels'``.
        Gating on ``(variable, metric)`` skips it and the transformation
        silently degrades to "no outturn data available for this vintage".
        """
        outturns = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-03-31", "2020-06-30"] * 2),
                "variable": ["gdp"] * 4,
                "metric": ["levels", "levels", "pop", "pop"],
                "vintage_date": pd.to_datetime(["2020-04-30", "2020-07-15"] * 2),
                "frequency": ["Q"] * 4,
                "value": [100.0, 101.0, 1.0, 1.1],
            }
        )
        forecasts = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-09-30"]),
                "variable": ["gdp"],
                "metric": ["pop"],
                "vintage_date": pd.to_datetime(["2020-07-20"]),
                "source": ["modelA"],
                "frequency": ["Q"],
                "value": [1.2],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        snapshot = raw[raw["vintage_date"] == pd.Timestamp("2020-07-20")]
        assert set(snapshot["metric"].unique()) == {"levels", "pop"}

    def test_levels_history_available_for_pop_forecasts(self):
        """End-to-end: a pop-only forecast set must still reach the main table.

        Exercises the ``compute_levels=True`` path that the gate broke.
        """
        dates = pd.date_range("2018-03-31", "2020-09-30", freq="QE")
        outturns = pd.DataFrame(
            {
                "date": dates,
                "variable": "gdp",
                "metric": "levels",
                "vintage_date": dates + pd.Timedelta(days=30),
                "frequency": "Q",
                "value": [100.0 + i for i in range(len(dates))],
            }
        )
        forecasts = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-09-30"] * 3),
                "variable": "gdp",
                "metric": "pop",
                "vintage_date": pd.to_datetime(["2020-07-06", "2020-07-13", "2020-07-20"]),
                "source": "modelA",
                "frequency": "Q",
                "value": [0.012, 0.013, 0.014],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False)

        assert fd._raw_outturns["_aligned"].any()
        assert not fd.df.empty
        assert set(fd.df["vintage_date_forecast"]) == set(forecasts["vintage_date"])


# -----------------------
# _align_outturn_vintages history-bounding tests
# -----------------------
def _long_history_outturns(n_quarters: int = 21) -> pd.DataFrame:
    """Quarterly 'gdp' outturns from 2015Q1, all released on a single vintage.

    Values are ``100 + i`` so YoY/level expectations can be computed by hand.
    """
    dates = pd.date_range("2015-03-31", periods=n_quarters, freq="QE")
    return pd.DataFrame(
        {
            "date": dates,
            "variable": "gdp",
            "vintage_date": pd.Timestamp("2020-02-01"),
            "frequency": "Q",
            "value": [100.0 + i for i in range(n_quarters)],
        }
    )


class TestAlignOutturnVintagesHistoryBound:
    """``_align_outturn_vintages`` should bound synthetic snapshots to the window
    actually consumed downstream instead of keeping the entire historical series.
    """

    def test_align_outturn_vintages_trims_old_history(self):
        """Dates older than the computed cutoff must be dropped from the snapshot."""
        outturns = _long_history_outturns()
        forecast_vintage = pd.Timestamp("2020-04-15")
        forecasts = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-03-31")],
                "variable": ["gdp"],
                "vintage_date": [forecast_vintage],
                "source": ["modelA"],
                "frequency": ["Q"],
                "value": [121.5],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        aligned = raw[(raw["_aligned"]) & (raw["vintage_date"] == forecast_vintage)]
        assert not aligned.empty

        assert aligned["date"].min() == pd.Timestamp("2018-12-31")
        assert (aligned["date"] < pd.Timestamp("2018-12-31")).sum() == 0

    def test_align_outturn_vintages_trim_preserves_yoy_computation(self):
        """Trimming must still retain enough history for a valid YoY value at the forecast."""
        outturns = _long_history_outturns()
        forecast_vintage = pd.Timestamp("2020-04-15")
        forecast_value = 121.5
        forecasts = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-03-31")],
                "variable": ["gdp"],
                "vintage_date": [forecast_vintage],
                "source": ["modelA"],
                "frequency": ["Q"],
                "value": [forecast_value],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=True)

        mt = fd.df
        yoy_rows = mt[
            (mt["metric"] == "yoy")
            & (mt["vintage_date_forecast"] == forecast_vintage)
            & (mt["date"] == pd.Timestamp("2020-03-31"))
        ]
        assert not yoy_rows.empty
        assert yoy_rows["value_forecast"].notna().all()

        # 2019-03-31 outturn ('gdp', 100 + 16 = 116.0) is the YoY base.
        expected_yoy = (forecast_value - 116.0) / 116.0
        assert yoy_rows["value_forecast"].iloc[0] == pytest.approx(expected_yoy)

    def test_align_outturn_vintages_widens_window_for_deeper_horizons(self):
        """A deeper forecast horizon must widen the retained window.

        A forecast targeting 2018Q4 from a 2020Q2 vintage sits at horizon -6, so
        the year-on-year base needs history down to horizon -11 (2017Q3).
        """
        outturns = _long_history_outturns()
        forecast_vintage = pd.Timestamp("2020-04-15")
        forecasts = pd.DataFrame(
            {
                "date": [pd.Timestamp("2018-12-31")],
                "variable": ["gdp"],
                "vintage_date": [forecast_vintage],
                "source": ["modelA"],
                "frequency": ["Q"],
                "value": [115.5],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        aligned = raw[(raw["_aligned"]) & (raw["vintage_date"] == forecast_vintage)]
        assert not aligned.empty
        assert aligned["date"].min() == pd.Timestamp("2017-09-30")

    def test_align_outturn_vintages_bounds_each_variable_separately(self):
        """Variables with different deepest horizons get different retained windows."""
        gdp = _long_history_outturns()
        cpi = _long_history_outturns().assign(variable="cpi")
        outturns = pd.concat([gdp, cpi], ignore_index=True)

        forecast_vintage = pd.Timestamp("2020-04-15")
        forecasts = pd.DataFrame(
            {
                # gdp is a backcast at horizon -1; cpi reaches back to horizon -6.
                "date": [pd.Timestamp("2020-03-31"), pd.Timestamp("2018-12-31")],
                "variable": ["gdp", "cpi"],
                "vintage_date": [forecast_vintage] * 2,
                "source": ["modelA"] * 2,
                "frequency": ["Q"] * 2,
                "value": [121.5, 115.5],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        aligned = raw[(raw["_aligned"]) & (raw["vintage_date"] == forecast_vintage)]

        # gdp: horizon -1 -> cutoff -6 (2018Q4); cpi: horizon -6 -> cutoff -11 (2017Q3).
        assert aligned[aligned["variable"] == "gdp"]["date"].min() == pd.Timestamp("2018-12-31")
        assert aligned[aligned["variable"] == "cpi"]["date"].min() == pd.Timestamp("2017-09-30")

    def test_align_outturn_vintages_bound_is_per_vintage(self):
        """Each vintage's window is anchored to that vintage, not to the earliest one."""
        outturns = _long_history_outturns()
        first_vintage = pd.Timestamp("2020-04-15")
        second_vintage = pd.Timestamp("2020-07-15")
        forecasts = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")],
                "variable": ["gdp"] * 2,
                "vintage_date": [first_vintage, second_vintage],
                "source": ["modelA"] * 2,
                "frequency": ["Q"] * 2,
                "value": [121.5, 122.5],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        aligned = raw[raw["_aligned"]]

        # Both forecasts sit at horizon -1, so each window starts 6 quarters back
        # from its own vintage.
        assert aligned[aligned["vintage_date"] == first_vintage]["date"].min() == pd.Timestamp("2018-12-31")
        assert aligned[aligned["vintage_date"] == second_vintage]["date"].min() == pd.Timestamp("2019-03-31")

    def test_align_outturn_vintages_deepens_existing_snapshot(self):
        """A later, deeper forecast at an existing vintage must deepen its snapshot."""
        outturns = _long_history_outturns()
        forecast_vintage = pd.Timestamp("2020-04-15")
        shallow = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-03-31")],
                "variable": ["gdp"],
                "vintage_date": [forecast_vintage],
                "source": ["modelA"],
                "frequency": ["Q"],
                "value": [121.5],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(shallow), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        aligned = raw[(raw["_aligned"]) & (raw["vintage_date"] == forecast_vintage)]
        assert aligned["date"].min() == pd.Timestamp("2018-12-31")

        # Same vintage, but reaching back to horizon -6 -> window must widen to 2017Q3.
        deep = pd.DataFrame(
            {
                "date": [pd.Timestamp("2018-12-31")],
                "variable": ["gdp"],
                "vintage_date": [forecast_vintage],
                "source": ["modelB"],
                "frequency": ["Q"],
                "value": [115.5],
            }
        )
        fd.add_forecasts(_with_forecast_horizon(deep), data_check=False, compute_levels=False)

        raw = fd._raw_outturns
        aligned = raw[(raw["_aligned"]) & (raw["vintage_date"] == forecast_vintage)]
        assert aligned["date"].min() == pd.Timestamp("2017-09-30")

    def test_align_outturn_vintages_rebuild_does_not_duplicate(self):
        """Rebuilding on each call must not accumulate duplicate snapshot rows."""
        outturns = _long_history_outturns()
        forecasts = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-03-31")],
                "variable": ["gdp"],
                "vintage_date": [pd.Timestamp("2020-04-15")],
                "source": ["modelA"],
                "frequency": ["Q"],
                "value": [121.5],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)
        after_first = len(fd._raw_outturns)

        fd.add_forecasts(
            _with_forecast_horizon(forecasts.assign(source="modelB", value=121.0)),
            data_check=False,
            compute_levels=False,
        )

        aligned = fd._raw_outturns[fd._raw_outturns["_aligned"]]
        assert not aligned.duplicated(subset=["variable", "metric", "vintage_date", "date"]).any()
        # Same vintage and depth as the first call, so the row count is unchanged.
        assert len(fd._raw_outturns) == after_first


class TestAddForecastsTransactional:
    def test_invalid_forecasts_do_not_mutate_outturns(self):
        """Rejected forecasts must leave raw and prepared outturns unchanged."""
        outturns = _long_history_outturns()
        forecasts = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-03-31")],
                "variable": ["gdp"],
                "vintage_date": [pd.Timestamp("2020-04-15")],
                "source": ["modelA"],
                "frequency": ["Q"],
                "value": [121.5],
            }
        )

        fd = NowcastData(outturns_data=outturns)
        fd.add_forecasts(_with_forecast_horizon(forecasts), data_check=False, compute_levels=False)
        raw_outturns_before = fd._raw_outturns.copy(deep=True)
        outturns_before = fd._outturns.copy(deep=True)

        invalid_forecasts = forecasts.assign(
            vintage_date=pd.Timestamp("2020-07-15"),
            metric="invalid",
        )
        with pytest.raises(ValueError, match="Invalid metric values found"):
            fd.add_forecasts(invalid_forecasts, data_check=False, compute_levels=False)

        pd.testing.assert_frame_equal(fd._raw_outturns, raw_outturns_before)
        pd.testing.assert_frame_equal(fd._outturns, outturns_before)


# -----------------------
# Intra-Period Visualisation Tests
# -----------------------
class TestIntraPeriodPlot:
    """Tests for the intra-period accuracy visualisation."""

    @pytest.mark.parametrize(
        ("frequency", "expected_boundaries"),
        [
            ("M", [91, 60, 31, 0]),
            ("Q", [91, 0]),
        ],
    )
    def test_period_boundaries_follow_calendar(self, frequency, expected_boundaries):
        """The public plotting API uses actual calendar period ends as markers."""
        import matplotlib

        matplotlib.use("Agg")
        target_date = pd.Timestamp("2024-03-31")
        data = pd.DataFrame(
            {
                "date": [target_date] * 4,
                "vintage_date_forecast": target_date - pd.to_timedelta([0, 31, 60, 91], unit="D"),
                "vintage_date_outturn": [target_date] * 4,
                "variable": ["gdp"] * 4,
                "metric": ["levels"] * 4,
                "frequency": [frequency] * 4,
                "forecast_horizon": [0] * 4,
                "forecast_error": [0.1, 0.2, 0.3, 0.4],
                "source": ["model"] * 4,
                "k": [0] * 4,
                "latest_vintage": [target_date] * 4,
            }
        )

        fig, ax = fe.plot_intra_period_accuracy(data, variable="gdp", frequency=frequency, return_plot=True)

        boundary_lines = ax.lines[1:]
        assert sorted(line.get_xdata()[0] for line in boundary_lines) == sorted(expected_boundaries)
        assert boundary_lines[0].get_label() == f"{'Month' if frequency == 'M' else 'Quarter'} boundary"
        matplotlib.pyplot.close(fig)

    @pytest.mark.parametrize("confidence_level", [0, 100, -1, 101])
    def test_invalid_confidence_level_raises(self, confidence_level):
        with pytest.raises(ValueError, match="confidence_level must be greater than 0 and less than 100"):
            _z_multiplier(confidence_level)

    def test_plot_rmse_and_mae(self, nowcast_outturns, nowcast_forecasts):
        """plot_intra_period_accuracy should return (fig, ax) for both RMSE and MAE."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        import matplotlib

        matplotlib.use("Agg")

        for stat in ("rmse", "mae"):
            fig, ax = plot_intra_period_accuracy(
                fd, variable="gdp", metric="levels", frequency="Q", statistic=stat, return_plot=True
            )
            assert fig is not None
            assert ax is not None

    def test_plot_missing_vintage_columns_raises(self):
        """Should raise ValueError if vintage date columns are missing."""
        df = pd.DataFrame(
            {
                "variable": ["gdp"],
                "metric": ["levels"],
                "frequency": ["Q"],
                "forecast_horizon": [0],
                "forecast_error": [0.5],
                "source": ["model"],
            }
        )
        with pytest.raises(ValueError, match="vintage_date_forecast"):
            plot_intra_period_accuracy(df, variable="gdp", return_plot=True)

    def test_plot_raises_when_not_nowcast_data(self, nowcast_outturns, nowcast_forecasts):
        """Should raise ValueError when using ForecastData instead of NowcastData."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        with pytest.raises(ValueError, match="NowcastData"):
            plot_intra_period_accuracy(fd, variable="gdp", return_plot=True)

    def test_plot_no_data_raises(self, nowcast_outturns, nowcast_forecasts):
        """Should raise ValueError when no data matches the filters."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        import matplotlib

        matplotlib.use("Agg")

        with pytest.raises(ValueError, match="No data"):
            plot_intra_period_accuracy(fd, variable="nonexistent", metric="levels", frequency="Q", return_plot=True)


# -----------------------
# Extra IDs Preserve Forecast Identity
# -----------------------
class TestExtraIdsPreserveForecastIdentity:
    """Forecasts distinguished by ``extra_ids`` must not be pooled by ``source`` alone."""

    @pytest.fixture
    def nowcast_fd_extra_ids(self, nowcast_outturns, nowcast_forecasts):
        """NowcastData with two variants per source, distinguished by an extra id."""
        variant_a = nowcast_forecasts.copy()
        variant_a["model"] = "A"
        variant_b = nowcast_forecasts.copy()
        variant_b["model"] = "B"
        # Make variant B clearly less accurate so pooled and split stats differ.
        variant_b["value"] = variant_b["value"] * 1.5

        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(
            pd.concat([variant_a, variant_b], ignore_index=True),
            extra_ids=["model"],
            data_check=False,
        )
        return fd

    @pytest.mark.parametrize("compute", [compute_intra_period_accuracy, compute_intra_period_bias])
    def test_statistics_split_by_extra_ids(self, nowcast_fd_extra_ids, compute):
        """Statistics must be reported per (source, model), not pooled by source."""
        result = compute(nowcast_fd_extra_ids, variable="gdp", metric="levels", frequency="Q")

        assert "model" in result.columns
        assert "unique_id" in result.columns
        assert set(result["unique_id"].unique()) == {
            f"{source} + {model}" for source in nowcast_fd_extra_ids.df["source"].unique() for model in ("A", "B")
        }

        # One row per (source, model, days_to_period_end) - no pooling across models.
        assert not result.duplicated(subset=["source", "model", "days_to_period_end"]).any()

        # The two variants must produce different statistics.
        pivot = result.pivot_table(index=["source", "days_to_period_end"], columns="model", values="value")
        assert not pivot["A"].equals(pivot["B"])

    @pytest.mark.parametrize("plot", [plot_intra_period_accuracy, plot_intra_period_bias])
    def test_intra_period_plots_draw_one_line_per_unique_id(self, nowcast_fd_extra_ids, plot):
        """Each distinct forecast identity gets its own line and legend entry."""
        import matplotlib

        matplotlib.use("Agg")

        _, ax = plot(nowcast_fd_extra_ids, variable="gdp", metric="levels", frequency="Q", return_plot=True)
        labels = {text.get_text() for text in ax.get_legend().get_texts()}
        expected = {
            f"{source} + {model}" for source in nowcast_fd_extra_ids.df["source"].unique() for model in ("A", "B")
        }
        assert expected <= labels

    def test_plot_nowcasts_draws_one_line_per_unique_id(self, nowcast_fd_extra_ids):
        """``plot_nowcasts`` must not merge forecasts that share a source."""
        import matplotlib

        matplotlib.use("Agg")

        forecasts = nowcast_fd_extra_ids.forecasts
        target_date = forecasts[forecasts["variable"] == "gdp"]["date"].max()

        _, ax = plot_nowcasts(
            nowcast_fd_extra_ids,
            variable="gdp",
            target_date=target_date,
            metric="levels",
            frequency="Q",
            return_plot=True,
        )
        labels = {text.get_text() for text in ax.get_legend().get_texts()}
        expected = {
            f"{source} + {model}" for source in nowcast_fd_extra_ids.df["source"].unique() for model in ("A", "B")
        }
        assert expected <= labels

    def test_source_only_data_is_unchanged(self, nowcast_outturns, nowcast_forecasts):
        """Without extra_ids the grouping stays keyed on source (backwards compatible)."""
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        result = compute_intra_period_accuracy(fd, variable="gdp", metric="levels", frequency="Q")

        assert "source" in result.columns
        assert (result["unique_id"] == result["source"]).all()


# -----------------------
# Efficiency Tests Block for Nowcasts
# -----------------------
class TestEfficiencyBlockedForNowcasts:
    """Efficiency analysis functions should raise ValueError for nowcasting data."""

    @pytest.fixture
    def nowcast_fd(self, nowcast_outturns, nowcast_forecasts):
        fd = NowcastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)
        return fd

    def test_weak_efficiency_raises(self, nowcast_fd):
        with pytest.raises(ValueError, match="not supported for nowcasting"):
            fe.weak_efficiency_analysis(data=nowcast_fd)

    def test_strong_efficiency_raises(self, nowcast_fd):
        with pytest.raises(ValueError, match="not supported for nowcasting"):
            fe.strong_efficiency_analysis(
                data=nowcast_fd,
                source="nowcast_dfm",
                outcome_variable="gdp",
                outcome_metric="levels",
                instrument_variable="cpi",
                instrument_metric="levels",
            )

    def test_blanchard_leigh_raises(self, nowcast_fd):
        with pytest.raises(ValueError, match="not supported for nowcasting"):
            fe.blanchard_leigh_horizon_analysis(
                data=nowcast_fd,
                source="nowcast_dfm",
                outcome_variable="gdp",
                outcome_metric="levels",
                instrument_variable="cpi",
                instrument_metric="levels",
            )

    def test_revision_predictability_raises(self, nowcast_fd):
        with pytest.raises(ValueError, match="not supported for nowcasting"):
            fe.revision_predictability_analysis(data=nowcast_fd)

    def test_revisions_errors_correlation_raises(self, nowcast_fd):
        with pytest.raises(ValueError, match="not supported for nowcasting"):
            fe.revisions_errors_correlation_analysis(data=nowcast_fd)
