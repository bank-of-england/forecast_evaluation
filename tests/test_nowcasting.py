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
from forecast_evaluation.visualisations.intra_period import plot_intra_period_accuracy


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def nowcast_outturns() -> pd.DataFrame:
    return create_sample_nowcast_outturns()


@pytest.fixture
def nowcast_forecasts() -> pd.DataFrame:
    return create_sample_nowcast_forecasts()


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

        # forecast_horizon is integer periods (e.g. -1 backcast, 0 nowcast, 1 one-quarter-ahead)
        fh = fd._raw_forecasts["forecast_horizon"]
        assert pd.api.types.is_integer_dtype(fh)
        # Backcasts have negative horizon (vintage after quarter-end)
        assert (fh < 0).any()
        # Horizons are small integers, not days
        assert fh.max() <= 10
        # Multiple weekly vintages per (source, date, horizon) group
        assert len(fd._raw_forecasts) > len(fh.unique()) * 2

        # k in main table is revision index (0, 1, 2, ...), not calendar-quarter distance
        if not fd.df.empty:
            assert fd.df["k"].min() >= -1

    def test_forecast_horizon_auto_computed(self, nowcast_outturns, nowcast_forecasts):
        """forecast_horizon should be computed automatically if missing, using integer-period method."""
        outturns_no_horizon = nowcast_outturns.drop(columns=["forecast_horizon"])

        fd = NowcastData(outturns_data=outturns_no_horizon)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert "forecast_horizon" in fd._raw_forecasts.columns
        # Integer horizons: at least -1, 0, 1
        unique_horizons = sorted(fd._raw_forecasts["forecast_horizon"].unique())
        assert -1 in unique_horizons or 0 in unique_horizons
        assert len(unique_horizons) >= 2

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


# -----------------------
# Intra-Period Visualisation Tests
# -----------------------
class TestIntraPeriodPlot:
    """Tests for the intra-period accuracy visualisation."""

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
