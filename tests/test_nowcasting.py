import pandas as pd
import pytest

from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.sample_data import (
    compute_days_in_period,
    create_sample_forecasts,
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
    create_sample_outturns,
)
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
    """Test that days_in_period is computed and propagated for nowcasting data."""

    def test_days_in_period_with_nowcasting(self, nowcast_outturns, nowcast_forecasts):
        """days_in_period should be computed, stored as int, flow to main table, and have valid range."""
        fd = ForecastData(outturns_data=nowcast_outturns, nowcasting=True)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert "days_in_period" in fd._raw_forecasts.columns
        assert "days_in_period" in fd.forecasts.columns
        assert pd.api.types.is_integer_dtype(fd._raw_forecasts["days_in_period"])

        dip = pd.to_numeric(fd._raw_forecasts["days_in_period"])
        assert dip.min() >= 0
        assert dip.max() <= 91

        if not fd.df.empty:
            assert "days_in_period" in fd.df.columns

    def test_days_in_period_not_added_without_nowcasting(self):
        """Standard forecasts (nowcasting=False) should NOT get days_in_period."""
        fd = ForecastData(outturns_data=create_sample_outturns(), forecasts_data=create_sample_forecasts())
        assert "days_in_period" not in fd._raw_forecasts.columns

    def test_sample_data_has_days_in_period(self, nowcast_forecasts):
        """The sample nowcast data should already include days_in_period."""
        assert "days_in_period" in nowcast_forecasts.columns


# -----------------------
# Core Nowcasting Flow Tests
# -----------------------
class TestNowcastingFlow:
    """Test the core add/filter/evaluate flow with nowcasting data."""

    def test_nowcast_data_properties(self, nowcast_outturns, nowcast_forecasts):
        """Nowcast data should preserve row count, vintages, sources, variables, and horizons."""
        fd = ForecastData(outturns_data=nowcast_outturns, nowcasting=True)
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

        # Forecast horizon preserved and non-negative
        assert (fd._raw_forecasts["forecast_horizon"] >= 0).all()

        # k in main table is in quarterly units
        if not fd.df.empty:
            assert fd.df["k"].max() <= 20
            assert fd.df["k"].min() >= -1

    def test_forecast_horizon_auto_computed(self, nowcast_outturns, nowcast_forecasts):
        """forecast_horizon should be computed automatically if missing."""
        forecasts_no_horizon = nowcast_forecasts.drop(columns=["forecast_horizon"])
        outturns_no_horizon = nowcast_outturns.drop(columns=["forecast_horizon"])

        fd = ForecastData(outturns_data=outturns_no_horizon, nowcasting=True)
        fd.add_forecasts(forecasts_no_horizon, data_check=False)

        assert "forecast_horizon" in fd._raw_forecasts.columns
        assert (fd._raw_forecasts["forecast_horizon"] >= 0).all()

    def test_mixed_weekly_and_quarterly_vintages(self, nowcast_outturns, nowcast_forecasts):
        """Can add both quarterly-vintage and weekly-vintage forecasts."""
        outturns = pd.concat([nowcast_outturns, create_sample_outturns()], ignore_index=True)
        outturns = outturns.drop_duplicates(subset=[c for c in outturns.columns if c != "value"])

        fd = ForecastData(outturns_data=outturns, nowcasting=True)
        fd.add_forecasts(create_sample_forecasts(), data_check=False)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        sources = set(fd._raw_forecasts["source"].unique())
        assert "mpr2" in sources
        assert "nowcast_dfm" in sources

    def test_filter_by_source(self, nowcast_outturns, nowcast_forecasts):
        """Filtering by source should work with nowcast data."""
        fd = ForecastData(outturns_data=nowcast_outturns, nowcasting=True)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        fd.filter(sources="nowcast_dfm")
        assert set(fd.forecasts["source"].unique()) == {"nowcast_dfm"}

    def test_filter_by_variable(self, nowcast_outturns, nowcast_forecasts):
        """Filtering by variable should work with nowcast data."""
        fd = ForecastData(outturns_data=nowcast_outturns, nowcasting=True)
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
        fd = ForecastData(outturns_data=nowcast_outturns, nowcasting=True)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        import matplotlib

        matplotlib.use("Agg")

        for stat in ("rmse", "mae"):
            fig, ax = plot_intra_period_accuracy(
                fd, variable="gdp", metric="levels", frequency="Q", statistic=stat, return_plot=True
            )
            assert fig is not None
            assert ax is not None

    def test_plot_missing_days_in_period_raises(self):
        """Should raise ValueError if days_in_period column is missing."""
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
        with pytest.raises(ValueError, match="days_in_period"):
            plot_intra_period_accuracy(df, variable="gdp", return_plot=True)

    def test_plot_no_data_raises(self, nowcast_outturns, nowcast_forecasts):
        """Should raise ValueError when no data matches the filters."""
        fd = ForecastData(outturns_data=nowcast_outturns, nowcasting=True)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        import matplotlib

        matplotlib.use("Agg")

        with pytest.raises(ValueError, match="No data"):
            plot_intra_period_accuracy(fd, variable="nonexistent", metric="levels", frequency="Q", return_plot=True)
