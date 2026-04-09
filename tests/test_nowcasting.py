import pandas as pd
import pytest

from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.sample_data import (
    compute_days_in_period,
    create_sample_forecasts,
    create_sample_nowcast_forecasts,
    create_sample_nowcast_outturns,
    create_sample_outturns,
)  # compute_days_in_period is internal but tested directly here
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

    def test_quarterly_mid_quarter(self):
        """Feb 14 is day 44 of Q1 (Jan 1 = day 0)."""
        dates = pd.Series([pd.Timestamp("2024-02-14")])
        freqs = pd.Series(["Q"])
        result = compute_days_in_period(dates, freqs)
        assert result.iloc[0] == 44

    def test_quarterly_first_day(self):
        """Jan 1 is day 0 of Q1."""
        dates = pd.Series([pd.Timestamp("2024-01-01")])
        freqs = pd.Series(["Q"])
        result = compute_days_in_period(dates, freqs)
        assert result.iloc[0] == 0

    def test_quarterly_last_day(self):
        """Mar 31 is day 90 of Q1."""
        dates = pd.Series([pd.Timestamp("2024-03-31")])
        freqs = pd.Series(["Q"])
        result = compute_days_in_period(dates, freqs)
        assert result.iloc[0] == 90

    def test_monthly(self):
        """Feb 14 is day 13 of February."""
        dates = pd.Series([pd.Timestamp("2024-02-14")])
        freqs = pd.Series(["M"])
        result = compute_days_in_period(dates, freqs)
        assert result.iloc[0] == 13

    def test_quarterly_q2(self):
        """May 10 is day 39 of Q2 (Apr 1 = day 0)."""
        dates = pd.Series([pd.Timestamp("2024-05-10")])
        freqs = pd.Series(["Q"])
        result = compute_days_in_period(dates, freqs)
        assert result.iloc[0] == 39

    def test_all_non_negative(self):
        """All values should be non-negative."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="W-MON")
        freqs = pd.Series(["Q"] * len(dates))
        result = compute_days_in_period(dates.to_series().reset_index(drop=True), freqs)
        assert (result >= 0).all()

    def test_quarterly_range(self):
        """Quarterly values should be between 0 and 91 (max days in a quarter)."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        freqs = pd.Series(["Q"] * len(dates))
        result = compute_days_in_period(dates.to_series().reset_index(drop=True), freqs)
        assert result.max() <= 91
        assert result.min() >= 0


# -----------------------
# Auto-detection + days_in_period in ForecastData
# -----------------------
class TestDaysInPeriodAutoDetection:
    """Test that days_in_period is auto-detected and computed for nowcasting data."""

    def test_days_in_period_auto_computed(self, nowcast_outturns, nowcast_forecasts):
        """days_in_period should be auto-detected and added for weekly vintage data."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert "days_in_period" in fd._raw_forecasts.columns
        assert "days_in_period" in fd.forecasts.columns

    def test_days_in_period_in_main_table(self, nowcast_outturns, nowcast_forecasts):
        """days_in_period should flow through to the main table."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        if not fd.df.empty:
            assert "days_in_period" in fd.df.columns

    def test_days_in_period_values_reasonable(self, nowcast_outturns, nowcast_forecasts):
        """days_in_period values should be between 0 and 91 for quarterly data."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        dip = pd.to_numeric(fd._raw_forecasts["days_in_period"])
        assert dip.min() >= 0
        assert dip.max() <= 91

    def test_days_in_period_not_added_for_quarterly_vintages(self):
        """Standard quarterly-vintage forecasts should NOT get days_in_period."""
        outturns = create_sample_outturns()
        forecasts = create_sample_forecasts()

        fd = ForecastData(outturns_data=outturns, forecasts_data=forecasts)
        assert "days_in_period" not in fd._raw_forecasts.columns

    def test_days_in_period_is_int(self, nowcast_outturns, nowcast_forecasts):
        """days_in_period should be stored as integer."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert pd.api.types.is_integer_dtype(fd._raw_forecasts["days_in_period"])

    def test_sample_data_has_days_in_period(self, nowcast_forecasts):
        """The sample nowcast data should already include days_in_period."""
        assert "days_in_period" in nowcast_forecasts.columns


# -----------------------
# Core Nowcasting Flow Tests
# -----------------------
class TestNowcastingFlow:
    """Test the core add/filter/evaluate flow with nowcasting data."""

    def test_add_nowcast_forecasts(self, nowcast_outturns, nowcast_forecasts):
        """Weekly vintage dates should be accepted and preserved."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert len(fd._raw_forecasts) == len(nowcast_forecasts)

    def test_vintage_dates_not_snapped(self, nowcast_outturns, nowcast_forecasts):
        """Vintage dates should be preserved as-is (not snapped to quarter-end)."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        vintages = fd._raw_forecasts["vintage_date"]
        has_non_quarter_end = (vintages.dt.month % 3 != 0) | (vintages != vintages + pd.offsets.MonthEnd(0))
        assert has_non_quarter_end.any()

    def test_target_dates_snapped_to_quarter_end(self, nowcast_outturns, nowcast_forecasts):
        """Target dates should still be snapped to quarter-end."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        dates = fd._raw_forecasts["date"]
        expected_quarter_ends = dates + pd.offsets.QuarterEnd(0)
        pd.testing.assert_series_equal(dates, expected_quarter_ends, check_names=False)

    def test_main_table_k_in_quarters(self, nowcast_outturns, nowcast_forecasts):
        """The k column in main_table should be in quarterly units, not days."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        if not fd.df.empty:
            assert fd.df["k"].max() <= 20
            assert fd.df["k"].min() >= -1

    def test_two_models_present(self, nowcast_outturns, nowcast_forecasts):
        """Both nowcast models should be in the data."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        sources = set(fd._raw_forecasts["source"].unique())
        assert sources == {"nowcast_dfm", "nowcast_bridge"}

    def test_two_variables_present(self, nowcast_outturns, nowcast_forecasts):
        """Both variables should be in the data."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        variables = set(fd._raw_forecasts["variable"].unique())
        assert variables == {"gdp", "cpi"}

    def test_forecast_horizon_preserved(self, nowcast_outturns, nowcast_forecasts):
        """User-supplied forecast_horizon should be preserved, not recomputed."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        assert (fd._raw_forecasts["forecast_horizon"] >= 0).all()

    def test_levels_transformation(self, nowcast_outturns, nowcast_forecasts):
        """Levels-based transformations should work with weekly vintages."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        metrics = fd.forecasts["metric"].unique()
        assert "levels" in metrics

    def test_weekly_vintages_span_one_year(self, nowcast_forecasts):
        """Sample data should cover approximately one year of weekly vintages."""
        vintage_range = nowcast_forecasts["vintage_date"].max() - nowcast_forecasts["vintage_date"].min()
        assert vintage_range >= pd.Timedelta(days=300)

    def test_mixed_weekly_and_quarterly_vintages(self, nowcast_outturns, nowcast_forecasts):
        """Can add both quarterly-vintage and weekly-vintage forecasts for same variable."""
        quarterly_forecasts = create_sample_forecasts()

        outturns = pd.concat([nowcast_outturns, create_sample_outturns()], ignore_index=True)
        outturns = outturns.drop_duplicates(subset=[c for c in outturns.columns if c != "value"])

        fd = ForecastData(outturns_data=outturns)
        fd.add_forecasts(quarterly_forecasts, data_check=False)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        sources = set(fd._raw_forecasts["source"].unique())
        assert "mpr2" in sources
        assert "nowcast_dfm" in sources
        assert "nowcast_bridge" in sources

    def test_filter_by_source(self, nowcast_outturns, nowcast_forecasts):
        """Filtering by source should work with nowcast data."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        fd.filter(sources="nowcast_dfm")
        assert set(fd.forecasts["source"].unique()) == {"nowcast_dfm"}

    def test_filter_by_variable(self, nowcast_outturns, nowcast_forecasts):
        """Filtering by variable should work with nowcast data."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        fd.filter(variables="gdp")
        assert set(fd.forecasts["variable"].unique()) == {"gdp"}


# -----------------------
# Intra-Period Visualisation Tests
# -----------------------
class TestIntraPeriodPlot:
    """Tests for the intra-period accuracy visualisation."""

    def test_plot_returns_fig_ax(self, nowcast_outturns, nowcast_forecasts):
        """plot_intra_period_accuracy should return (fig, ax) when return_plot=True."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        import matplotlib

        matplotlib.use("Agg")

        result = plot_intra_period_accuracy(
            fd, variable="gdp", metric="levels", frequency="Q", forecast_horizon=0, return_plot=True
        )
        assert result is not None
        fig, ax = result
        assert fig is not None
        assert ax is not None

    def test_plot_mae_statistic(self, nowcast_outturns, nowcast_forecasts):
        """plot_intra_period_accuracy should work with MAE statistic."""
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        import matplotlib

        matplotlib.use("Agg")

        result = plot_intra_period_accuracy(
            fd, variable="gdp", metric="levels", frequency="Q", statistic="mae", return_plot=True
        )
        assert result is not None

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
        fd = ForecastData(outturns_data=nowcast_outturns)
        fd.add_forecasts(nowcast_forecasts, data_check=False)

        import matplotlib

        matplotlib.use("Agg")

        with pytest.raises(ValueError, match="No data"):
            plot_intra_period_accuracy(fd, variable="nonexistent", metric="levels", frequency="Q", return_plot=True)
