"""Tests for ForecastData with outturn_vintages=False."""

import pandas as pd
import pytest

from forecast_evaluation.core.outturns_revisions_table import create_outturn_revisions
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.loader import load_fer_forecasts, load_fer_outturns
from forecast_evaluation.utils import filter_k


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def outturns_no_vintages() -> pd.DataFrame:
    """Outturns without vintage_date or forecast_horizon columns."""
    n = 13
    return pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=n, freq="QE"),
            "variable": ["gdpkp"] * n,
            "frequency": ["Q"] * n,
            "value": list(range(100, 100 + n)),
        }
    )


@pytest.fixture
def outturns_no_vintages_multi_var() -> pd.DataFrame:
    """Outturns without vintages for multiple variables."""
    n = 13
    gdp = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=n, freq="QE"),
            "variable": ["gdpkp"] * n,
            "frequency": ["Q"] * n,
            "value": list(range(100, 100 + n)),
        }
    )
    cpi = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=n, freq="QE"),
            "variable": ["cpisa"] * n,
            "frequency": ["Q"] * n,
            "value": list(range(200, 200 + n)),
        }
    )
    return pd.concat([gdp, cpi], ignore_index=True)


@pytest.fixture
def forecasts() -> pd.DataFrame:
    """Forecasts with vintage_date (forecasts always need it)."""
    n = 13
    return pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=n, freq="QE"),
            "variable": ["gdpkp"] * n,
            "vintage_date": pd.to_datetime("2021-12-31"),
            "source": ["model_a"] * n,
            "frequency": ["Q"] * n,
            "value": list(range(99, 99 + n)),
            "forecast_horizon": list(range(0, n)),
        }
    )


@pytest.fixture
def forecasts_multi_source() -> pd.DataFrame:
    """Forecasts from two sources."""
    n = 13
    src_a = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=n, freq="QE"),
            "variable": ["gdpkp"] * n,
            "vintage_date": pd.to_datetime("2021-12-31"),
            "source": ["model_a"] * n,
            "frequency": ["Q"] * n,
            "value": list(range(99, 99 + n)),
            "forecast_horizon": list(range(0, n)),
        }
    )
    src_b = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=n, freq="QE"),
            "variable": ["gdpkp"] * n,
            "vintage_date": pd.to_datetime("2021-12-31"),
            "source": ["model_b"] * n,
            "frequency": ["Q"] * n,
            "value": list(range(98, 98 + n)),
            "forecast_horizon": list(range(0, n)),
        }
    )
    return pd.concat([src_a, src_b], ignore_index=True)


@pytest.fixture
def fd_no_vintages(outturns_no_vintages, forecasts) -> ForecastData:
    """ForecastData constructed with outturn_vintages=False."""
    return ForecastData(
        outturns_data=outturns_no_vintages,
        forecasts_data=forecasts,
        outturn_vintages=False,
    )


# -----------------------
# Constructor Tests
# -----------------------
class TestConstructorNoVintages:
    def test_init_outturns_without_vintage_date_column(self, outturns_no_vintages, forecasts):
        """Outturns missing vintage_date and forecast_horizon columns should be accepted."""
        fd = ForecastData(
            outturns_data=outturns_no_vintages,
            forecasts_data=forecasts,
            outturn_vintages=False,
        )
        assert not fd._main_table.empty
        assert fd.outturn_vintages is False

    def test_init_outturns_only(self, outturns_no_vintages):
        """Can add outturns alone without vintage columns."""
        fd = ForecastData(outturns_data=outturns_no_vintages, outturn_vintages=False)
        assert not fd.outturns.empty
        assert fd.outturn_vintages is False

    def test_outturn_vintages_property_default_true(self):
        """Default outturn_vintages is True."""
        fd = ForecastData()
        assert fd.outturn_vintages is True

    def test_vintage_date_auto_populated_as_nat(self, outturns_no_vintages):
        """vintage_date should be auto-populated as NaT when outturn_vintages=False."""
        fd = ForecastData(outturns_data=outturns_no_vintages, outturn_vintages=False)
        assert fd.outturns["vintage_date"].isna().all()

    def test_forecast_horizon_auto_populated(self, outturns_no_vintages):
        """forecast_horizon should be auto-populated when outturn_vintages=False."""
        fd = ForecastData(outturns_data=outturns_no_vintages, outturn_vintages=False)
        assert "forecast_horizon" in fd.outturns.columns

    def test_outturns_with_vintage_date_false_still_accepted(self, forecasts):
        """If the user provides vintage_date anyway, it should still work."""
        outturns = pd.DataFrame(
            {
                "date": pd.date_range(start="2022-01-01", periods=8, freq="QE"),
                "variable": ["gdpkp"] * 8,
                "vintage_date": pd.to_datetime("2025-09-30"),
                "frequency": ["Q"] * 8,
                "forecast_horizon": [-14] * 8,
                "value": [100, 101, 102, 103, 104, 105, 106, 107],
            }
        )
        fd = ForecastData(
            outturns_data=outturns,
            forecasts_data=forecasts,
            outturn_vintages=False,
        )
        assert not fd._main_table.empty

    def test_multiple_variables(self, outturns_no_vintages_multi_var, forecasts):
        """Works with multiple variables in outturns."""
        fd = ForecastData(
            outturns_data=outturns_no_vintages_multi_var,
            forecasts_data=forecasts,
            outturn_vintages=False,
        )
        assert not fd._main_table.empty


# -----------------------
# Main Table Tests
# -----------------------
class TestMainTableNoVintages:
    def test_main_table_has_k_column(self, fd_no_vintages):
        """Main table should have a k column set to 0."""
        assert "k" in fd_no_vintages.df.columns
        assert (fd_no_vintages.df["k"] == 0).all()

    def test_main_table_has_latest_vintage_nat(self, fd_no_vintages):
        """Main table should have latest_vintage column, all NaT."""
        assert "latest_vintage" in fd_no_vintages.df.columns
        assert fd_no_vintages.df["latest_vintage"].isna().all()

    def test_forecast_error_computed(self, fd_no_vintages):
        """Forecast errors should still be computed."""
        assert "forecast_error" in fd_no_vintages.df.columns
        assert fd_no_vintages.df["forecast_error"].notna().any()

    def test_forecast_error_values(self, outturns_no_vintages, forecasts):
        """Forecast errors should be outturn - forecast."""
        fd = ForecastData(
            outturns_data=outturns_no_vintages,
            forecasts_data=forecasts,
            outturn_vintages=False,
        )
        mt = fd.df
        errors = mt["value_outturn"] - mt["value_forecast"]
        pd.testing.assert_series_equal(mt["forecast_error"], errors, check_names=False)

    def test_main_table_has_expected_columns(self, fd_no_vintages):
        """Main table should have all standard columns."""
        expected = {
            "date",
            "variable",
            "vintage_date_forecast",
            "vintage_date_outturn",
            "unique_id",
            "metric",
            "frequency",
            "forecast_horizon",
            "value_forecast",
            "value_outturn",
            "k",
            "latest_vintage",
            "forecast_error",
            "source",
        }
        assert expected.issubset(set(fd_no_vintages.df.columns))


# -----------------------
# filter_k Tests
# -----------------------
class TestFilterKNoVintages:
    def test_filter_k_returns_all_rows(self, fd_no_vintages):
        """filter_k should be a no-op when latest_vintage is all NaT."""
        mt = fd_no_vintages.df
        filtered = filter_k(mt, k=12)
        assert len(filtered) == len(mt)


# -----------------------
# Statistical Tests (via filter_k gateway)
# -----------------------
class TestStatisticalTestsNoVintages:
    def test_accuracy(self, fd_no_vintages):
        """compute_accuracy_statistics should work without vintages."""
        import forecast_evaluation as fe

        result = fe.compute_accuracy_statistics(data=fd_no_vintages, k=12)
        assert len(result) > 0

    def test_bias(self, fd_no_vintages):
        """bias_analysis should work without vintages."""
        import forecast_evaluation as fe

        result = fe.bias_analysis(data=fd_no_vintages, k=12, verbose=False)
        assert len(result) > 0

    def test_diebold_mariano(self, outturns_no_vintages):
        """diebold_mariano_table should work without vintages."""
        import numpy as np

        import forecast_evaluation as fe

        # Use a single forecast_horizon so both sources have 13 paired
        # observations, enough for the DM test's lag structure.
        rng = np.random.default_rng(123)
        target_dates = pd.date_range(start="2022-01-01", periods=13, freq="QE")
        rows = []
        for src, bias in [("model_a", 0.5), ("model_b", 1.5)]:
            for d in target_dates:
                v = d - pd.offsets.QuarterEnd(1)
                rows.append(
                    {
                        "date": d,
                        "variable": "gdpkp",
                        "vintage_date": v,
                        "source": src,
                        "frequency": "Q",
                        "value": 100 + bias + rng.normal(0, 2),
                        "forecast_horizon": 1,
                    }
                )
        forecasts = pd.DataFrame(rows)
        fd = ForecastData(
            outturns_data=outturns_no_vintages,
            forecasts_data=forecasts,
            outturn_vintages=False,
        )
        result = fe.diebold_mariano_table(data=fd, benchmark_model="model_b", k=12)
        assert len(result) > 0


# -----------------------
# Guard Tests
# -----------------------
class TestGuardsNoVintages:
    def test_create_outturn_revisions_raises(self, fd_no_vintages):
        """create_outturn_revisions should raise ValueError."""
        with pytest.raises(ValueError, match="outturn vintages"):
            create_outturn_revisions(fd_no_vintages)

    def test_plot_outturn_revisions_raises(self, fd_no_vintages):
        """plot_outturn_revisions should raise ValueError."""
        with pytest.raises(ValueError, match="outturn vintages"):
            fd_no_vintages.plot_outturn_revisions(variable="gdpkp", metric="levels")

    def test_plot_outturns_raises(self, fd_no_vintages):
        """plot_outturns should raise ValueError."""
        with pytest.raises(ValueError, match="outturn vintages"):
            fd_no_vintages.plot_outturns(variable="gdpkp", metric="levels")


# -----------------------
# Merge Tests
# -----------------------
class TestMergeNoVintages:
    def test_merge_same_outturn_vintages_setting(self, outturns_no_vintages, forecasts):
        """Merging two ForecastData with outturn_vintages=False should succeed."""
        fd1 = ForecastData(outturns_data=outturns_no_vintages, outturn_vintages=False)
        fd2 = ForecastData(
            outturns_data=outturns_no_vintages,
            forecasts_data=forecasts,
            outturn_vintages=False,
        )
        fd1.merge(fd2)
        assert not fd1._main_table.empty

    def test_merge_different_outturn_vintages_raises(self, outturns_no_vintages):
        """Merging ForecastData with different outturn_vintages should raise ValueError."""
        fd_no = ForecastData(outturns_data=outturns_no_vintages, outturn_vintages=False)

        from forecast_evaluation.data.sample_data import create_sample_forecasts, create_sample_outturns

        fd_yes = ForecastData(
            outturns_data=create_sample_outturns(),
            forecasts_data=create_sample_forecasts(),
        )

        with pytest.raises(ValueError, match="outturn_vintages"):
            fd_no.merge(fd_yes)

        with pytest.raises(ValueError, match="outturn_vintages"):
            fd_yes.merge(fd_no)


# -----------------------
# Benchmark Model Tests
# -----------------------
class TestBenchmarkModelsNoVintages:
    @pytest.fixture
    def outturns_long(self) -> pd.DataFrame:
        """Long outturn series (20 quarters) without vintages, enough for AR estimation."""
        n = 20
        return pd.DataFrame(
            {
                "date": pd.date_range(start="2018-01-01", periods=n, freq="QE"),
                "variable": ["gdpkp"] * n,
                "frequency": ["Q"] * n,
                "value": [100 + i * 0.5 + (i % 3) * 0.2 for i in range(n)],
            }
        )

    @pytest.fixture
    def forecasts_multi_vintage(self) -> pd.DataFrame:
        """Forecasts with 3 distinct vintage_date values."""
        vintages = [
            pd.Timestamp("2022-04-01"),
            pd.Timestamp("2022-07-01"),
            pd.Timestamp("2022-10-01"),
        ]
        rows = []
        for v in vintages:
            for h, d in enumerate(pd.date_range(start=v, periods=4, freq="QE")):
                rows.append(
                    {
                        "date": d,
                        "variable": "gdpkp",
                        "vintage_date": v,
                        "source": "model_a",
                        "frequency": "Q",
                        "value": 110.0 + h,
                        "forecast_horizon": h,
                    }
                )
        return pd.DataFrame(rows)

    @pytest.fixture
    def fd_bench(self, outturns_long, forecasts_multi_vintage) -> ForecastData:
        """ForecastData with long outturns and multi-vintage forecasts, outturn_vintages=False."""
        return ForecastData(
            outturns_data=outturns_long,
            forecasts_data=forecasts_multi_vintage,
            outturn_vintages=False,
        )

    # --- Random Walk ---
    def test_rw_produces_vintaged_forecasts(self, fd_bench):
        """Random walk should produce one forecast set per forecast vintage."""
        from forecast_evaluation.core.random_walk_model import build_random_walk_model

        result = build_random_walk_model(fd_bench, variable="gdpkp", metric="levels", frequency="Q")
        vintage_dates = result["vintage_date"].unique()
        assert len(vintage_dates) == 3

    def test_rw_respects_realtime_constraint(self, fd_bench, outturns_long):
        """For each vintage V, the baseline (h=-1) value should equal the last outturn before V."""
        from forecast_evaluation.core.random_walk_model import build_random_walk_model

        result = build_random_walk_model(fd_bench, variable="gdpkp", metric="levels", frequency="Q")
        baselines = result[result["forecast_horizon"] == -1]

        for _, row in baselines.iterrows():
            v = row["vintage_date"]
            expected_last = outturns_long[outturns_long["date"] < v].sort_values("date").iloc[-1]["value"]
            assert row["value"] == expected_last, (
                f"Vintage {v}: baseline value {row['value']} != last outturn before vintage {expected_last}"
            )

    def test_rw_forecast_values_constant(self, fd_bench):
        """Random walk forecast values should all equal the baseline value for each vintage."""
        from forecast_evaluation.core.random_walk_model import build_random_walk_model

        result = build_random_walk_model(fd_bench, variable="gdpkp", metric="levels", frequency="Q")
        for v in result["vintage_date"].unique():
            v_data = result[result["vintage_date"] == v]
            baseline_val = v_data[v_data["forecast_horizon"] == -1]["value"].iloc[0]
            forecast_vals = v_data[v_data["forecast_horizon"] >= 0]["value"]
            assert (forecast_vals == baseline_val).all()

    # --- AR(p) ---
    def test_ar_produces_vintaged_forecasts(self, fd_bench):
        """AR(p) should produce one forecast set per forecast vintage."""
        from forecast_evaluation.core.ar_p_model import build_ar_p_model

        result = build_ar_p_model(
            fd_bench, variable="gdpkp", metric="levels", frequency="Q", estimation_start_date=None
        )
        vintage_dates = result["vintage_date"].unique()
        assert len(vintage_dates) == 3

    def test_ar_respects_realtime_constraint(self, fd_bench, outturns_long):
        """For each vintage V, the baseline (h=-1) value should equal the last outturn before V."""
        from forecast_evaluation.core.ar_p_model import build_ar_p_model

        result = build_ar_p_model(
            fd_bench, variable="gdpkp", metric="levels", frequency="Q", estimation_start_date=None
        )
        baselines = result[result["forecast_horizon"] == -1]

        for _, row in baselines.iterrows():
            v = row["vintage_date"]
            expected_last = outturns_long[outturns_long["date"] < v].sort_values("date").iloc[-1]["value"]
            assert row["value"] == expected_last, (
                f"Vintage {v}: baseline value {row['value']} != last outturn before vintage {expected_last}"
            )

    # --- add_* wrappers ---
    def test_add_rw_integrates(self, fd_bench):
        """add_random_walk_forecasts should add benchmark rows to main table."""
        from forecast_evaluation.core.random_walk_model import add_random_walk_forecasts

        add_random_walk_forecasts(fd_bench, variable="gdpkp", metric="levels", frequency="Q")
        sources = fd_bench.df["source"].unique()
        assert "baseline random walk model" in sources

    def test_add_ar_integrates(self, fd_bench):
        """add_ar_p_forecasts should add benchmark rows to main table."""
        from forecast_evaluation.core.ar_p_model import add_ar_p_forecasts

        add_ar_p_forecasts(fd_bench, variable="gdpkp", metric="levels", frequency="Q", estimation_start_date=None)
        sources = fd_bench.df["source"].unique()
        assert "baseline ar(p) model" in sources

    # --- Error cases ---
    def test_rw_raises_when_no_forecasts(self, outturns_long):
        """build_random_walk_model should raise ValueError when no forecasts exist."""
        from forecast_evaluation.core.random_walk_model import build_random_walk_model

        fd_empty = ForecastData(outturns_data=outturns_long, outturn_vintages=False)
        with pytest.raises(ValueError, match="no forecasts"):
            build_random_walk_model(fd_empty, variable="gdpkp", metric="levels", frequency="Q")

    def test_ar_raises_when_no_forecasts(self, outturns_long):
        """build_ar_p_model should raise ValueError when no forecasts exist."""
        from forecast_evaluation.core.ar_p_model import build_ar_p_model

        fd_empty = ForecastData(outturns_data=outturns_long, outturn_vintages=False)
        with pytest.raises(ValueError, match="no forecasts"):
            build_ar_p_model(fd_empty, variable="gdpkp", metric="levels", frequency="Q")


# -----------------------
# FER Data Snapshot Tests
# -----------------------
class TestFERDataNoOutturnVintages:
    @pytest.fixture
    def fd_fer_last_vintage(self) -> ForecastData:
        """ForecastData from minimal FER data using only the last outturn vintage."""
        outturns = load_fer_outturns(minimal=True)
        forecasts = load_fer_forecasts(minimal=True)

        last_vintage = outturns["vintage_date"].max()
        outturns = outturns[outturns["vintage_date"] == last_vintage].drop(columns=["vintage_date", "forecast_horizon"])

        return ForecastData(
            outturns_data=outturns,
            forecasts_data=forecasts,
            outturn_vintages=False,
            compute_levels=False,
        )

    def test_main_table_snapshot(self, fd_fer_last_vintage, snapshot):
        """Main table built from FER data with a single outturn vintage matches snapshot."""
        random_rows = fd_fer_last_vintage._main_table.sample(n=10, random_state=42)
        assert random_rows.to_dict() == snapshot
