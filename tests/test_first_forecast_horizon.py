"""Tests for first_forecast_horizon: per-variable horizon filtering and benchmark training cutoff."""

import pandas as pd
import pytest

from forecast_evaluation.core.ar_p_model import build_ar_p_model
from forecast_evaluation.core.random_walk_model import add_random_walk_forecasts, build_random_walk_model
from forecast_evaluation.data.ForecastData import ForecastData

VINTAGE_DATE = pd.Timestamp("2022-12-31")
VARIABLES = ("var_a", "var_b", "var_c")
# first_forecast_horizon per variable: -1, 0, 1
FFH_DICT = {"var_a": -1, "var_b": 0, "var_c": 1}


# -----------------------
# Helpers
# -----------------------


def make_outturns(n: int = 20) -> pd.DataFrame:
    """Quarterly outturns without vintage columns (outturn_vintages=False), ending at 2022-Q4."""
    frames = []
    for i, var in enumerate(VARIABLES):
        dates = pd.date_range(end="2022-12-31", periods=n, freq="QE")
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": var,
                    "frequency": "Q",
                    "value": [float(100 + i * 10 + j) for j in range(n)],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_forecasts(horizons=range(-1, 9)) -> pd.DataFrame:
    """Forecasts spanning horizons -1..8 for each variable, single vintage 2022-Q4."""
    frames = []
    for var in VARIABLES:
        dates = [VINTAGE_DATE + pd.offsets.QuarterEnd(h) for h in horizons]
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": var,
                    "vintage_date": VINTAGE_DATE,
                    "source": "test_model",
                    "frequency": "Q",
                    "value": [float(100 + h) for h in horizons],
                    "forecast_horizon": list(horizons),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def outturns() -> pd.DataFrame:
    return make_outturns()


@pytest.fixture
def forecasts() -> pd.DataFrame:
    return make_forecasts()


@pytest.fixture
def fd_dict(outturns, forecasts) -> ForecastData:
    """ForecastData with per-variable first_forecast_horizon dict."""
    fd = ForecastData(first_forecast_horizon=FFH_DICT, outturn_vintages=False)
    fd.add_outturns(outturns)
    fd.add_forecasts(forecasts, data_check=False)
    return fd


@pytest.fixture
def fd_default(outturns, forecasts) -> ForecastData:
    """ForecastData with default first_forecast_horizon=0."""
    fd = ForecastData(outturn_vintages=False)
    fd.add_outturns(outturns)
    fd.add_forecasts(forecasts, data_check=False)
    return fd


# -----------------------
# Horizon filter tests
# -----------------------


class TestFirstForecastHorizonFilter:
    """first_forecast_horizon controls which horizons survive into fd.forecasts."""

    def test_dict_min_horizon_per_variable(self, fd_dict):
        """Each variable's minimum horizon matches its entry in the dict."""
        assert fd_dict.forecasts.loc[fd_dict.forecasts["variable"] == "var_a", "forecast_horizon"].min() == -1
        assert fd_dict.forecasts.loc[fd_dict.forecasts["variable"] == "var_b", "forecast_horizon"].min() == 0
        assert fd_dict.forecasts.loc[fd_dict.forecasts["variable"] == "var_c", "forecast_horizon"].min() == 1

    def test_dict_max_horizon_unaffected(self, fd_dict):
        """first_forecast_horizon is a lower-bound only; the upper end is untouched."""
        for var in VARIABLES:
            mask = (fd_dict.forecasts["variable"] == var) & (fd_dict.forecasts["metric"] == "levels")
            assert fd_dict.forecasts.loc[mask, "forecast_horizon"].max() == 8

    def test_int_applies_to_all_variables(self, outturns, forecasts):
        """A scalar value applies uniformly: all variables respect the same floor."""
        fd = ForecastData(first_forecast_horizon=1, outturn_vintages=False)
        fd.add_outturns(outturns)
        fd.add_forecasts(forecasts, data_check=False)

        for var in VARIABLES:
            assert fd.forecasts.loc[fd.forecasts["variable"] == var, "forecast_horizon"].min() == 1

    def test_default_excludes_backcasts(self, fd_default):
        """Default first_forecast_horizon=0 means h<0 never appears in fd.forecasts."""
        assert (fd_default.forecasts["forecast_horizon"] < 0).sum() == 0

    def test_variable_absent_from_dict_defaults_to_zero(self, outturns, forecasts):
        """Variables not in the dict default to first_forecast_horizon=0."""
        fd = ForecastData(first_forecast_horizon={"var_a": -1}, outturn_vintages=False)
        fd.add_outturns(outturns)
        fd.add_forecasts(forecasts, data_check=False)

        assert fd.forecasts.loc[fd.forecasts["variable"] == "var_a", "forecast_horizon"].min() == -1
        assert fd.forecasts.loc[fd.forecasts["variable"] == "var_b", "forecast_horizon"].min() == 0
        assert fd.forecasts.loc[fd.forecasts["variable"] == "var_c", "forecast_horizon"].min() == 0

    def test_no_outturn_rows_leak_into_forecasts(self, fd_dict):
        """Helper outturn rows prepended for YoY/MoM transforms must not appear in forecasts."""
        assert set(fd_dict.forecasts["source"].unique()) == {"test_model"}

    def test_forecast_horizon_dtype_is_integer(self, fd_dict):
        """forecast_horizon must stay integer after the NaN-heavy concat in transformations."""
        assert pd.api.types.is_integer_dtype(fd_dict.forecasts["forecast_horizon"])


# -----------------------
# Benchmark training cutoff tests
# -----------------------


class TestFirstForecastHorizonBenchmarks:
    """Benchmark models must shift their training cutoff with first_forecast_horizon."""

    # expected dates based on outturns ending 2022-12-31 and vintage 2022-12-31
    EXPECTED = {
        "var_a": {
            "ffh": -1,
            "last_obs_h": -2,
            "last_obs_date": pd.Timestamp("2022-06-30"),
            "first_fc_date": pd.Timestamp("2022-09-30"),
        },
        "var_b": {
            "ffh": 0,
            "last_obs_h": -1,
            "last_obs_date": pd.Timestamp("2022-09-30"),
            "first_fc_date": pd.Timestamp("2022-12-31"),
        },
        "var_c": {
            "ffh": 1,
            "last_obs_h": 0,
            "last_obs_date": pd.Timestamp("2022-12-31"),
            "first_fc_date": pd.Timestamp("2023-03-31"),
        },
    }

    @pytest.fixture
    def fd_for_benchmarks(self, outturns, forecasts) -> ForecastData:
        """ForecastData with forecasts loaded so benchmarks can read vintage dates."""
        fd = ForecastData(first_forecast_horizon=FFH_DICT, outturn_vintages=False)
        fd.add_outturns(outturns)
        fd.add_forecasts(forecasts, data_check=False)
        return fd

    @pytest.mark.parametrize("var", VARIABLES)
    def test_rw_last_obs_horizon_label(self, fd_for_benchmarks, var):
        """Raw RW output: the 'last training value' entry is labelled first_forecast_horizon - 1."""
        ffh = FFH_DICT[var]
        rw = build_random_walk_model(
            fd_for_benchmarks, variable=var, metric="levels", frequency="Q", first_forecast_horizon=ffh
        )
        expected_h = self.EXPECTED[var]["last_obs_h"]
        assert expected_h in rw["forecast_horizon"].values

    @pytest.mark.parametrize("var", VARIABLES)
    def test_rw_last_obs_date(self, fd_for_benchmarks, var):
        """Training cutoff shifts correctly: last obs date differs per first_forecast_horizon."""
        ffh = FFH_DICT[var]
        rw = build_random_walk_model(
            fd_for_benchmarks, variable=var, metric="levels", frequency="Q", first_forecast_horizon=ffh
        )
        expected_h = self.EXPECTED[var]["last_obs_h"]
        expected_date = self.EXPECTED[var]["last_obs_date"]
        actual_date = rw.loc[rw["forecast_horizon"] == expected_h, "date"].iloc[0]
        assert actual_date == expected_date

    @pytest.mark.parametrize("var", VARIABLES)
    def test_rw_first_forecast_date(self, fd_for_benchmarks, var):
        """First forecast is produced at the correct date for each first_forecast_horizon."""
        ffh = FFH_DICT[var]
        rw = build_random_walk_model(
            fd_for_benchmarks, variable=var, metric="levels", frequency="Q", first_forecast_horizon=ffh
        )
        expected_date = self.EXPECTED[var]["first_fc_date"]
        actual_date = rw.loc[rw["forecast_horizon"] == ffh, "date"].iloc[0]
        assert actual_date == expected_date

    def test_rw_cutoff_shifts_independently_per_variable(self, fd_for_benchmarks):
        """The three variables use different training windows, not the same cutoff."""
        last_obs_dates = {}
        for var, meta in self.EXPECTED.items():
            rw = build_random_walk_model(
                fd_for_benchmarks, variable=var, metric="levels", frequency="Q", first_forecast_horizon=meta["ffh"]
            )
            last_obs_dates[var] = rw.loc[rw["forecast_horizon"] == meta["last_obs_h"], "date"].iloc[0]

        assert last_obs_dates["var_a"] < last_obs_dates["var_b"]
        assert last_obs_dates["var_b"] < last_obs_dates["var_c"]

    def test_add_rw_forecasts_end_to_end(self, fd_for_benchmarks):
        """add_random_walk_forecasts respects per-variable first_forecast_horizon end-to-end."""
        add_random_walk_forecasts(fd_for_benchmarks, metric="levels")

        rw = fd_for_benchmarks.forecasts[fd_for_benchmarks.forecasts["source"] == "baseline random walk model"]

        assert rw.loc[rw["variable"] == "var_a", "forecast_horizon"].min() == -1
        assert rw.loc[rw["variable"] == "var_b", "forecast_horizon"].min() == 0
        assert rw.loc[rw["variable"] == "var_c", "forecast_horizon"].min() == 1

    @pytest.mark.parametrize("var", VARIABLES)
    def test_arp_training_cutoff(self, fd_for_benchmarks, var):
        """AR(p) benchmark: same training-cutoff shift logic as the random walk."""
        ffh = FFH_DICT[var]
        ar = build_ar_p_model(
            fd_for_benchmarks,
            variable=var,
            metric="levels",
            frequency="Q",
            first_forecast_horizon=ffh,
            estimation_start_date=None,
        )

        expected_h = self.EXPECTED[var]["last_obs_h"]
        expected_date = self.EXPECTED[var]["last_obs_date"]
        actual_date = ar.loc[ar["forecast_horizon"] == expected_h, "date"].iloc[0]
        assert actual_date == expected_date


# -----------------------
# Non-levels backcast tests
# -----------------------


class TestFirstForecastHorizonNonLevels:
    """Backcasts (first_forecast_horizon < 0) must work through pop/yoy transformations."""

    def test_backcast_produces_pop_and_yoy_rows(self, outturns, forecasts):
        """Levels backcasts at h=-1 should yield pop and yoy rows at h=-1 too."""
        fd = ForecastData(first_forecast_horizon=-1, outturn_vintages=False)
        fd.add_outturns(outturns)
        fd.add_forecasts(forecasts, data_check=False)

        backcasts = fd.forecasts[fd.forecasts["forecast_horizon"] == -1]
        assert set(backcasts["metric"].unique()) >= {"levels", "pop", "yoy"}
        for var in VARIABLES:
            assert (backcasts["variable"] == var).any(), f"{var} missing from backcasts"

    def test_pop_forecasts_with_backcast(self, outturns):
        """Forecasts supplied directly as pop with ffh=-1 should survive the transform."""
        horizons = list(range(-1, 5))
        frames = []
        for var in VARIABLES:
            dates = [VINTAGE_DATE + pd.offsets.QuarterEnd(h) for h in horizons]
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "variable": var,
                        "vintage_date": VINTAGE_DATE,
                        "source": "test_model",
                        "frequency": "Q",
                        "metric": "pop",
                        "value": [0.5] * len(horizons),
                        "forecast_horizon": horizons,
                    }
                )
            )
        pop_forecasts = pd.concat(frames, ignore_index=True)

        fd = ForecastData(first_forecast_horizon=-1, outturn_vintages=False)
        fd.add_outturns(outturns)
        fd.add_forecasts(pop_forecasts, data_check=False)

        # pop rows at h=-1 must pass through
        pop_backcasts = fd.forecasts[(fd.forecasts["forecast_horizon"] == -1) & (fd.forecasts["metric"] == "pop")]
        for var in VARIABLES:
            assert (pop_backcasts["variable"] == var).any()


# -----------------------
# Validation tests
# -----------------------


class TestFirstForecastHorizonValidation:
    """Dict keys must match real variable names; partial dicts are allowed."""

    def test_unknown_variable_raises(self, outturns, forecasts):
        fd = ForecastData(first_forecast_horizon={"typo_var": -1}, outturn_vintages=False)
        fd.add_outturns(outturns)
        with pytest.raises(ValueError, match="typo_var"):
            fd.add_forecasts(forecasts, data_check=False)

    def test_partial_dict_allowed(self, outturns, forecasts):
        """Specifying only some variables is fine; the rest default to 0."""
        fd = ForecastData(first_forecast_horizon={"var_a": -1}, outturn_vintages=False)
        fd.add_outturns(outturns)
        fd.add_forecasts(forecasts, data_check=False)  # must not raise

        assert fd.forecasts.loc[fd.forecasts["variable"] == "var_a", "forecast_horizon"].min() == -1
        assert fd.forecasts.loc[fd.forecasts["variable"] == "var_b", "forecast_horizon"].min() == 0

    def test_unknown_variable_lists_known_in_message(self, outturns, forecasts):
        """Error message should list valid variable names to help debugging."""
        fd = ForecastData(first_forecast_horizon={"typo_var": -1, "var_a": 0}, outturn_vintages=False)
        fd.add_outturns(outturns)
        with pytest.raises(ValueError) as exc_info:
            fd.add_forecasts(forecasts, data_check=False)
        msg = str(exc_info.value)
        # Only the typo should be flagged, not var_a
        assert "typo_var" in msg
        assert "var_a" in msg  # listed under "Known variables"
