"""Tests for the user-supplied forecast_horizon contract."""

import pandas as pd
import pytest

from forecast_evaluation.core.ar_p_model import build_ar_p_model
from forecast_evaluation.core.random_walk_model import build_random_walk_model
from forecast_evaluation.data.ForecastData import ForecastData

VINTAGE_DATE = pd.Timestamp("2022-12-31")
VARIABLES = ("var_a", "var_b", "var_c")
LAST_OBSERVATION_DATES = {
    "var_a": pd.Timestamp("2022-06-30"),
    "var_b": pd.Timestamp("2022-09-30"),
    "var_c": pd.Timestamp("2022-12-31"),
}
REALTIME_VINTAGE_DATE = pd.Timestamp("2024-12-31")
REALTIME_LAST_OBSERVATION_DATE = pd.Timestamp("2024-03-31")
REALTIME_FIRST_FORECAST_DATE = pd.Timestamp("2024-06-30")


def make_outturns(n: int = 20) -> pd.DataFrame:
    """Return quarterly final outturns ending in 2022 Q4."""
    frames = []
    for index, variable in enumerate(VARIABLES):
        dates = pd.date_range(end="2022-12-31", periods=n, freq="QE")
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": variable,
                    "frequency": "Q",
                    "value": [float(100 + index * 10 + period) for period in range(n)],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_forecasts() -> pd.DataFrame:
    """Return forecasts whose horizons identify each variable's last observation."""
    frames = []
    for variable, last_observation_date in LAST_OBSERVATION_DATES.items():
        horizons = list(range(0, 7))
        dates = [last_observation_date + pd.offsets.QuarterEnd(horizon + 1) for horizon in horizons]
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": variable,
                    "vintage_date": VINTAGE_DATE,
                    "source": "test_model",
                    "frequency": "Q",
                    "value": [float(100 + horizon) for horizon in horizons],
                    "forecast_horizon": horizons,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_realtime_benchmark_outturns() -> pd.DataFrame:
    """Return final outturn history extending beyond a realtime training cutoff."""
    dates = pd.date_range(start="2018-03-31", end="2025-12-31", freq="QE")
    return pd.DataFrame(
        {
            "date": dates,
            "variable": "realtime_var",
            "frequency": "Q",
            "value": [float(index) for index in range(len(dates))],
        }
    )


def make_realtime_benchmark_forecasts() -> pd.DataFrame:
    """Declare a horizon-zero forecast for t-2 when the last observation is t-3."""
    dates = pd.date_range(start=REALTIME_FIRST_FORECAST_DATE, periods=3, freq="QE")
    return pd.DataFrame(
        {
            "date": dates,
            "variable": "realtime_var",
            "vintage_date": REALTIME_VINTAGE_DATE,
            "source": "test_model",
            "frequency": "Q",
            "value": [100.0, 101.0, 102.0],
            "forecast_horizon": [0, 1, 2],
        }
    )


@pytest.fixture
def outturns() -> pd.DataFrame:
    return make_outturns()


@pytest.fixture
def forecasts() -> pd.DataFrame:
    return make_forecasts()


@pytest.fixture
def data(outturns, forecasts) -> ForecastData:
    return ForecastData(outturns_data=outturns, forecasts_data=forecasts, data_check=False, outturn_vintages=False)


class TestForecastHorizonInput:
    def test_missing_forecast_horizon_raises(self, outturns, forecasts):
        """Forecast horizons must be supplied by the forecaster, not inferred."""
        with pytest.raises(
            ValueError,
            match="The 'forecast_horizon' column represents forecast_target_date - last_target_used_for_estimation - 1",
        ):
            ForecastData(
                outturns_data=outturns,
                forecasts_data=forecasts.drop(columns="forecast_horizon"),
                outturn_vintages=False,
            )

    def test_negative_forecast_horizons_are_filtered(self, outturns, forecasts):
        """Validated backcasts are excluded from the stored forecast tables."""
        forecasts = forecasts.copy()
        forecasts.loc[0, "forecast_horizon"] = -1

        data = ForecastData(outturns_data=outturns, forecasts_data=forecasts, data_check=False, outturn_vintages=False)

        assert len(data._raw_forecasts) == len(forecasts) - 1
        assert (data._raw_forecasts["forecast_horizon"] >= 0).all()
        assert (data.forecasts["forecast_horizon"] >= 0).all()
        assert (data.df["forecast_horizon"] >= 0).all()

    def test_all_negative_forecast_horizons_warn(self, outturns, forecasts):
        """Adding only backcasts warns that no usable forecasts remain."""
        forecasts = forecasts.assign(forecast_horizon=-1)

        with pytest.warns(UserWarning, match="No forecasts available after filtering/validation"):
            data = ForecastData(
                outturns_data=outturns,
                forecasts_data=forecasts,
                data_check=False,
                outturn_vintages=False,
            )

        assert data._raw_forecasts.empty
        assert data.forecasts.empty
        assert data.df.empty

    def test_non_negative_horizons_are_retained(self, data):
        """Non-negative forecast horizons are retained for every variable."""
        levels = data.forecasts[data.forecasts["metric"] == "levels"]
        actual = levels.groupby("variable")["forecast_horizon"].min().to_dict()

        assert actual == dict.fromkeys(VARIABLES, 0)
        assert (levels["forecast_horizon"] >= 0).all()

    def test_forecast_horizon_is_an_integer_after_transformations(self, data):
        """Derived rows retain the supplied information-horizon labels."""
        assert pd.api.types.is_integer_dtype(data.forecasts["forecast_horizon"])
        assert set(data.forecasts["source"].unique()) == {"test_model"}

    def test_custom_filter_uses_information_horizon(self, data):
        """ForecastData filtering remains keyed to the declared information horizon."""
        data.filter(
            custom_filter=lambda frame: (
                frame if "forecast_horizon" not in frame else frame[frame["forecast_horizon"] >= 0]
            )
        )

        assert (data.forecasts["forecast_horizon"] >= 0).all()
        assert (data.df["forecast_horizon"] >= 0).all()


class TestForecastHorizonAndVintageDistance:
    def test_information_horizon_is_not_replaced_by_vintage_distance(self, outturns):
        """The user-provided horizon can intentionally differ from calendar geometry."""
        forecasts = pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-03-31")],
                "variable": ["var_a"],
                "vintage_date": [VINTAGE_DATE],
                "source": ["test_model"],
                "frequency": ["Q"],
                "value": [101.0],
                "forecast_horizon": [4],
            }
        )

        data = ForecastData(outturns_data=outturns, forecasts_data=forecasts, data_check=False, outturn_vintages=False)
        row = data._raw_forecasts.iloc[0]

        assert row["forecast_horizon"] == 4
        assert row["target_minus_vintage"] == 1


class TestBenchmarkTrainingCutoffs:
    @pytest.mark.parametrize("variable", VARIABLES)
    def test_random_walk_uses_declared_horizon_to_choose_training_cutoff(self, data, variable):
        """The first baseline forecast follows the last observation implied by its horizon."""
        result = build_random_walk_model(data, variable=variable, metric="levels", frequency="Q", forecast_periods=1)
        first_forecast = result.loc[result["forecast_horizon"] == 0].iloc[0]
        expected_last_observation = LAST_OBSERVATION_DATES[variable]

        assert first_forecast["date"] == expected_last_observation + pd.offsets.QuarterEnd()

    @pytest.mark.parametrize("variable", VARIABLES)
    def test_ar_p_uses_declared_horizon_to_choose_training_cutoff(self, data, variable):
        """AR(p) benchmarks use the same last-observation contract as random walk."""
        result = build_ar_p_model(
            data,
            variable=variable,
            metric="levels",
            frequency="Q",
            forecast_periods=1,
            estimation_start_date=None,
        )
        first_forecast = result.loc[result["forecast_horizon"] == 0].iloc[0]
        expected_last_observation = LAST_OBSERVATION_DATES[variable]

        assert first_forecast["date"] == expected_last_observation + pd.offsets.QuarterEnd()

    @pytest.mark.parametrize("variable", VARIABLES)
    def test_random_walk_forecast_horizons_start_at_zero(self, data, variable):
        """Benchmark forecasts begin at horizon zero after the training cutoff."""
        result = build_random_walk_model(data, variable=variable, metric="levels", frequency="Q", forecast_periods=1)

        assert result["forecast_horizon"].min() == 0

    def test_random_walk_uses_t_minus_3_cutoff_and_starts_at_t_minus_2(self):
        """A horizon-zero benchmark forecast starts one period after the last observation."""
        outturns = make_realtime_benchmark_outturns()
        data = ForecastData(
            outturns_data=outturns,
            forecasts_data=make_realtime_benchmark_forecasts(),
            data_check=False,
            outturn_vintages=False,
        )

        result = build_random_walk_model(
            data,
            variable="realtime_var",
            metric="levels",
            frequency="Q",
            forecast_periods=3,
        )

        expected_dates = pd.date_range(start=REALTIME_FIRST_FORECAST_DATE, periods=3, freq="QE")
        assert result["date"].tolist() == expected_dates.tolist()
        assert result["forecast_horizon"].tolist() == [0, 1, 2]
        expected_value = outturns.loc[outturns["date"] == REALTIME_LAST_OBSERVATION_DATE, "value"].iloc[0]
        assert result.loc[result["forecast_horizon"] == 0, "value"].iloc[0] == expected_value

    def test_random_walk_uses_earliest_horizon_across_forecast_ids(self, outturns):
        """The cutoff uses the earliest supplied horizon even when it is not zero."""
        forecasts = pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-03-31")],
                "variable": ["var_a", "var_a"],
                "vintage_date": [VINTAGE_DATE, VINTAGE_DATE],
                "source": ["model_a", "model_b"],
                "frequency": ["Q", "Q"],
                "value": [101.0, 102.0],
                "forecast_horizon": [1, 2],
            }
        )
        data = ForecastData(
            outturns_data=outturns,
            forecasts_data=forecasts,
            data_check=False,
            outturn_vintages=False,
        )

        result = build_random_walk_model(data, variable="var_a", metric="levels", frequency="Q", forecast_periods=1)

        first_forecast = result.iloc[0]
        assert first_forecast["date"] == pd.Timestamp("2022-12-31")
        expected_value = outturns.loc[
            (outturns["variable"] == "var_a") & (outturns["date"] == pd.Timestamp("2022-09-30")), "value"
        ].iloc[0]
        assert first_forecast["value"] == expected_value

    def test_ar_p_uses_t_minus_3_cutoff_and_starts_at_t_minus_2(self):
        """AR(p) benchmarks use the declared realtime cutoff rather than final outturn history."""
        data = ForecastData(
            outturns_data=make_realtime_benchmark_outturns(),
            forecasts_data=make_realtime_benchmark_forecasts(),
            data_check=False,
            outturn_vintages=False,
        )

        result = build_ar_p_model(
            data,
            variable="realtime_var",
            metric="levels",
            frequency="Q",
            forecast_periods=3,
            max_lag=1,
            estimation_start_date=None,
        )

        expected_dates = pd.date_range(start=REALTIME_FIRST_FORECAST_DATE, periods=3, freq="QE")
        assert result["date"].tolist() == expected_dates.tolist()
        assert result["forecast_horizon"].tolist() == [0, 1, 2]
