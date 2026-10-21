"""Tests for DensityForecastData class."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_evaluation.data import DensityForecastData, ForecastData
from forecast_evaluation.data.sample_data import (
    create_sample_density_forecasts,
    create_sample_forecasts,
    create_sample_outturns,
)
from forecast_evaluation.data.schema import create_data_schema
from forecast_evaluation.data.utils import compute_target_minus_vintage, construct_unique_id


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def sample_outturns() -> pd.DataFrame:
    return create_sample_outturns()


@pytest.fixture
def sample_density_forecasts() -> pd.DataFrame:
    """Create sample density forecast data with multiple quantiles."""
    return create_sample_density_forecasts()


@pytest.fixture
def multi_density_forecasts(sample_density_forecasts) -> pd.DataFrame:
    return pd.concat(
        [
            sample_density_forecasts.assign(variable=variable, source=source, metric="levels")
            for variable in ("gdpkp", "cpisa")
            for source in ("mpr2", "alternative")
        ],
        ignore_index=True,
    )


@pytest.fixture
def sample_density_forecasts_single_quantile() -> pd.DataFrame:
    """Create sample density forecast data with single quantile (median)."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = 0.5
    return forecasts


def _swap_quantile_values(
    forecasts: pd.DataFrame,
    date: str,
    lower: float,
    upper: float,
    *,
    id_column: str | None = None,
    id_value: object = None,
) -> pd.DataFrame:
    selected = forecasts["date"].eq(pd.Timestamp(date))
    if id_column is not None:
        id_mask = forecasts[id_column].isna() if pd.isna(id_value) else forecasts[id_column].eq(id_value)
        selected &= id_mask

    lower_mask = selected & np.isclose(forecasts["quantile"].to_numpy(dtype=float), lower)
    upper_mask = selected & np.isclose(forecasts["quantile"].to_numpy(dtype=float), upper)
    assert lower_mask.sum() == upper_mask.sum() == 1

    crossed = forecasts.copy()
    crossed.loc[lower_mask, "value"] = forecasts.loc[upper_mask, "value"].iloc[0]
    crossed.loc[upper_mask, "value"] = forecasts.loc[lower_mask, "value"].iloc[0]
    return crossed


def _assert_density_values_monotone(table: pd.DataFrame) -> None:
    key_columns = [column for column in table.columns if column not in ("quantile", "value")]
    for density_group in table.groupby(key_columns, sort=False, dropna=False):
        values = density_group[1].sort_values("quantile")["value"]
        assert values.is_monotonic_increasing


# -----------------------
# Constructor Tests
# -----------------------
def test_init_empty():
    """Test creating an empty DensityForecastData object."""
    dfd = DensityForecastData()
    assert dfd.df.empty
    assert dfd.forecasts.empty
    assert dfd.density_forecasts.empty
    assert dfd.outturns is not None


def test_constructor_parameters_match_parent_and_are_keyword_only():
    parent_params = inspect.signature(ForecastData.__init__).parameters
    density_params = inspect.signature(DensityForecastData.__init__).parameters

    assert list(density_params)[:-1] == list(parent_params)
    assert list(density_params)[-1] == "crossing"
    for name in (
        "extra_ids",
        "metric",
        "compute_levels",
        "data_check",
        "outturn_vintages",
        "default_k",
        "first_forecast_horizon",
        "crossing",
    ):
        assert density_params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_init_accepts_parent_options(sample_outturns):
    outturns = sample_outturns.drop(columns="metric", errors="ignore")
    dfd = DensityForecastData(
        outturns_data=outturns,
        metric="pop",
        compute_levels=False,
        data_check=False,
        outturn_vintages=False,
        default_k=3,
        first_forecast_horizon=1,
    )

    assert dfd._raw_outturns["metric"].eq("pop").all()
    assert dfd.default_k == 3
    assert dfd.outturn_vintages is False
    assert dfd.first_forecast_horizon == 1
    assert dfd._compute_levels is False


def test_add_density_forecast_options_are_keyword_only():
    params = inspect.signature(DensityForecastData.add_density_forecasts).parameters

    for name in ("extra_ids", "metric", "data_check", "first_forecast_horizon", "crossing"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_init_with_density_data(sample_outturns, sample_density_forecasts):
    """Test initialization with density forecast data."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    assert not dfd._density_df.empty
    assert not dfd.density_forecasts.empty
    assert "quantile" in dfd.density_forecasts.columns
    # Parent's forecasts should remain empty
    assert dfd.forecasts.empty


def test_init_with_fer_load(monkeypatch, sample_outturns):
    """Load point forecasts during parent initialisation without resetting density tables."""

    def add_fer_data(self):
        self.add_outturns(sample_outturns)
        self.add_forecasts(create_sample_forecasts(), data_check=False)

    monkeypatch.setattr(ForecastData, "add_fer_data", add_fer_data)
    dfd = DensityForecastData(load_fer=True)

    assert not dfd.forecasts.empty
    assert dfd.density_forecasts.empty


def test_init_with_single_quantile(sample_outturns, sample_density_forecasts_single_quantile):
    """Test initialisation with single quantile (median) data."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts_single_quantile)
    assert not dfd._density_df.empty
    # Check that all quantile values are 0.5 (handle both float and string)
    quantiles = dfd.density_forecasts["quantile"].unique()
    assert len(quantiles) == 1
    assert float(quantiles[0]) == 0.5


def test_init_missing_quantile_column_raises(sample_outturns):
    """Test that missing quantile column raises an error."""
    forecasts_without_quantile = create_sample_forecasts()

    with pytest.raises(ValueError, match="Density forecasts must include a 'quantile' column"):
        DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts_without_quantile)


def test_init_with_extra_ids(sample_outturns, sample_density_forecasts):
    """Test initialization with extra ID columns."""
    forecasts = pd.concat(
        [
            sample_density_forecasts.assign(region="UK"),
            sample_density_forecasts.assign(region="US", value=sample_density_forecasts["value"] + 10),
        ],
        ignore_index=True,
    )
    dfd = DensityForecastData(
        outturns_data=sample_outturns, forecasts_data=forecasts, extra_ids=["region"], data_check=False
    )
    assert dfd.id_columns == ["source", "region"]
    expected = forecasts.assign(quantile=forecasts["quantile"].round(10)).sort_values(
        ["region", "date", "quantile"]
    ).reset_index(drop=True)
    for table in (dfd._density_df, dfd.density_forecasts):
        assert set(table["unique_id"]) == {"mpr2 + UK", "mpr2 + US"}
        assert table.groupby("region")["unique_id"].nunique().to_dict() == {"UK": 1, "US": 1}
        actual = table.sort_values(["region", "date", "quantile"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            actual[["region", "date", "quantile", "value"]],
            expected[["region", "date", "quantile", "value"]],
            check_dtype=False,
        )


@pytest.mark.parametrize("on_init", [True, False])
def test_quantile_is_not_an_id(sample_outturns, sample_density_forecasts, on_init):
    """Explicit quantile labels must still form one density per source."""
    dfd = DensityForecastData(
        outturns_data=sample_outturns,
        forecasts_data=sample_density_forecasts if on_init else None,
        extra_ids=["quantile"] if on_init else None,
    )
    if not on_init:
        dfd.add_density_forecasts(sample_density_forecasts, extra_ids=["quantile"])

    assert dfd.id_columns == ["source"]
    assert dfd.density_forecasts["unique_id"].nunique() == 1
    assert dfd.density_forecasts.groupby(["date", "unique_id"])["quantile"].nunique().max() == 50


# -----------------------
# Add Forecasts Tests
# -----------------------
def test_add_density_forecasts(sample_outturns, sample_density_forecasts):
    """Test adding density forecasts to an existing object."""
    dfd = DensityForecastData(outturns_data=sample_outturns)
    assert dfd.density_forecasts.empty

    dfd.add_density_forecasts(sample_density_forecasts)
    assert not dfd.density_forecasts.empty
    assert "quantile" in dfd.density_forecasts.columns


def test_add_density_forecasts_without_quantile_raises(sample_outturns):
    """Test that adding forecasts without quantile column raises an error."""
    dfd = DensityForecastData(outturns_data=sample_outturns)
    forecasts_without_quantile = create_sample_forecasts()

    with pytest.raises(ValueError, match="Density forecasts must include a 'quantile' column"):
        dfd.add_density_forecasts(forecasts_without_quantile)


def test_density_forecasts_preserve_provided_metrics(sample_outturns):
    """Keep supplied pop and yoy quantiles without deriving other metrics."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = 0.5
    forecasts = pd.concat(
        [forecasts.assign(metric=metric) for metric in ("pop", "yoy")],
        ignore_index=True,
    )

    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)

    actual = dfd.density_forecasts.sort_values(["metric", "date"]).reset_index(drop=True)
    expected = forecasts.sort_values(["metric", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual[["date", "metric", "quantile", "value"]],
        expected[["date", "metric", "quantile", "value"]],
        check_dtype=False,
    )


def test_density_forecasts_default_to_levels(sample_outturns, sample_density_forecasts):
    """Default density forecasts to levels and warn when metric is omitted."""
    with pytest.warns(UserWarning, match="Forecasts input has no 'metric' column"):
        dfd = DensityForecastData(
            outturns_data=sample_outturns.assign(metric="levels"),
            forecasts_data=sample_density_forecasts,
            data_check=False,
        )

    assert dfd.density_forecasts["metric"].eq("levels").all()


def test_density_metric_argument_fills_null_metric_values(sample_outturns, sample_density_forecasts):
    forecasts = sample_density_forecasts.assign(metric=pd.NA)
    dfd = DensityForecastData(
        outturns_data=sample_outturns.assign(metric="pop"),
        forecasts_data=forecasts,
        metric="pop",
        data_check=False,
    )

    assert dfd.density_forecasts["metric"].eq("pop").all()


@pytest.mark.parametrize(
    ("quantiles", "data_check", "expect_warning"),
    [([0.25, 0.5, 0.75], True, True), ([0.25, 0.5, 0.75], False, False), ([0.05, 0.95], True, False)],
)
def test_density_data_check_uses_near_median_path(sample_outturns, quantiles, data_check, expect_warning):
    outturns = sample_outturns.assign(metric="levels")
    outturns = outturns[outturns["variable"] == "gdpkp"].sort_values("date").iloc[:3]
    forecasts = pd.DataFrame(
        [
            {
                "date": outturn["date"],
                "variable": outturn["variable"],
                "vintage_date": outturn["vintage_date"],
                "source": "check",
                "frequency": outturn["frequency"],
                "forecast_horizon": 1,
                "metric": "levels",
                "quantile": quantile,
                    "value": float(outturn["value"]) + date_index * 10 + quantiles.index(quantile) * 100,
            }
                for date_index, (_, outturn) in enumerate(outturns.iterrows())
            for quantile in quantiles
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DensityForecastData(outturns_data=outturns, forecasts_data=forecasts, data_check=data_check)

    check_warnings = [warning for warning in caught if "[Data check]" in str(warning.message)]
    assert bool(check_warnings) is expect_warning


def test_density_forecasts_drop_negative_horizons_with_warning(sample_outturns, sample_density_forecasts):
    forecasts = sample_density_forecasts.iloc[:4].copy()
    forecasts["metric"] = "levels"
    forecasts["forecast_horizon"] = [-1, -1, 0, 0]

    with pytest.warns(UserWarning, match="Dropped 2 density forecast rows with negative forecast_horizon"):
        dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts, data_check=False)

    assert len(dfd._density_df) == len(dfd._density_forecasts) == 2
    assert dfd._density_df["forecast_horizon"].eq(0).all()


def test_density_with_only_negative_horizons_adds_nothing(sample_outturns, sample_density_forecasts):
    forecasts = sample_density_forecasts.iloc[:4].assign(forecast_horizon=-1, metric="levels")

    with pytest.warns(UserWarning, match="No density forecasts available after filtering/validation"):
        dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts, data_check=False)

    assert dfd._density_df.empty
    assert dfd._density_forecasts.empty


def test_density_first_forecast_horizon_is_forwarded(sample_outturns, sample_density_forecasts):
    forecasts = sample_density_forecasts.drop(columns="forecast_horizon")
    forecasts = forecasts[forecasts["date"] > forecasts["date"].min()].copy()

    with pytest.warns(FutureWarning, match="first_forecast_horizon"):
        dfd = DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=forecasts,
            data_check=False,
            first_forecast_horizon=1,
        )

    expected = compute_target_minus_vintage(forecasts)["target_minus_vintage"] - 1
    pd.testing.assert_series_equal(
        dfd._density_df["forecast_horizon"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


def test_density_plot_missing_metric_raises_clear_error(sample_outturns, sample_density_forecasts):
    """Explain when the requested metric has no density forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    with pytest.raises(ValueError, match="No density forecasts found.*metric='pop'"):
        dfd.plot_density_vintage(variable="gdpkp", vintage_date="2025-12-31", metric="pop")


@pytest.mark.parametrize("density_first", [True, False])
def test_point_and_density_forecasts_must_share_frequency(sample_outturns, sample_density_forecasts, density_first):
    """Reject point and density forecast tables with different frequencies."""
    dfd = DensityForecastData(outturns_data=sample_outturns)
    monthly_forecasts = create_sample_forecasts().assign(frequency="M")

    if density_first:
        dfd.add_density_forecasts(sample_density_forecasts)
        with pytest.raises(ValueError, match="existing data has frequency 'Q'"):
            dfd.add_forecasts(monthly_forecasts, data_check=False)
    else:
        dfd.add_forecasts(create_sample_forecasts(), data_check=False)
        with pytest.raises(ValueError, match="existing data has frequency 'Q'"):
            dfd.add_density_forecasts(sample_density_forecasts.assign(frequency="M"))


def test_clear_filter_density_only(sample_outturns, sample_density_forecasts):
    """Restore density forecasts without requiring point forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    original = dfd.density_forecasts.copy(deep=True)
    outturns_before = dfd.outturns.copy(deep=True)
    raw_outturns_before = dfd._raw_outturns.copy(deep=True)
    dfd.filter(start_date="2024-09-30", end_date="2026-09-30")

    expected = original[original["date"].between("2024-09-30", "2026-09-30")]
    assert len(expected) == 200
    assert 0 < len(expected) < len(original)
    pd.testing.assert_frame_equal(dfd.density_forecasts.reset_index(drop=True), expected.reset_index(drop=True))
    pd.testing.assert_frame_equal(dfd._density_df, original)
    expected_outturns = outturns_before[outturns_before["date"].between("2024-09-30", "2026-09-30")]
    assert 0 < len(expected_outturns) < len(outturns_before)
    pd.testing.assert_frame_equal(dfd.outturns.reset_index(drop=True), expected_outturns.reset_index(drop=True))
    pd.testing.assert_frame_equal(dfd._raw_outturns, raw_outturns_before)

    dfd.clear_filter()

    pd.testing.assert_frame_equal(dfd.density_forecasts, original)
    pd.testing.assert_frame_equal(dfd.outturns, outturns_before)
    pd.testing.assert_frame_equal(dfd._raw_outturns, raw_outturns_before)
    assert dfd.forecasts.empty


def test_density_forecasts_retain_calendar_backcasts(sample_outturns):
    """Retain valid information-horizon forecasts before the vintage period."""
    forecasts = pd.DataFrame(
        {
            "date": pd.Timestamp("2022-12-31"),
            "variable": "gdpkp",
            "vintage_date": pd.Timestamp("2023-03-31"),
            "source": "mpr2",
            "frequency": "Q",
            "quantile": [0.25, 0.75],
            "value": [101.0, 103.0],
            "forecast_horizon": [0, 0],
        }
    )

    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)

    retained = dfd.density_forecasts[dfd.density_forecasts["metric"] == "levels"]
    assert len(retained) == 2
    assert retained["forecast_horizon"].tolist() == [0, 0]
    assert retained["target_minus_vintage"].tolist() == [-1, -1]


def test_rejected_density_forecasts_do_not_mutate_state(sample_outturns, sample_density_forecasts):
    """A rejected density forecast batch must leave the instance unchanged."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    before_id_columns = dfd._id_columns.copy()
    before_raw_forecasts = dfd._raw_forecasts.copy(deep=True)
    before_forecasts = dfd._forecasts.copy(deep=True)
    before_density_df = dfd._density_df.copy(deep=True)
    before_density_forecasts = dfd._density_forecasts.copy(deep=True)

    invalid = sample_density_forecasts.assign(value=lambda frame: frame["value"] + 1)
    with pytest.raises(ValueError):
        dfd.add_density_forecasts(invalid)

    assert dfd._id_columns == before_id_columns
    pd.testing.assert_frame_equal(dfd._raw_forecasts, before_raw_forecasts)
    pd.testing.assert_frame_equal(dfd._forecasts, before_forecasts)
    pd.testing.assert_frame_equal(dfd._density_df, before_density_df)
    pd.testing.assert_frame_equal(dfd._density_forecasts, before_density_forecasts)


# -----------------------
# Filtering Tests
# -----------------------
def test_filter_preserves_quantiles(sample_outturns, multi_density_forecasts):
    """Test that filtering preserves quantile information."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=multi_density_forecasts)
    original = dfd.density_forecasts.copy(deep=True)
    outturns_before = dfd.outturns.copy(deep=True)
    assert set(original["variable"]) == {"gdpkp", "cpisa"}
    assert set(outturns_before["variable"]) == {"gdpkp", "cpisa"}

    dfd.filter(variables=["gdpkp"])

    expected = original[original["variable"].eq("gdpkp")]
    assert len(expected) == 1300
    pd.testing.assert_frame_equal(dfd.density_forecasts.reset_index(drop=True), expected.reset_index(drop=True))
    pd.testing.assert_frame_equal(dfd._density_df, original)
    assert dfd.density_forecasts.groupby(["source", "date"])["quantile"].nunique().eq(50).all()
    expected_outturns = outturns_before[outturns_before["variable"].eq("gdpkp")]
    pd.testing.assert_frame_equal(dfd.outturns.reset_index(drop=True), expected_outturns.reset_index(drop=True))


def test_filter_by_source(sample_outturns, multi_density_forecasts):
    """Test filtering by source works with density forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=multi_density_forecasts)
    original = dfd.density_forecasts.copy(deep=True)
    outturns_before = dfd.outturns.copy(deep=True)
    assert set(original["source"]) == {"mpr2", "alternative"}

    dfd.filter(sources=["alternative"])

    expected = original[original["source"].eq("alternative")]
    assert len(expected) == 1300
    pd.testing.assert_frame_equal(dfd.density_forecasts.reset_index(drop=True), expected.reset_index(drop=True))
    pd.testing.assert_frame_equal(dfd.outturns, outturns_before)


# -----------------------
# Copy Tests
# -----------------------
def test_copy_is_deep(sample_outturns, sample_density_forecasts):
    """Test that copy creates a deep copy."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    dfd.add_forecasts(create_sample_forecasts(), data_check=False)
    dfd_copy = dfd.copy()

    for name in ("_density_df", "_density_forecasts", "_raw_forecasts", "_forecasts", "_raw_outturns", "_outturns"):
        original = getattr(dfd, name)
        copied = getattr(dfd_copy, name)
        assert not original.empty
        snapshot = original.copy(deep=True)

        original.iloc[0, original.columns.get_loc("value")] += 100
        pd.testing.assert_frame_equal(copied, snapshot)

        changed_original = original.copy(deep=True)
        copied.iloc[-1, copied.columns.get_loc("value")] -= 100
        pd.testing.assert_frame_equal(original, changed_original)


# -----------------------
# Integration Tests
# -----------------------
def test_density_forecasts_with_custom_filter(sample_outturns, multi_density_forecasts):
    """Test custom filtering works with density forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=multi_density_forecasts)
    dfd.add_forecasts(create_sample_forecasts(), data_check=False)
    original = dfd.density_forecasts.copy(deep=True)
    unchanged = {name: getattr(dfd, name).copy(deep=True) for name in ("outturns", "forecasts", "df")}

    def custom_filter(df):
        return df[df["quantile"] >= 0.5] if "quantile" in df.columns else df

    dfd.filter(custom_filter=custom_filter)

    expected = original[original["quantile"].ge(0.5)]
    assert len(expected) == 1300
    pd.testing.assert_frame_equal(dfd.density_forecasts.reset_index(drop=True), expected.reset_index(drop=True))
    pd.testing.assert_frame_equal(dfd._density_df, original)
    for name, snapshot in unchanged.items():
        assert not snapshot.empty
        pd.testing.assert_frame_equal(getattr(dfd, name), snapshot)


def test_density_forecasts_repr(sample_outturns, sample_density_forecasts):
    """Test that repr works for density forecasts."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    with pd.option_context("display.max_columns", None):
        repr_str = repr(dfd)
        assert repr_str == repr(dfd.density_forecasts)
    for label in ("quantile", "value", "gdpkp", "mpr2", "650 rows"):
        assert label in repr_str


def test_density_forecasts_repr_html_matches_repr(density_only):
    """Notebook HTML shows the density table, as repr does."""
    html = density_only._repr_html_()

    assert "quantile" in html
    assert html == density_only.density_forecasts._repr_html_()
    assert repr(density_only) == repr(density_only.density_forecasts)


def test_density_forecasts_repr_html_with_point_forecasts(density_and_point):
    """Point forecasts do not replace the density table in the HTML view."""
    assert "quantile" in density_and_point._repr_html_()


def test_density_module_has_no_demo_block():
    """The module is a library: no demo-only sample-data import or __main__ block."""
    import importlib

    module = importlib.import_module("forecast_evaluation.data.DensityForecastData")

    assert not hasattr(module, "create_sample_density_forecasts")
    assert "__main__" not in Path(module.__file__).read_text(encoding="utf-8")


# -----------------------
# Validation Tests
# -----------------------
def test_quantile_validation_out_of_range():
    """Test that quantile values outside [0,1] are caught by validation."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = 1.5  # Invalid quantile

    outturns = create_sample_outturns()

    with pytest.raises(ValueError, match="quantile must be between 0 and 1"):
        DensityForecastData(outturns_data=outturns, forecasts_data=forecasts)


def test_quantile_validation_negative():
    """Test that negative quantile values are caught by validation."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = -0.1  # Invalid quantile

    outturns = create_sample_outturns()

    with pytest.raises(ValueError, match="quantile must be between 0 and 1"):
        DensityForecastData(outturns_data=outturns, forecasts_data=forecasts)


@pytest.mark.parametrize("quantile", [-4e-11, 1.00000000004])
def test_quantiles_outside_bounds_are_not_rounded_into_range(sample_outturns, density_df, quantile):
    forecasts = density_df.iloc[[0]].copy()
    forecasts["quantile"] = quantile

    with pytest.raises(Exception, match="quantile"):
        DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts, data_check=False)


def test_default_crossing_rearrangement_warns_once_with_summary(sample_outturns, density_df):
    swaps = [("2026-06-30", 0.49, 0.51), ("2027-03-31", 0.15, 0.83)]
    crossed = density_df
    for date, lower, upper in swaps:
        crossed = _swap_quantile_values(crossed, date, lower, upper)

    expected_changes = []
    for date, lower, upper in swaps:
        date_rows = crossed[crossed["date"].eq(pd.Timestamp(date))]
        lower_value = date_rows.loc[np.isclose(date_rows["quantile"], lower), "value"].iloc[0]
        upper_value = date_rows.loc[np.isclose(date_rows["quantile"], upper), "value"].iloc[0]
        expected_changes.append(abs(lower_value - upper_value))
    expected_largest_change = max(expected_changes)

    data = DensityForecastData(outturns_data=sample_outturns)
    with pytest.warns(UserWarning, match="Rearranged 2 of 13 densities") as caught:
        data.add_density_forecasts(crossed, data_check=False)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert f"largest absolute change {expected_largest_change:g}" in message
    assert "example key:" in message
    assert "variable=gdpkp" in message
    assert "date=2026-06-30" in message

    expected = DensityForecastData(
        outturns_data=sample_outturns, forecasts_data=density_df, data_check=False
    )
    for table in (data._density_df, data._density_forecasts):
        actual_sorted = table.sort_values(["date", "quantile"]).reset_index(drop=True)
        expected_sorted = expected._density_df.sort_values(["date", "quantile"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_sorted, expected_sorted, check_dtype=False)
        _assert_density_values_monotone(table)


def test_crossing_error_mode_forwards_from_constructor_and_rolls_back_add(
    sample_outturns, density_df
):
    crossed = _swap_quantile_values(density_df, "2027-03-31", 0.15, 0.83)

    with pytest.raises(ValueError, match="not non-decreasing") as constructor_error:
        DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=crossed,
            crossing="error",
            data_check=False,
        )
    assert "1 of 13 densities" in str(constructor_error.value)
    assert "date=2027-03-31" in str(constructor_error.value)

    data = DensityForecastData(outturns_data=sample_outturns, forecasts_data=density_df, data_check=False)
    table_names = (
        "_raw_outturns",
        "_outturns",
        "_raw_forecasts",
        "_forecasts",
        "_main_table",
        "_density_df",
        "_density_forecasts",
    )
    before_tables = {name: getattr(data, name).copy(deep=True) for name in table_names}
    before_id_columns = data.id_columns.copy()
    before_horizon = data.first_forecast_horizon

    with pytest.raises(ValueError, match="not non-decreasing"):
        data.add_density_forecasts(
            crossed.assign(region="UK"),
            extra_ids=["region"],
            crossing="error",
            data_check=False,
        )

    assert data.id_columns == before_id_columns
    assert data.first_forecast_horizon == before_horizon
    for name, before in before_tables.items():
        pd.testing.assert_frame_equal(getattr(data, name), before)


def test_crossing_error_mode_is_scoped_to_each_added_batch(sample_outturns, density_df):
    base = density_df.iloc[[0]]
    lower = base.assign(quantile=0.25, value=2.0)
    upper = base.assign(quantile=0.75, value=1.0)
    data = DensityForecastData(outturns_data=sample_outturns)

    data.add_density_forecasts(lower, crossing="error", data_check=False)
    data.add_density_forecasts(upper, crossing="error", data_check=False)

    combined = data.density_forecasts.sort_values("quantile")
    assert combined["quantile"].tolist() == [0.25, 0.75]
    assert combined["value"].tolist() == [2.0, 1.0]
    assert not combined["value"].is_monotonic_increasing


def test_crossing_ignore_preserves_values_without_rearrangement_warning(sample_outturns, density_df):
    crossed = _swap_quantile_values(density_df, "2027-03-31", 0.15, 0.83)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=crossed,
            crossing="ignore",
            data_check=False,
        )

    assert not [warning for warning in caught if "Rearranged" in str(warning.message)]
    date_rows = data.density_forecasts[data.density_forecasts["date"].eq(pd.Timestamp("2027-03-31"))]
    assert not date_rows.sort_values("quantile")["value"].is_monotonic_increasing
    actual = data.density_forecasts.sort_values(["date", "quantile"]).reset_index(drop=True)
    expected = crossed.assign(quantile=crossed["quantile"].round(10)).sort_values(
        ["date", "quantile"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual[["date", "quantile", "value"]],
        expected[["date", "quantile", "value"]],
        check_dtype=False,
    )


def test_tied_and_monotone_quantiles_are_valid(sample_outturns, density_df):
    tied = density_df.copy()
    date_mask = tied["date"].eq(pd.Timestamp("2026-06-30"))
    lower_mask = date_mask & np.isclose(tied["quantile"], 0.49)
    upper_mask = date_mask & np.isclose(tied["quantile"], 0.51)
    tied.loc[upper_mask, "value"] = tied.loc[lower_mask, "value"].iloc[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=tied,
            crossing="error",
            data_check=False,
        )

    assert not [warning for warning in caught if "Rearranged" in str(warning.message)]
    _assert_density_values_monotone(data.density_forecasts)
    tied_rows = data.density_forecasts[data.density_forecasts["date"].eq(pd.Timestamp("2026-06-30"))]
    tied_values = tied_rows.loc[np.isclose(tied_rows["quantile"], 0.49), "value"].iloc[0]
    tied_upper = tied_rows.loc[np.isclose(tied_rows["quantile"], 0.51), "value"].iloc[0]
    assert tied_values == tied_upper


def test_density_finiteness_validation_does_not_change_point_or_outturn_schemas():
    point_forecast = create_sample_forecasts().iloc[[0]].copy()
    point_forecast["value"] = np.inf
    point_forecast["quantile"] = 0.5
    validated_point = create_data_schema(forecast=True, optional_columns=["quantile"]).validate(point_forecast)
    assert np.isposinf(validated_point["value"].iloc[0])

    outturn = create_sample_outturns().iloc[[0]].copy()
    outturn["value"] = np.inf
    validated_outturn = create_data_schema(forecast=False).validate(outturn)
    assert np.isposinf(validated_outturn["value"].iloc[0])


def test_point_forecast_quantile_extra_id_does_not_enable_density_finiteness(sample_outturns):
    forecasts = create_sample_forecasts().iloc[[0]].copy()
    forecasts["quantile"] = 0.5
    forecasts["metric"] = "levels"
    forecasts["value"] = np.inf
    point_data = ForecastData(outturns_data=sample_outturns, compute_levels=False)

    point_data.add_forecasts(
        forecasts,
        extra_ids=["quantile"],
        compute_levels=False,
        data_check=False,
    )

    assert np.isposinf(point_data.forecasts["value"].iloc[0])


@pytest.mark.parametrize(
    ("quantile", "value"),
    [(0.0, np.inf), (1.0, -np.inf), (0.5, np.nan)],
)
def test_non_finite_density_values_are_rejected(sample_outturns, density_df, quantile, value):
    forecasts = density_df.iloc[[0]].copy()
    forecasts["quantile"] = quantile
    forecasts["value"] = value

    with pytest.raises(ValueError, match="value must be finite"):
        DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts, data_check=False)


def test_finite_values_at_closed_quantile_bounds_are_accepted_and_sampleable(sample_outturns, density_df):
    forecasts = density_df[density_df["date"].eq(density_df["date"].min())].iloc[:2].copy()
    forecasts["quantile"] = [0.0, 1.0]
    forecasts["value"] = [-1.0, 1.0]
    data = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts, data_check=False)

    assert set(data.density_forecasts["quantile"]) == {0.0, 1.0}
    samples = data.sample_from_density(n_samples=100, random_state=1)
    assert np.isfinite(samples["value"]).all()


def test_numeric_quantiles_are_canonicalized_and_duplicates_checked_after_rounding(sample_outturns, density_df):
    forecast = density_df.iloc[[0]].copy()
    forecast["quantile"] = pd.Series(["0.12345678906"], index=forecast.index, dtype=object)
    canonical = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecast, data_check=False)
    assert canonical.density_forecasts["quantile"].iloc[0] == 0.1234567891

    first = density_df.iloc[[0]].copy().assign(quantile=0.15)
    second = first.copy()
    second["quantile"] = 0.15000000000000002
    second["value"] = second["value"] + 1
    numeric_like = pd.concat([first, second], ignore_index=True)
    numeric_like["quantile"] = pd.Series(["0.15", "0.15000000000000002"], dtype=object)
    with pytest.raises(ValueError, match="Duplicate records found with different values"):
        DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=numeric_like,
            data_check=False,
        )


def test_non_numeric_quantile_uses_existing_schema_validation(sample_outturns, density_df):
    forecasts = density_df.iloc[[0]].copy()
    forecasts["quantile"] = "not-a-quantile"

    with pytest.raises(ValueError, match="quantile"):
        DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts, data_check=False)


def test_crossing_rearrangement_groups_extra_ids_with_na(sample_outturns, density_df):
    uk = density_df.assign(region="UK")
    missing_region = density_df.assign(region=pd.NA)
    forecasts = pd.concat([uk, missing_region], ignore_index=True)
    crossed = _swap_quantile_values(
        forecasts,
        "2027-03-31",
        0.15,
        0.83,
        id_column="region",
        id_value=pd.NA,
    )

    with pytest.warns(UserWarning, match="Rearranged 1 of 26 densities") as caught:
        data = DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=crossed,
            extra_ids=["region"],
            data_check=False,
        )

    rearrangement_warnings = [warning for warning in caught if "Rearranged" in str(warning.message)]
    assert len(rearrangement_warnings) == 1
    assert "region=" in str(rearrangement_warnings[0].message)
    affected_date = pd.Timestamp("2027-03-31")
    missing_rows = data.density_forecasts[
        data.density_forecasts["region"].isna() & data.density_forecasts["date"].eq(affected_date)
    ]
    uk_rows = data.density_forecasts[
        data.density_forecasts["region"].eq("UK") & data.density_forecasts["date"].eq(affected_date)
    ]
    assert missing_rows.sort_values("quantile")["value"].is_monotonic_increasing
    pd.testing.assert_series_equal(
        uk_rows.sort_values("quantile")["value"].reset_index(drop=True),
        uk.loc[uk["date"].eq(affected_date)].sort_values("quantile")["value"].reset_index(drop=True),
        check_names=False,
    )


def test_sampling_reproduces_labelled_quantiles_after_rearrangement(sample_outturns, crossed_density_df):
    with pytest.warns(UserWarning, match="Rearranged 1 of 13 densities"):
        data = DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=crossed_density_df,
            data_check=False,
        )

    samples = data.sample_from_density(n_samples=2000, random_state=1)
    affected_date = pd.Timestamp("2027-03-31")
    sampled_values = samples.loc[samples["date"].eq(affected_date), "value"].to_numpy()
    labelled = data.density_forecasts[data.density_forecasts["date"].eq(affected_date)]
    for quantile_level, value in labelled[["quantile", "value"]].itertuples(index=False, name=None):
        empirical_quantile = float((sampled_values <= value).mean())
        assert abs(empirical_quantile - quantile_level) < 0.05


def test_unsorted_rows_preserve_density_and_sampling_results(sample_outturns, density_df):
    ordered = DensityForecastData(outturns_data=sample_outturns, forecasts_data=density_df, data_check=False)
    shuffled_input = density_df.sample(frac=1, random_state=7).reset_index(drop=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shuffled = DensityForecastData(
            outturns_data=sample_outturns,
            forecasts_data=shuffled_input,
            data_check=False,
        )

    assert not [warning for warning in caught if "Rearranged" in str(warning.message)]
    ordered_table = ordered.density_forecasts.sort_values(["date", "quantile"]).reset_index(drop=True)
    shuffled_table = shuffled.density_forecasts.sort_values(["date", "quantile"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered_table, shuffled_table, check_dtype=False)
    pd.testing.assert_frame_equal(
        ordered.sample_from_density(n_samples=30, random_state=9),
        shuffled.sample_from_density(n_samples=30, random_state=9),
    )


# -----------------------
# Edge Cases
# -----------------------
def test_single_quantile_at_boundary(sample_outturns):
    """Test density forecasts with quantile at boundary values (0 and 1)."""
    forecasts = create_sample_forecasts()
    forecasts["quantile"] = 0.0  # Minimum valid quantile

    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)
    # Handle both float and string types
    quantiles = [float(q) for q in dfd.density_forecasts["quantile"]]
    assert all(q == 0.0 for q in quantiles)

    forecasts["quantile"] = 1.0  # Maximum valid quantile
    forecasts["vintage_date"] = pd.Timestamp("2023-01-01")  # Change to avoid duplicates
    dfd.add_density_forecasts(forecasts)
    quantiles = [float(q) for q in dfd.density_forecasts["quantile"].unique()]
    assert 1.0 in quantiles


# -----------------------
# Conversion Tests
# -----------------------
@pytest.mark.parametrize("has_median", [True, False])
def test_to_point_forecast_median(sample_outturns, sample_density_forecasts, has_median):
    """Select exact medians silently or the lower nearest level with one warning."""
    forecasts = sample_density_forecasts.copy()
    if has_median:
        forecasts.loc[np.isclose(forecasts["quantile"], 0.49), "quantile"] = 0.5
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecasts)
    density_before = dfd.density_forecasts.copy(deep=True)
    selected_quantile = 0.5 if has_median else 0.49
    expected = density_before[np.isclose(density_before["quantile"], selected_quantile)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dfd.to_point_forecast(method="median")

    if has_median:
        assert not caught
    else:
        assert len(caught) == 1
        assert caught[0].category is UserWarning
        message = str(caught[0].message)
        assert "0.5" in message and "0.49" in message and "nearest" in message.lower()
    levels = dfd.forecasts[dfd.forecasts["metric"].eq("levels")]
    columns = ["date", "source", "variable", "metric", "frequency", "vintage_date", "forecast_horizon", "value"]
    assert len(levels) == len(expected) == 13
    pd.testing.assert_frame_equal(
        levels[columns].reset_index(drop=True), expected[columns].reset_index(drop=True), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        dfd._raw_forecasts[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(dfd.density_forecasts, density_before)


def test_to_point_forecast_missing_median_strict(sample_outturns, sample_density_forecasts):
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    density_before = dfd.density_forecasts.copy(deep=True)

    with pytest.raises(ValueError, match="on_missing='raise'"):
        dfd.to_point_forecast(method="median", on_missing="raise")

    assert dfd.forecasts.empty
    pd.testing.assert_frame_equal(dfd.density_forecasts, density_before)


def test_to_point_forecast_specific_quantile(sample_outturns, sample_density_forecasts):
    """Test converting density forecasts to point forecasts using a specific quantile."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    density_before = dfd.density_forecasts.copy(deep=True)
    expected = density_before[np.isclose(density_before["quantile"], 0.75)]
    assert len(expected) == 13

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dfd.to_point_forecast(method="0.75", on_missing="raise")

    assert not caught
    levels = dfd.forecasts[dfd.forecasts["metric"].eq("levels")]
    columns = ["date", "source", "variable", "metric", "frequency", "vintage_date", "forecast_horizon", "value"]
    assert len(levels) == 13
    pd.testing.assert_frame_equal(
        levels[columns].reset_index(drop=True), expected[columns].reset_index(drop=True), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        dfd._raw_forecasts[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(dfd.density_forecasts, density_before)


def test_to_point_forecast_invalid_method(sample_outturns, sample_density_forecasts):
    """Test that invalid method raises an error."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    with pytest.raises(ValueError, match="Invalid method"):
        dfd.to_point_forecast(method="invalid_method")


# -----------------------
# Sampling Tests
# -----------------------
def test_sample_from_density_statistics(sample_outturns, sample_density_forecasts):
    """Test that samples from density have correct mean and variance."""
    dfd = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    # 2,000 draws is ample for the tolerances asserted below (the observed errors are an
    # order of magnitude inside them) and keeps this test roughly five times faster than
    # the 10,000 draws it used to take.
    n_samples = 2000
    samples = dfd.sample_from_density(n_samples=n_samples, random_state=42)

    # get the levels
    samples = samples[samples["metric"] == "levels"]

    # Check that samples were generated
    assert not samples.empty
    assert len(samples) > 0
    assert "value" in samples.columns
    assert "sample_id" in samples.columns

    for horizon, date in enumerate(sample_density_forecasts["date"].unique()):
        date_samples = samples[samples["date"] == date]["value"]

        assert not date_samples.empty
        expected_mean = 115 + horizon
        expected_std = 0.5

        sample_mean = date_samples.mean()
        sample_std = date_samples.std()

        assert abs(sample_mean - expected_mean) < 0.05, (
            f"Mean mismatch at date {date}: expected {expected_mean}, got {sample_mean}"
        )
        assert abs(sample_std - expected_std) / expected_std < 0.10, (
            f"Std mismatch at date {date}: expected {expected_std}, got {sample_std}"
        )


# -----------------------
# Merge Tests
# -----------------------
def test_merge_non_overlapping_density_data(sample_outturns):
    """Test merging two DensityForecastData instances with non-overlapping forecast data."""
    forecast_df_1 = create_sample_density_forecasts()
    forecast_df_1["source"] = "source1"

    forecast_df_2 = create_sample_density_forecasts()
    forecast_df_2["source"] = "source2"

    fd1 = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecast_df_1)
    fd2 = DensityForecastData(outturns_data=sample_outturns, forecasts_data=forecast_df_2)
    fd1_original_len = len(fd1.density_forecasts)

    fd1.merge(fd2)

    assert len(fd1.density_forecasts) == fd1_original_len + len(fd2.density_forecasts)
    assert set(fd1.density_forecasts["source"].unique()) == {"source1", "source2"}


def test_merge_density_with_empty_instance(sample_outturns, sample_density_forecasts):
    """Test merging with an empty DensityForecastData instance."""
    fd1 = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    fd2 = DensityForecastData()

    merged = fd1.copy()

    merged.merge(fd2)

    assert len(merged.density_forecasts) == len(fd1.density_forecasts)
    assert len(merged.outturns) == len(fd1.outturns)


def test_merge_empty_density_with_populated(sample_outturns, sample_density_forecasts):
    """Test merging an empty instance with a populated one."""
    fd1 = DensityForecastData()
    fd2 = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)

    merged = fd1.copy()
    merged.merge(fd2)

    assert len(merged.density_forecasts) == len(fd2.density_forecasts)
    assert len(merged.outturns) == len(fd2.outturns)


def test_merge_density_with_different_extra_ids():
    """Test merging DensityForecastData instances with different extra_ids."""
    outturns_df = create_sample_outturns()

    forecast_df_1 = create_sample_density_forecasts()
    forecast_df_1["label1"] = "A"

    forecast_df_2 = create_sample_density_forecasts()
    forecast_df_2["label2"] = "B"

    fd1 = DensityForecastData(outturns_data=outturns_df, forecasts_data=forecast_df_1, extra_ids=["label1"])
    fd2 = DensityForecastData(outturns_data=outturns_df, forecasts_data=forecast_df_2, extra_ids=["label2"])

    merged = fd1.copy()
    merged.merge(fd2)

    # Both id columns should exist in merged data
    assert "label1" in merged.density_forecasts.columns
    assert "label2" in merged.density_forecasts.columns
    assert merged.id_columns == ["source", "label1", "label2"]
    for table in (merged._density_df, merged._density_forecasts):
        expected = construct_unique_id(table, merged.id_columns)
        pd.testing.assert_series_equal(table["unique_id"], expected, check_names=False)
        assert set(table["unique_id"]) == {"mpr2 + A + ", "mpr2 +  + B"}
        assert table["unique_id"].str.count(" \\+ ").eq(2).all()

    merged.filter(sources=["mpr2 + A + "])
    assert len(merged.density_forecasts) == len(forecast_df_1)
    merged.clear_filter()
    merged.filter(sources=["mpr2 + A"])
    assert merged.density_forecasts.empty
    merged.clear_filter()
    merged.filter(sources=["mpr2"])
    assert len(merged.density_forecasts) == len(forecast_df_1) + len(forecast_df_2)


def test_density_merge_is_atomic(sample_outturns, sample_density_forecasts):
    outturns = sample_outturns.assign(metric="levels")
    forecasts = sample_density_forecasts.assign(metric="levels")
    target = DensityForecastData(outturns_data=outturns, data_check=False)
    target.add_density_forecasts(forecasts, data_check=False)
    target.add_forecasts(create_sample_forecasts().assign(source="point", metric="levels"), data_check=False)
    conflicting = DensityForecastData(
        outturns_data=outturns,
        forecasts_data=forecasts.assign(value=lambda frame: frame["value"] + 1),
        data_check=False,
    )
    conflicting.add_outturns(outturns.assign(variable="cpiy"))

    table_names = (
        "_raw_outturns",
        "_outturns",
        "_raw_forecasts",
        "_forecasts",
        "_main_table",
        "_density_df",
        "_density_forecasts",
    )
    before = {name: getattr(target, name).copy(deep=True) for name in table_names}
    before_id_columns = target.id_columns.copy()
    before_horizon = target.first_forecast_horizon
    assert not target._raw_forecasts.empty
    assert not target._main_table.empty

    with pytest.raises(ValueError, match="Duplicate records found with different values"):
        target.merge(conflicting)

    assert target.id_columns == before_id_columns
    assert target.first_forecast_horizon == before_horizon
    for name, table in before.items():
        pd.testing.assert_frame_equal(getattr(target, name), table)


def test_forecastdata_merge_density_warns(sample_outturns, sample_density_forecasts):
    outturns = sample_outturns.assign(metric="levels")
    density = DensityForecastData(
        outturns_data=outturns,
        forecasts_data=sample_density_forecasts.assign(metric="levels"),
        data_check=False,
    )
    plain = ForecastData(outturns_data=outturns)

    with pytest.warns(UserWarning, match="Density forecasts.*were not merged"):
        plain.merge(density)


def test_merge_density_with_different_outturns():
    """Test merging with different outturn datasets."""
    outturns_df_1 = create_sample_outturns()
    outturns_df_1 = outturns_df_1[outturns_df_1["variable"] == "gdpkp"]

    outturns_df_2 = create_sample_outturns()
    outturns_df_2["variable"] = "cpiy"

    forecast_df_1 = create_sample_density_forecasts()
    forecast_df_1 = forecast_df_1[forecast_df_1["variable"] == "gdpkp"]

    forecast_df_2 = create_sample_density_forecasts()
    forecast_df_2["variable"] = "cpiy"

    fd1 = DensityForecastData(outturns_data=outturns_df_1, forecasts_data=forecast_df_1)
    fd2 = DensityForecastData(outturns_data=outturns_df_2, forecasts_data=forecast_df_2)

    merged = fd1.copy()
    merged.merge(fd2)

    # Should have outturns for both variables
    assert "gdpkp" in merged.outturns["variable"].values
    assert "cpiy" in merged.outturns["variable"].values
    # Should have density forecasts for both variables
    assert "gdpkp" in merged.density_forecasts["variable"].values
    assert "cpiy" in merged.density_forecasts["variable"].values


def test_merge_with_regular_forecast_data(sample_outturns, sample_density_forecasts):
    """Test merging a regular ForecastData instance into a DensityForecastData instance."""
    from forecast_evaluation.data import ForecastData

    # Create DensityForecastData (starts with density forecasts but empty point forecasts)
    fd_density = DensityForecastData(outturns_data=sample_outturns, forecasts_data=sample_density_forecasts)
    original_density_len = len(fd_density.density_forecasts)
    assert fd_density.forecasts.empty

    # Create regular ForecastData (has point forecasts)
    point_forecasts = create_sample_forecasts()
    point_forecasts["source"] = "point_source"
    fd_regular = ForecastData(outturns_data=sample_outturns, forecasts_data=point_forecasts)

    # Merge regular into density
    fd_density.merge(fd_regular)

    # Density forecasts should remain unchanged (regular ForecastData has none)
    assert len(fd_density.density_forecasts) == original_density_len

    # Point forecasts from the regular instance should be added to the density object
    assert not fd_density.forecasts.empty
    assert len(fd_density.forecasts) == fd_regular.forecasts.shape[0]
    assert "point_source" in fd_density.forecasts["source"].unique()
