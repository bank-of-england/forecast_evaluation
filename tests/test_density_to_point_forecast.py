"""Tests for DensityForecastData.to_point_forecast: extra ids, level selection and repeat calls."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from forecast_evaluation.data import DensityForecastData
from forecast_evaluation.data.ForecastData import ForecastData

DATES = pd.date_range("2025-12-31", periods=13, freq="QE")


def _toy_density(levels, source="toy", mu=115.0, sigma=0.5, skip=None) -> pd.DataFrame:
    """Normal quantiles (mean mu + i) on ``levels`` for 13 dates; ``skip(i, q)`` omits a level at a date."""
    rows = [
        {
            "date": date,
            "variable": "gdpkp",
            "vintage_date": pd.Timestamp("2025-12-31"),
            "source": source,
            "frequency": "Q",
            "quantile": q,
            "value": float(norm.ppf(q, loc=mu + i, scale=sigma)),
            "forecast_horizon": i,
            "metric": "levels",
        }
        for i, date in enumerate(DATES)
        for q in levels
        if not (skip and skip(i, q))
    ]
    return pd.DataFrame(rows)


def _levels(data: DensityForecastData, **where) -> pd.Series:
    """Point-forecast 'levels' values by date, optionally restricted by column equalities."""
    fc = data.forecasts
    mask = fc["metric"] == "levels"
    for column, value in where.items():
        mask &= fc[column] == value
    return fc[mask].set_index("date")["value"].sort_index()


def _stored(data: DensityForecastData, level: float, **where) -> pd.Series:
    """Density 'levels' values by date at ``level``, optionally restricted by column equalities."""
    den = data._density_df
    mask = (den["metric"] == "levels") & np.isclose(den["quantile"], level, rtol=0, atol=1e-9)
    for column, value in where.items():
        mask &= den[column] == value
    return den[mask].set_index("date")["value"].sort_index()


def _call(data: DensityForecastData, *args, **kwargs) -> list[str]:
    """Run ``to_point_forecast`` and return the UserWarning messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data.to_point_forecast(*args, **kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _selection_warnings(messages: list[str]) -> list[str]:
    return [m for m in messages if m.startswith("to_point_forecast(")]


def _assert_same_values(actual: pd.Series, expected: pd.Series) -> None:
    pd.testing.assert_series_equal(actual, expected, check_names=False, check_freq=False)


# -----------------------
# A3: extra ids
# -----------------------
def test_signature():
    params = inspect.signature(DensityForecastData.to_point_forecast).parameters

    assert params["method"].default == "median"
    for name in ("on_missing", "source_suffix", "compute_levels", "data_check"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["on_missing"].default == "nearest"
    assert params["source_suffix"].default is None
    assert params["compute_levels"].default is None
    assert params["data_check"].default is False


def test_keeps_extra_ids(density_extra_ids):
    _call(density_extra_ids, "0.25")

    fc = density_extra_ids.forecasts
    assert fc["region"].eq("UK").all()
    assert set(fc["unique_id"]) == set(density_extra_ids.density_forecasts["unique_id"]) == {"mpr2 + UK"}
    _assert_same_values(_levels(density_extra_ids), _stored(density_extra_ids, 0.25))


@pytest.mark.parametrize(("method", "level"), [("0.25", 0.25), ("median", 0.49)])
def test_two_regions(density_two_regions, method, level):
    _call(density_two_regions, method)

    fc = density_two_regions.forecasts
    assert set(fc["region"]) == {"UK", "EW"}
    assert set(fc["unique_id"]) == {"mpr2 + UK", "mpr2 + EW"}
    for region in ("UK", "EW"):
        _assert_same_values(
            _levels(density_two_regions, region=region), _stored(density_two_regions, level, region=region)
        )


def test_passes_add_options(density_extra_ids, monkeypatch):
    calls = []
    monkeypatch.setattr(ForecastData, "add_forecasts", lambda self, df, **kwargs: calls.append((df, kwargs)))

    density_extra_ids.to_point_forecast("0.25", compute_levels=False, data_check=False)

    assert len(calls) == 1
    df, kwargs = calls[0]
    assert kwargs == {"extra_ids": ["region"], "compute_levels": False, "data_check": False}
    assert len(df) == 13
    assert df["region"].eq("UK").all()
    assert "quantile" not in df.columns


def test_constructor_compute_levels_is_default_for_conversion(density_outturns_df, density_df, monkeypatch):
    data = DensityForecastData(
        outturns_data=density_outturns_df,
        forecasts_data=density_df,
        compute_levels=False,
        data_check=False,
    )
    calls = []
    monkeypatch.setattr(ForecastData, "add_forecasts", lambda self, df, **kwargs: calls.append(kwargs))

    data.to_point_forecast("0.25")

    assert calls == [{"extra_ids": [], "compute_levels": False, "data_check": False}]


# -----------------------
# A9: level selection
# -----------------------
@pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
def test_median_tie_breaks_to_lower_level_whatever_the_row_order(density_outturns_df, density_df, order):
    if order == "descending":
        density_df = density_df.sort_values("quantile", ascending=False)
    elif order == "shuffled":
        density_df = density_df.sample(frac=1, random_state=1)
    data = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=density_df)

    _call(data, "median")

    _assert_same_values(_levels(data), _stored(data, 0.49))


@pytest.mark.parametrize("reverse", [False, True])
def test_even_grid_tie_uses_lower_level(density_outturns_df, reverse):
    levels = [0.1, 0.4, 0.6, 0.9]
    data = DensityForecastData(
        outturns_data=density_outturns_df, forecasts_data=_toy_density(levels[::-1] if reverse else levels)
    )

    _call(data, "median")

    _assert_same_values(_levels(data), _stored(data, 0.4))


def test_mixed_grids_keep_every_source(density_outturns_df):
    frame = pd.concat(
        [_toy_density([0.1, 0.3, 0.5, 0.7, 0.9], source="A"), _toy_density([0.4, 0.6], source="B", mu=120.0)],
        ignore_index=True,
    )
    data = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=frame)

    messages = _selection_warnings(_call(data, "median"))

    assert set(data.forecasts["source"]) == {"A", "B"}
    _assert_same_values(_levels(data, source="A"), _stored(data, 0.5, source="A"))
    _assert_same_values(_levels(data, source="B"), _stored(data, 0.4, source="B"))
    (message,) = messages
    assert "source=B" in message
    assert "source=A" not in message
    assert "1 of 2" in message


def test_one_level_per_path(density_outturns_df):
    # 0.5 is stored at the first date only, so it cannot be used for the whole path.
    frame = _toy_density([0.25, 0.5, 0.75], skip=lambda i, q: i > 0 and q == 0.5)
    data = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=frame)

    messages = _selection_warnings(_call(data, "median"))

    assert len(messages) == 1
    _assert_same_values(_levels(data), _stored(data, 0.25))


def test_path_without_common_level_raises(density_outturns_df):
    frame = _toy_density([0.25, 0.75], skip=lambda i, q: (i == 0 and q == 0.25) or (i == 1 and q == 0.75))
    data = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=frame)

    with pytest.raises(ValueError, match="no quantile level stored at every date"):
        data.to_point_forecast("median")

    assert data.forecasts.empty


@pytest.mark.parametrize(
    ("method", "match"),
    [
        (1.5, "must be a number in"),
        (-0.1, "must be a number in"),
        ("1.5", "must be a number in"),
        (float("nan"), "must be a number in"),
        (float("inf"), "must be a number in"),
        (None, "must be a number in"),
        (True, "must be a number in"),
        ("foo", "Invalid method"),
        ("Median", "Invalid method"),
    ],
)
def test_method_validation(density_only, method, match):
    with pytest.raises(ValueError, match=match):
        density_only.to_point_forecast(method)

    assert density_only.forecasts.empty


def test_method_mean_not_implemented(density_only):
    with pytest.raises(NotImplementedError):
        density_only.to_point_forecast("mean")


def test_method_accepts_numbers_and_boundaries(density_only):
    _call(density_only, 0.25)

    _assert_same_values(_levels(density_only), _stored(density_only, 0.25))


def test_level_matches_float_noise(density_outturns_df):
    # 0.1 + 0.2 is stored as 0.30000000000000004, which == 0.3 does not match.
    frame = _toy_density([0.1, 0.1 + 0.2, 0.7, 0.9])
    data = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=frame)

    messages = _call(data, "0.3")

    assert not _selection_warnings(messages)
    _assert_same_values(_levels(data), _stored(data, 0.3))


def test_warning_names_requested_used_level_and_distance(density_only):
    messages = _selection_warnings(_call(density_only, "median"))

    (message,) = messages
    for expected in ("0.5", "0.49", "0.01", "1 of 1", "lower level", "on_missing='raise'"):
        assert expected in message


def test_bvar_grid_exact_median_is_quiet_and_other_levels_warn(density_outturns_df, bvar_grid_density_df):
    data = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=bvar_grid_density_df)

    assert not _selection_warnings(_call(data, "median"))
    _assert_same_values(_levels(data), _stored(data, 0.5))

    other = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=bvar_grid_density_df)
    (message,) = _selection_warnings(_call(other, "0.25"))
    assert "0.16" in message
    assert "0.09" in message
    _assert_same_values(_levels(other), _stored(other, 0.16))


def test_on_missing_raise_lists_at_most_five_densities(density_outturns_df):
    frame = pd.concat([_toy_density([0.4, 0.6], source=f"s{i}") for i in range(7)], ignore_index=True)
    data = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=frame)

    with pytest.raises(ValueError) as excinfo:
        data.to_point_forecast("median", on_missing="raise")

    message = str(excinfo.value)
    assert "7 of 7" in message
    assert message.count("source=") == 5
    assert "and 2 more" in message
    assert data.forecasts.empty


def test_on_missing_raise_names_only_the_lacking_source(density_outturns_df):
    frame = pd.concat(
        [_toy_density([0.1, 0.5, 0.9], source="A"), _toy_density([0.4, 0.6], source="B", mu=120.0)], ignore_index=True
    )
    data = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=frame)

    with pytest.raises(ValueError, match="source=B") as excinfo:
        data.to_point_forecast("median", on_missing="raise")

    assert "source=A" not in str(excinfo.value)


def test_on_missing_raise_passes_when_level_is_stored(density_only):
    messages = _call(density_only, "0.25", on_missing="raise")

    assert not _selection_warnings(messages)
    _assert_same_values(_levels(density_only), _stored(density_only, 0.25))


def test_on_missing_invalid_value(density_only):
    with pytest.raises(ValueError, match="on_missing"):
        density_only.to_point_forecast("0.25", on_missing="interpolate")


# -----------------------
# A9: repeat calls and source collisions
# -----------------------
@pytest.mark.parametrize("method", ["0.25", "median"])
def test_repeat_identical_call_is_quiet_noop(density_only, method):
    _call(density_only, method)
    forecasts = density_only.forecasts.copy()
    raw = density_only._raw_forecasts.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        density_only.to_point_forecast(method)

    pd.testing.assert_frame_equal(density_only.forecasts, forecasts)
    pd.testing.assert_frame_equal(density_only._raw_forecasts, raw)


def test_existing_identical_point_rows_are_a_quiet_noop(density_only, density_df):
    rows = density_df[np.isclose(density_df["quantile"], 0.25)].drop(columns="quantile")
    density_only.add_forecasts(rows, data_check=False)
    forecasts = density_only.forecasts.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        density_only.to_point_forecast("0.25")

    pd.testing.assert_frame_equal(density_only.forecasts, forecasts)


def test_conflicting_repeat_raises_clear_error(density_only):
    _call(density_only, "0.25")
    forecasts = density_only.forecasts.copy()

    with pytest.raises(ValueError, match="source_suffix"):
        density_only.to_point_forecast("0.75")

    pd.testing.assert_frame_equal(density_only.forecasts, forecasts)


def test_conflict_with_existing_point_model_and_suffix_resolves_it(density_only, density_df):
    rows = density_df[np.isclose(density_df["quantile"], 0.25)].drop(columns="quantile")
    density_only.add_forecasts(rows.assign(value=rows["value"] + 10), data_check=False)

    with pytest.raises(ValueError, match="source_suffix"):
        density_only.to_point_forecast("0.25")

    _call(density_only, "0.25", source_suffix=" (q0.25)")

    assert set(density_only.forecasts["source"]) == {"mpr2", "mpr2 (q0.25)"}


def test_source_suffix_stores_levels_side_by_side(density_only):
    _call(density_only, "0.25", source_suffix=" (q0.25)")
    _call(density_only, "0.75", source_suffix=" (q0.75)")

    fc = density_only.forecasts
    assert set(fc["source"]) == {"mpr2 (q0.25)", "mpr2 (q0.75)"}
    assert (fc["unique_id"] == fc["source"]).all()
    _assert_same_values(_levels(density_only, source="mpr2 (q0.25)"), _stored(density_only, 0.25))
    _assert_same_values(_levels(density_only, source="mpr2 (q0.75)"), _stored(density_only, 0.75))


def test_source_suffix_with_extra_ids(density_two_regions):
    _call(density_two_regions, "0.25", source_suffix=" (q0.25)")

    assert set(density_two_regions.forecasts["unique_id"]) == {"mpr2 (q0.25) + UK", "mpr2 (q0.25) + EW"}


def test_repeat_with_source_suffix_and_extra_ids_is_quiet_noop(density_two_regions):
    _call(density_two_regions, "0.25", source_suffix=" (q0.25)")
    forecasts = density_two_regions.forecasts.copy()
    raw = density_two_regions._raw_forecasts.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        density_two_regions.to_point_forecast("0.25", source_suffix=" (q0.25)")

    assert len(density_two_regions.forecasts) == len(forecasts)
    pd.testing.assert_frame_equal(density_two_regions.forecasts, forecasts)
    pd.testing.assert_frame_equal(density_two_regions._raw_forecasts, raw)


def test_na_in_extra_id_keeps_rows_and_repeat_is_quiet_noop(density_two_regions):
    table = density_two_regions._density_forecasts
    table.loc[table["region"] == "EW", "region"] = pd.NA

    _call(density_two_regions, "0.25")

    fc = density_two_regions.forecasts
    is_levels = fc["metric"] == "levels"
    na_rows = fc[is_levels & fc["region"].isna()].set_index("date")["value"].sort_index()
    uk_rows = fc[is_levels & fc["region"].notna()].set_index("date")["value"].sort_index()
    den = density_two_regions.density_forecasts
    at_level = (den["metric"] == "levels") & np.isclose(den["quantile"], 0.25, rtol=0, atol=1e-9)
    expected_na = den[at_level & den["region"].isna()].set_index("date")["value"].sort_index()
    assert len(na_rows) == 13
    _assert_same_values(na_rows, expected_na)
    _assert_same_values(uk_rows, _stored(density_two_regions, 0.25, region="UK"))

    forecasts = density_two_regions.forecasts.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        density_two_regions.to_point_forecast("0.25")

    pd.testing.assert_frame_equal(density_two_regions.forecasts, forecasts)


def test_vintages_with_different_grids_resolve_a_level_each(density_outturns_df):
    first = pd.Timestamp("2025-12-31")
    second = pd.Timestamp("2025-09-30")
    later = _toy_density([0.1, 0.4, 0.6, 0.9])
    earlier = _toy_density([0.3, 0.55, 0.8]).assign(vintage_date=second)
    earlier["forecast_horizon"] += 1
    data = DensityForecastData(
        outturns_data=density_outturns_df, forecasts_data=pd.concat([later, earlier], ignore_index=True)
    )

    messages = _selection_warnings(_call(data, "median"))

    # 0.5 is equally near 0.4 and 0.6 (lower wins) in one vintage, and nearest to 0.55 in the other.
    _assert_same_values(_levels(data, vintage_date=first), _stored(data, 0.4, vintage_date=first))
    _assert_same_values(_levels(data, vintage_date=second), _stored(data, 0.55, vintage_date=second))
    (message,) = messages
    assert "2 of 2" in message


def test_source_suffix_must_be_a_string(density_only):
    with pytest.raises(ValueError, match="source_suffix"):
        density_only.to_point_forecast("0.25", source_suffix=1)


# -----------------------
# A10: derived growth
# -----------------------
def test_derived_growth_is_growth_of_the_level_path(density_only):
    _call(density_only, "0.25")

    fc = density_only.forecasts
    series = {m: fc[fc["metric"] == m].set_index("date")["value"].sort_index() for m in ("levels", "pop", "yoy")}
    levels = series["levels"]

    pop = levels.pct_change().dropna()
    np.testing.assert_allclose(series["pop"].loc[pop.index], pop)
    yoy = (levels / levels.shift(4) - 1).dropna()
    np.testing.assert_allclose(series["yoy"].loc[yoy.index], yoy)


def test_docstring_states_growth_caveat():
    doc = " ".join(DensityForecastData.to_point_forecast.__doc__.split())

    assert "growth rates of the chosen quantile's LEVEL path, not quantiles of the growth distribution" in doc
    assert "Interpolation between stored levels is not offered" in doc
