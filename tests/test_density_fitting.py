"""Tests for fitting density families to stored quantiles (``fit_parametric_density``)."""

import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from forecast_evaluation.data.DensityForecastData import DensityForecastData
from forecast_evaluation.data.sample_data import create_sample_outturns
from forecast_evaluation.density import AzzaliniSkewT, Normal, TwoPieceNormal, fitting

LEVELS = np.round(np.arange(1, 20) * 0.05, 10)
SPARSE = np.round(np.arange(1, 10) * 0.1, 10)
N_DATES = 6
_i = np.arange(N_DATES)

KNOWN = {
    "normal": (Normal(115 + _i, 0.5 + 0.1 * _i), "all"),
    "two_piece_normal": (TwoPieceNormal(2 + 0.2 * _i, 0.6 + 0.05 * _i, np.full(N_DATES, 1.2)), None),
    "skew_t": (AzzaliniSkewT(1 + 0.1 * _i, np.full(N_DATES, 0.8), 2 + 0.2 * _i, np.where(_i % 2, 8.0, 12.0)), "all"),
}


def _density_data(dist, levels=LEVELS, n_dates=N_DATES) -> DensityForecastData:
    """DensityForecastData with one density per quarter-end date, quantiles from ``dist.ppf``."""
    values = np.atleast_2d(dist.ppf(levels))
    dates = pd.date_range(start="2025-12-31", periods=n_dates, freq="QE")
    rows = [
        {
            "date": date,
            "variable": "gdpkp",
            "vintage_date": pd.Timestamp("2025-12-31"),
            "source": "toy",
            "frequency": "Q",
            "quantile": float(q),
            "value": float(values[i, j]),
            "forecast_horizon": i,
            "metric": "levels",
        }
        for i, date in enumerate(dates)
        for j, q in enumerate(levels)
    ]
    return DensityForecastData(
        outturns_data=create_sample_outturns().assign(metric="levels"),
        forecasts_data=pd.DataFrame(rows),
        data_check=False,
    )


def _fit_quietly(data, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return data.fit_parametric_density(*args, **kwargs)


@pytest.mark.parametrize("family", ["normal", "two_piece_normal", "skew_t", "mpz"])
def test_recovers_known_params(family):
    if family == "mpz":
        dist = Normal(115 + _i, 0.5)
        fitted = _fit_quietly(_density_data(dist), "mpz")
        params = fitted.params.sort_values("date")
        np.testing.assert_allclose(np.vstack(params["values"]), dist.ppf(LEVELS), rtol=0, atol=1e-12)
        assert all(lev == tuple(LEVELS) for lev in params["levels"])
        np.testing.assert_allclose(fitted.cdf(dist.ppf(LEVELS)), np.broadcast_to(LEVELS, (N_DATES, LEVELS.size)))
        return

    dist, match_levels = KNOWN[family]
    fitted = _fit_quietly(_density_data(dist), family, match_levels)
    params = fitted.params.sort_values("date")
    for name in dist.param_names:
        np.testing.assert_allclose(params[name].to_numpy(), getattr(dist, name), rtol=1e-5, atol=1e-7, err_msg=name)
    assert fitted.diagnostics["success"].all()
    assert (fitted.diagnostics["relative_error"] < 1e-4).all()


def test_nearest_level_matching_and_warning():
    data = _density_data(Normal(115 + _i, 0.5), levels=SPARSE)

    with pytest.warns(UserWarning, match="nearest stored level") as record:
        fitted = data.fit_parametric_density("two_piece_normal", match_levels=(0.25, 0.5, 0.77))
    assert len([w for w in record if "nearest stored level" in str(w.message)]) == 1
    # 0.25 ties between 0.2 and 0.3 and resolves to the lower level.
    assert set(fitted.diagnostics["levels_used"]) == {(0.2, 0.5, 0.8)}
    np.testing.assert_allclose(fitted.diagnostics["max_level_distance"], 0.05)

    close = _fit_quietly(data, "two_piece_normal", match_levels=(0.205, 0.5, 0.795))
    assert set(close.diagnostics["levels_used"]) == {(0.2, 0.5, 0.8)}
    np.testing.assert_allclose(close.diagnostics["max_level_distance"], 0.005)


def test_raises_when_too_few_levels():
    bvar = _density_data(Normal(115 + _i, 0.5), levels=[0.16, 0.5, 0.84])
    with pytest.raises(ValueError, match="at least 4 distinct"):
        bvar.fit_parametric_density("skew_t")
    with pytest.raises(ValueError, match="at least 2 distinct"):
        bvar.fit_parametric_density("normal", match_levels=[0.5, 0.52])
    with pytest.raises(ValueError, match="Unknown method"):
        bvar.fit_parametric_density("kde")
    with pytest.raises(TypeError, match="df_grid"):
        bvar.fit_parametric_density("normal", df_grid=range(5, 10))


def test_value_grid_contract_columns():
    loc, scale = 115 + _i, 0.5 + 0.1 * _i
    fitted = _fit_quietly(_density_data(Normal(loc, scale)), "normal")
    assert fitted.keys["date"].is_monotonic_increasing
    key_cols = list(fitted.keys.columns)
    assert "unique_id" in key_cols and "quantile" not in key_cols

    grid = fitted.to_value_grid(n_points=50)
    assert list(grid.columns) == key_cols + ["value", "pdf", "cdf"]
    assert len(grid) == N_DATES * 50
    x = grid["value"].to_numpy().reshape(N_DATES, 50)
    cdf = grid["cdf"].to_numpy().reshape(N_DATES, 50)
    np.testing.assert_allclose(cdf[:, 0], 1e-6, rtol=1e-6)
    np.testing.assert_allclose(cdf[:, -1], 1 - 1e-6, rtol=1e-9)
    assert np.all(np.diff(cdf, axis=1) > 0)
    np.testing.assert_allclose(
        grid["pdf"].to_numpy().reshape(N_DATES, 50), stats.norm.pdf(x, loc[:, None], scale[:, None]), rtol=1e-9
    )

    qgrid = fitted.to_quantile_grid([0.1, 0.5, 0.9])
    assert list(qgrid.columns) == key_cols + ["quantile", "value"]
    np.testing.assert_allclose(
        qgrid["value"].to_numpy().reshape(N_DATES, 3),
        stats.norm.ppf([0.1, 0.5, 0.9], loc[:, None], scale[:, None]),
        rtol=1e-9,
    )

    moments = fitted.moments()
    np.testing.assert_allclose(moments["mean"], loc, rtol=1e-9)
    np.testing.assert_allclose(moments["variance"], scale**2, rtol=1e-9)


def test_from_params_roundtrip():
    fitted = _fit_quietly(_density_data(KNOWN["two_piece_normal"][0]), "two_piece_normal")
    rebuilt = fitting.FittedDensityForecasts.from_params(fitted.params, "two_piece_normal")
    pd.testing.assert_frame_equal(rebuilt.keys, fitted.keys)
    np.testing.assert_allclose(rebuilt.ppf(LEVELS), fitted.ppf(LEVELS))

    mpz = _fit_quietly(_density_data(Normal(115 + _i, 0.5)), "mpz")
    rebuilt = fitting.FittedDensityForecasts.from_params(mpz.params, "mpz")
    np.testing.assert_allclose(rebuilt.cdf([114.0, 116.0]), mpz.cdf([114.0, 116.0]))


def test_approximation_warning():
    skewed = TwoPieceNormal(np.zeros(N_DATES), np.full(N_DATES, 0.3), np.full(N_DATES, 2.0))
    with pytest.warns(UserWarning, match="differ from the stored quantiles"):
        fitted = _density_data(skewed).fit_parametric_density("normal")
    assert (fitted.diagnostics["relative_error"] > 0.05).all()


def test_skewt_fit_speed_smoke():
    n = 40
    i = np.arange(n)
    dist = AzzaliniSkewT(1 + 0.05 * i, 0.5 + 0.01 * i, np.linspace(-4, 4, n), np.full(n, 10.0))
    data = _density_data(dist, n_dates=n)
    fitting._skt_table.cache_clear()
    start = time.perf_counter()
    fitted = _fit_quietly(data, "skew_t")
    elapsed = time.perf_counter() - start
    print(f"skew-t fit of {n} densities (cold table): {elapsed:.2f}s")
    assert fitted.diagnostics["success"].all()
    assert (fitted.diagnostics["relative_error"] < 1e-3).all()
    assert elapsed < 60
