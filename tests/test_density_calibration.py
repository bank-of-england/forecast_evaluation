"""Tests for density calibration: evaluation table, PIT, HAC, Knueppel and Mitchell-Weale."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from forecast_evaluation.data.DensityForecastData import DensityForecastData
from forecast_evaluation.tests import build_density_eval_table, compute_pit, knuppel_test, mitchell_weale_test
from forecast_evaluation.tests._hac import nw_hac
from forecast_evaluation.tests.density_calibration import knuppel_statistic, mitchell_weale_statistic

LEVELS = np.round(np.arange(1, 20) * 0.05, 10)
QE = pd.offsets.QuarterEnd


def _make_data(n=200, scale=1.0, horizons=(0,), sources=("a",), n_vintages=1, seed=0) -> DensityForecastData:
    """Outturns y ~ 100 + N(0, 1) (vintage j adds 0.5 (j - 1)); densities N(100, scale) for every date."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1980-03-31", periods=n, freq="QE")
    y = 100 + rng.standard_normal(n)
    outturns = pd.DataFrame(
        [
            {
                "date": d,
                "vintage_date": d + QE(j),
                "variable": "x",
                "frequency": "Q",
                "metric": "levels",
                "value": y[i] + 0.5 * (j - 1),
            }
            for j in range(1, n_vintages + 1)
            for i, d in enumerate(dates)
        ]
    )
    q = stats.norm.ppf(LEVELS, 100, scale)
    forecasts = pd.DataFrame(
        [
            {
                "date": d,
                "vintage_date": d - QE(h) if h else d,
                "variable": "x",
                "frequency": "Q",
                "metric": "levels",
                "source": src,
                "forecast_horizon": h,
                "quantile": lev,
                "value": val,
            }
            for src in sources
            for h in horizons
            for d in dates
            for lev, val in zip(LEVELS, q)
        ]
    )
    return DensityForecastData(outturns_data=outturns, forecasts_data=forecasts, data_check=False)


def test_nw_hac_zero_lags_is_plain_covariance():
    u = np.random.default_rng(1).standard_normal((50, 2))
    hac, lags = nw_hac(u, lags=0)
    np.testing.assert_allclose(hac, u.T @ u / 49)
    assert lags == 0
    _, auto = nw_hac(u)
    assert 1 <= auto <= 25
    # A constant series makes the Andrews rule 0/0; the lag falls back to the n/2 cap.
    assert nw_hac(np.full(120, 0.1))[1] == 60


def test_eval_table_respects_k_and_latest():
    data = _make_data(n=8, n_vintages=3)
    fitted = data.fit_parametric_density("normal")
    base = build_density_eval_table(fitted, k=0)["outturn"].to_numpy()
    assert len(base) == 8
    for k, shift in ((1, 0.5), (2, 1.0), ("latest", 1.0), (12, 1.0), (None, 1.0)):
        table = build_density_eval_table(fitted, k=k)
        np.testing.assert_allclose(table["outturn"].to_numpy(), base + shift, err_msg=str(k))
        np.testing.assert_allclose(table["pit"], stats.norm.cdf(table["outturn"], 100, 1), rtol=1e-9)
    with pytest.raises(ValueError, match="k must be"):
        build_density_eval_table(fitted, k="first")


def test_uniform_pits_do_not_reject():
    fitted = _make_data().fit_parametric_density("normal")
    pit = compute_pit(fitted)
    assert len(pit) == 200 and pit["pit"].between(0, 1).all()
    assert knuppel_test(fitted)["p_value"].iloc[0] > 0.01
    assert mitchell_weale_test(fitted)["p_value"].iloc[0] > 0.01


def test_shifted_density_rejects():
    fitted = _make_data(scale=0.5).fit_parametric_density("normal")
    assert knuppel_test(fitted)["p_value"].iloc[0] < 1e-3
    assert mitchell_weale_test(fitted)["p_value"].iloc[0] < 1e-3


def test_grouping_and_testresult_metadata():
    fitted = _make_data(n=40, horizons=(0, 1), sources=("a", "b")).fit_parametric_density("normal")
    result = knuppel_test(fitted, lags=2)
    df = result.to_df()
    assert len(df) == 4
    assert set(df["horizon"]) == {0, 1} and set(df["source"]) == {"a", "b"}
    assert {"stat", "p_value", "n_observations", "start_date", "end_date"} <= set(df.columns)
    assert result._metadata["test_name"] == "knuppel_test"
    assert result._metadata["parameters"]["k"] == 12
    assert result._metadata["parameters"]["lags"] == 2

    mw = mitchell_weale_test(fitted, k="latest")
    assert mw._metadata["test_name"] == "mitchell_weale_test"
    assert mw._metadata["parameters"]["alpha"] == 0.10
    assert {"stat_uncensored", "stat_frequency"} <= set(mw.to_df().columns)


def test_array_statistics_handle_nan_and_short_samples():
    pit = np.random.default_rng(2).uniform(size=60)
    with_nan = np.concatenate([pit, [np.nan]])
    assert knuppel_statistic(with_nan) == knuppel_statistic(pit)
    assert np.isnan(knuppel_statistic(pit[:2])[0])
    assert np.isnan(mitchell_weale_statistic(pit[:2])["stat"])
