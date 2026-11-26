"""Path calibration tests: Rossi-Sekhposyan (2019) sup tests.

Ports the MTP MATLAB ``rs_test`` / ``CVfinalbootstrapInoue``
(``size_statistic_h2_shifted_h1rankfix``, ``CVfinalbootstrsuptest_shifted_h1rankfix``,
``closedform_h1_cv_anchored``). One bootstrap is run per test (the MATLAB production script
additionally averages rejections over repeated bootstraps of the same data, which is not done here).
Bootstrap multipliers come from a seeded ``np.random.Generator`` or can be injected (``eta``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from forecast_evaluation.tests.density_calibration import GROUP_COLS, _eval_table, _metadata
from forecast_evaluation.tests.results import TestResult

RS_LEVELS = (0.01, 0.05, 0.10)
_MIN_OBS = 3


def _default_rvec_stats() -> np.ndarray:
    return np.linspace(0, 1, 1000)


def _default_rvec_boot() -> np.ndarray:
    return np.linspace(0, 1, 1001)


def _order_stat(sorted_values: np.ndarray, n: int, prob: float):
    """MATLAB ``x(round(n * prob))`` on ascending ``sorted_values`` (1-based, half away from zero)."""
    return sorted_values[int(np.floor(n * prob + 0.5)) - 1]


def rs_statistics(pit, rvec=None) -> tuple[float, float]:
    """Rossi-Sekhposyan KS and CvM statistics of PITs (NaNs dropped) on the grid ``rvec``."""
    r = _default_rvec_stats() if rvec is None else np.asarray(rvec, dtype=float)
    z = np.asarray(pit, dtype=float)
    z = z[~np.isnan(z)]
    v = ((z[:, None] < r[None, :]) - r[None, :]).sum(axis=0) / np.sqrt(z.size)
    return float(np.max(np.abs(v))), float(np.mean(v**2))


def _block_sums(centered: np.ndarray, el: int) -> np.ndarray:
    cs = np.vstack([np.zeros((1, centered.shape[1])), np.cumsum(centered, axis=0)])
    n_windows = centered.shape[0] - el + 1
    return cs[el : el + n_windows] - cs[:n_windows]


def _multipliers(eta, rng, shape, name):
    if eta is None:
        return np.random.default_rng(rng).standard_normal(shape)
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 2 or eta.shape[0] != shape[0]:
        raise ValueError(f"{name} eta must have {shape[0]} rows (one per bootstrap multiplier); got {eta.shape}.")
    return eta


def rossi_sekhposyan_statistic(
    pit, *, rvec=None, n_boot: int = 200, block_length: int | None = None, rng=None, eta=None
) -> dict:
    """Rossi-Sekhposyan (2019) PIT calibration test with Inoue block-multiplier bootstrap.

    Parameters
    ----------
    pit : array-like
        PITs in time order (NaNs dropped).
    rvec : array-like, optional
        Grid on [0, 1]; default 1000 equally spaced points.
    n_boot : int, optional
        Bootstrap replications (default 200, as in MATLAB); ignored when ``eta`` is given.
    block_length : int, optional
        Default ``floor(T ** (1/3))``.
    rng : int or np.random.Generator, optional
        Seed or generator for the multipliers.
    eta : array-like, optional
        Injected multipliers, shape (T - block_length + 1, n_boot).

    Returns
    -------
    dict
        ``ks``, ``cvm``, ``ks_cv`` and ``cvm_cv`` (critical values at 1%, 5%, 10%), ``ks_reject``,
        ``cvm_reject`` (boolean arrays in the same order), ``block_length`` and ``n_observations``.
    """
    r = _default_rvec_stats() if rvec is None else np.asarray(rvec, dtype=float)
    z = np.asarray(pit, dtype=float)
    z = z[~np.isnan(z)]
    T = z.size
    if T < _MIN_OBS:
        nan3 = np.full(3, np.nan)
        return {
            "ks": np.nan,
            "cvm": np.nan,
            "ks_cv": nan3,
            "cvm_cv": nan3,
            "ks_reject": np.zeros(3, bool),
            "cvm_reject": np.zeros(3, bool),
            "block_length": np.nan,
            "n_observations": T,
        }
    ks, cvm = rs_statistics(z, r)
    el = int(np.floor(T ** (1 / 3))) if block_length is None else int(block_length)
    e = _multipliers(eta, rng, (T - el + 1, int(n_boot)), "Rossi-Sekhposyan")
    B = e.shape[1]

    blocks = _block_sums((z[:, None] <= r[None, :]) - r[None, :], el)
    k_star = (e / np.sqrt(el)).T @ blocks / np.sqrt(T)
    ks_boot = np.sort(np.max(np.abs(k_star), axis=1))
    cvm_boot = np.sort(np.mean(k_star**2, axis=1))
    ks_cv = np.array([_order_stat(ks_boot, B, 1 - a) for a in RS_LEVELS])
    cvm_cv = np.array([_order_stat(cvm_boot, B, 1 - a) for a in RS_LEVELS])
    return {
        "ks": ks,
        "cvm": cvm,
        "ks_cv": ks_cv,
        "cvm_cv": cvm_cv,
        "ks_reject": ks > ks_cv,
        "cvm_reject": cvm > cvm_cv,
        "block_length": el,
        "n_observations": T,
    }


# -----------------------
# Grouped TestResult interfaces
# -----------------------


def rossi_sekhposyan_test(
    fitted,
    k: int | Literal["latest"] | None = None,
    *,
    rvec=None,
    n_boot: int = 200,
    seed=42,
    same_date_range: bool = True,
    data=None,
) -> TestResult:
    """Rossi-Sekhposyan (2019) PIT calibration test per variable, unique_id, metric, frequency, horizon.

    Parameters
    ----------
    seed : int, np.random.Generator or None, optional
        Seed or generator for the bootstrap multipliers. Defaults to 42 for repeatable results;
        pass None for fresh randomness or a shared generator to advance its stream.

    Returns
    -------
    TestResult
        Group columns, ``ks_stat``, ``cvm_stat``, critical values ``ks_cv_1``/``ks_cv_5``/``ks_cv_10``
        (and ``cvm_cv_*``), rejections ``ks_reject_*``/``cvm_reject_*``, ``block_length``,
        ``n_observations``, ``start_date`` and ``end_date``. Groups with fewer than 3 PITs give NaN.
    """
    table, data, k = _eval_table(fitted, k, same_date_range, data)
    gen = np.random.default_rng(seed)
    rows = []
    for key, g in table.groupby(GROUP_COLS, sort=True, dropna=False):
        res = rossi_sekhposyan_statistic(
            g.sort_values("vintage_date_forecast")["pit"].to_numpy(), rvec=rvec, n_boot=n_boot, rng=gen
        )
        row = {**dict(zip(GROUP_COLS, key)), "ks_stat": res["ks"], "cvm_stat": res["cvm"]}
        for name in ("ks", "cvm"):
            for i, level in enumerate(RS_LEVELS):
                pct = int(round(level * 100))
                row[f"{name}_cv_{pct}"] = res[f"{name}_cv"][i]
                row[f"{name}_reject_{pct}"] = bool(res[f"{name}_reject"][i])
        row.update(
            block_length=res["block_length"],
            n_observations=res["n_observations"],
            start_date=g["vintage_date_forecast"].min(),
            end_date=g["vintage_date_forecast"].max(),
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    metadata = _metadata(
        "rossi_sekhposyan_test", fitted, df, k=k, n_boot=n_boot, seed=seed, same_date_range=same_date_range
    )
    return TestResult(df, data.id_columns, metadata)
