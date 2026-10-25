"""Path calibration tests: Rossi-Sekhposyan (2019) and Galvao-Mantoan-Mitchell (GMM) sup tests.

Ports the MTP MATLAB ``rs_test`` / ``CVfinalbootstrapInoue`` and the GMM package
(``size_statistic_h2_shifted_h1rankfix``, ``CVfinalbootstrsuptest_shifted_h1rankfix``,
``closedform_h1_cv_anchored``). One bootstrap is run per test (the MATLAB production script
additionally averages rejections over repeated bootstraps of the same data, which is not done here).
Bootstrap multipliers come from a seeded ``np.random.Generator`` or can be injected (``eta``).
"""

from __future__ import annotations

import functools
from typing import Literal

import numpy as np
import pandas as pd

from forecast_evaluation.tests.density_calibration import GROUP_COLS, _eval_table, _metadata
from forecast_evaluation.tests.results import TestResult

RS_LEVELS = (0.01, 0.05, 0.10)
RS_TABULATED_5PCT = (1.339, 0.460)  # Rossi-Sekhposyan (2019) KS / CvM 5% critical values, horizon 1
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


@functools.lru_cache(maxsize=8)
def _bridge_draws(rvec: tuple, n_sim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(rvec)
    dr = np.diff(r)
    if r[0] != 0 or r[-1] != 1 or np.any(dr <= 0):
        raise ValueError("rvec must be strictly increasing from 0 to 1.")
    gen = np.random.default_rng(seed)
    ks, cvm = np.empty(n_sim), np.empty(n_sim)
    chunk = 5000
    for start in range(0, n_sim, chunk):
        m = min(chunk, n_sim - start)
        steps = gen.standard_normal((m, dr.size)) * np.sqrt(dr)
        w = np.concatenate([np.zeros((m, 1)), np.cumsum(steps, axis=1)], axis=1)
        bridge = w - r[None, :] * w[:, -1:]
        ks[start : start + m] = np.max(np.abs(bridge), axis=1)
        cvm[start : start + m] = np.mean(bridge**2, axis=1)
    return np.sort(ks), np.sort(cvm)


def brownian_bridge_h1_draws(
    rvec=None, n_sim: int = 200_000, seed: int = 987654, anchor: tuple[float, float] | None = RS_TABULATED_5PCT
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted draws of sup|B| and mean(B^2) for a Brownian bridge on ``rvec`` (horizon-1 null).

    With ``anchor`` (default R&S 2019 tabulated KS 1.339, CvM 0.460) the draws are rescaled so their
    own 95th percentiles (``x[ceil(0.95 n)]``) equal the anchor, as MATLAB ``closedform_h1_cv_anchored``.
    """
    r = _default_rvec_boot() if rvec is None else np.asarray(rvec, dtype=float)
    ks, cvm = _bridge_draws(tuple(r.tolist()), int(n_sim), int(seed))
    if anchor is not None:
        i95 = int(np.ceil(n_sim * 0.95)) - 1
        ks, cvm = ks * (anchor[0] / ks[i95]), cvm * (anchor[1] / cvm[i95])
    return ks, cvm


def gmm_weights(n_horizons: int) -> tuple[np.ndarray, np.ndarray]:
    """GMM (decreasing, increasing) horizon weights, linear in the horizon index and summing to one."""
    H = int(n_horizons)
    increasing = (1 / H) / ((H + 1) / 2) * np.arange(1, H + 1)
    return increasing[::-1].copy(), increasing


def _resolve_weights(weights, H):
    if isinstance(weights, str):
        decreasing, increasing = gmm_weights(H)
        table = {"unweighted": np.ones(H), "decreasing": decreasing, "increasing": increasing}
        if weights not in table:
            raise ValueError("weights must be 'unweighted', 'decreasing', 'increasing' or an array.")
        return table[weights]
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != H or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError(f"weights array must have {H} finite non-negative entries.")
    return w


def galvao_mantoan_mitchell_statistic(
    pits,
    *,
    weights="unweighted",
    alpha: float = 0.05,
    rvec_stats=None,
    rvec=None,
    n_boot: int = 200,
    block_length: int | None = None,
    rng=None,
    eta=None,
    h1_draws: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """GMM sup test of joint PIT calibration across horizons (columns of ``pits``, first = horizon 1).

    Per-horizon R&S statistics (on ``rvec_stats``) are combined by ``max_h w_h * stat_h``. Critical
    values come from a block-multiplier bootstrap (on ``rvec``) whose horizons share index-shifted
    multipliers; the first horizon's bootstrap column is rank-mapped onto Brownian-bridge draws
    (``h1_draws``, default ``brownian_bridge_h1_draws(rvec)``).

    Returns
    -------
    dict
        ``ks``/``cvm`` (per horizon), ``weights``, ``ks_stat``/``cvm_stat`` (sup), ``ks_cv``/``cvm_cv``
        (sup critical values), ``ks_cv_bonf``/``cvm_cv_bonf`` and ``ks_cv_h``/``cvm_cv_h`` (per horizon),
        ``ks_cv_bonf_used``/``cvm_cv_bonf_used`` (the Bonferroni value compared with the statistic),
        ``ks_reject``/``cvm_reject`` and ``ks_reject_bonf``/``cvm_reject_bonf``, ``block_length``.
    """
    Z = np.asarray(pits, dtype=float)
    if Z.ndim != 2 or np.isnan(Z).any():
        raise ValueError("pits must be a complete (T, H) array.")
    T, H = Z.shape
    w = _resolve_weights(weights, H)
    rb = _default_rvec_boot() if rvec is None else np.asarray(rvec, dtype=float)
    stats_ = np.array([rs_statistics(Z[:, h], rvec_stats) for h in range(H)])
    ks, cvm = stats_[:, 0], stats_[:, 1]

    el = int(np.floor(T ** (1 / 3))) if block_length is None else int(block_length)
    n_windows = T - el + 1
    e = _multipliers(eta, rng, (n_windows + H, int(n_boot)), "GMM")
    B = e.shape[1]
    shared = e / np.sqrt(el)

    ks_boot, cvm_boot = np.empty((B, H)), np.empty((B, H))
    for h in range(H):
        emp = (Z[:, h : h + 1] <= rb[None, :]).astype(float)
        k_star = shared[h + 1 : h + 1 + n_windows].T @ _block_sums(emp - emp.mean(axis=0), el) / np.sqrt(T)
        ks_boot[:, h] = np.max(np.abs(k_star), axis=1)
        cvm_boot[:, h] = np.mean(k_star**2, axis=1)

    k_sim, c_sim = brownian_bridge_h1_draws(rb) if h1_draws is None else h1_draws
    for col, sim in ((ks_boot, np.asarray(k_sim, float)), (cvm_boot, np.asarray(c_sim, float))):
        ranks = np.empty(B)
        ranks[np.argsort(col[:, 0], kind="stable")] = np.arange(1, B + 1)
        idx = np.clip(np.floor(ranks / B * sim.size + 0.5), 1, sim.size).astype(int) - 1
        col[:, 0] = sim[idx]

    out = {"ks": ks, "cvm": cvm, "weights": w, "block_length": el, "n_observations": T}
    bonf_p = 1 - alpha / H
    for name, stat, boot in (("ks", ks, ks_boot), ("cvm", cvm, cvm_boot)):
        sorted_boot = np.sort(boot, axis=0)
        cv_h = _order_stat(sorted_boot, B, 1 - alpha)
        cv_bonf = _order_stat(sorted_boot, B, bonf_p)
        cv_sup = _order_stat(np.sort(np.max(boot * w, axis=1)), B, 1 - alpha)
        sup = float(np.max(stat * w))
        if isinstance(weights, str) and weights == "unweighted":
            h_max = int(np.argmax(stat))
            cv_bonf_used = cv_bonf[h_max]
            reject_bonf = stat[h_max] > cv_bonf_used
        else:
            cv_bonf_used = np.max(cv_bonf * w)
            reject_bonf = sup > cv_bonf_used
        out.update(
            {
                f"{name}_stat": sup,
                f"{name}_cv": float(cv_sup),
                f"{name}_cv_h": cv_h,
                f"{name}_cv_bonf": cv_bonf,
                f"{name}_cv_bonf_used": float(cv_bonf_used),
                f"{name}_reject": bool(sup > cv_sup),
                f"{name}_reject_bonf": bool(reject_bonf),
            }
        )
    return out


# -----------------------
# Grouped TestResult interfaces
# -----------------------


def rossi_sekhposyan_test(
    fitted,
    k: int | Literal["latest"] | None = None,
    *,
    rvec=None,
    n_boot: int = 200,
    seed=None,
    same_date_range: bool = True,
    data=None,
) -> TestResult:
    """Rossi-Sekhposyan (2019) PIT calibration test per variable, unique_id, metric, frequency, horizon.

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


def galvao_mantoan_mitchell_test(
    fitted,
    k: int | Literal["latest"] | None = None,
    *,
    weights="unweighted",
    bonferroni: bool = False,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed=None,
    n_sim_h1: int = 200_000,
    same_date_range: bool = True,
    data=None,
) -> TestResult:
    """Galvao-Mantoan-Mitchell joint (path) calibration test across horizons.

    For each variable, unique_id, metric and frequency, PITs are arranged by forecast vintage (rows)
    and horizon (columns, ascending; the first is treated as the one-step horizon); only vintages
    with a PIT at every horizon are used.

    Parameters
    ----------
    weights : {"unweighted", "decreasing", "increasing"} or array-like
        Horizon weights for the sup statistic; GMM linear weights sum to one.
    bonferroni : bool, optional
        Decide with Bonferroni-corrected per-horizon critical values instead of the joint bootstrap.

    Returns
    -------
    TestResult
        Columns: ``variable``, ``unique_id``, ``metric``, ``frequency``, ``horizons``, ``ks_stat``,
        ``ks_cv``, ``ks_reject``, ``cvm_stat``, ``cvm_cv``, ``cvm_reject``, ``block_length``,
        ``n_observations``, ``start_date``, ``end_date``. With ``bonferroni`` the ``*_cv`` columns hold
        the Bonferroni critical value used: at the arg-max horizon (unweighted) or ``max_h w_h cv_h``.
    """
    table, data, k = _eval_table(fitted, k, same_date_range, data)
    gen = np.random.default_rng(seed)
    path_cols = GROUP_COLS[:-1]
    rows = []
    for key, g in table.groupby(path_cols, sort=True, dropna=False):
        wide = g.pivot_table(index="vintage_date_forecast", columns="horizon", values="pit").dropna()
        horizons = tuple(int(h) for h in wide.columns)
        row = {**dict(zip(path_cols, key)), "horizons": horizons}
        if len(wide) < _MIN_OBS:
            rows.append({**row, "n_observations": len(wide)})
            continue
        rb = _default_rvec_boot()
        res = galvao_mantoan_mitchell_statistic(
            wide.to_numpy(),
            weights=weights,
            alpha=alpha,
            n_boot=n_boot,
            rng=gen,
            h1_draws=brownian_bridge_h1_draws(rb, n_sim=n_sim_h1),
        )
        for name in ("ks", "cvm"):
            suffix = "_bonf" if bonferroni else ""
            cv = res[f"{name}_cv_bonf_used"] if bonferroni else res[f"{name}_cv"]
            row.update(
                {f"{name}_stat": res[f"{name}_stat"], f"{name}_cv": cv, f"{name}_reject": res[f"{name}_reject{suffix}"]}
            )
        row.update(
            block_length=res["block_length"],
            n_observations=res["n_observations"],
            start_date=wide.index.min(),
            end_date=wide.index.max(),
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    metadata = _metadata(
        "galvao_mantoan_mitchell_test",
        fitted,
        df,
        k=k,
        weights=weights if isinstance(weights, str) else list(np.asarray(weights, float)),
        bonferroni=bonferroni,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        same_date_range=same_date_range,
    )
    return TestResult(df, data.id_columns, metadata)
