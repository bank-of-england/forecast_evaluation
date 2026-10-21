"""Fit density families to stored quantile forecasts.

All parametric fits share one engine: for a family whose quantiles are ``loc + scale * z(tau; theta)``,
(loc, scale) is closed-form least squares given theta, so the fit is a profile search over theta
(TPN: the ratio ``sigma1 / (sigma1 + sigma2)``; skew-t: ``delta = shape / sqrt(1 + shape^2)`` on an
integer df grid). A grid of standardised quantiles (cached for the skew-t) locates the best theta,
which is then refined inside its grid bracket by safeguarded parabolic steps using the exact ppf.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from forecast_evaluation.density.distributions import (
    AzzaliniSkewT,
    Normal,
    SemiParametricMPZ,
    TwoPieceNormal,
    _bracketed_newton,
    _skt_std_cdf,
    _skt_std_pdf,
    _skt_std_ppf,
)

__all__ = ["FittedDensityForecasts", "fit_quantile_arrays", "fit_quantile_densities"]

METHODS = ("normal", "two_piece_normal", "skew_t", "mpz")
_FAMILIES = {"normal": Normal, "two_piece_normal": TwoPieceNormal, "skew_t": AzzaliniSkewT, "mpz": SemiParametricMPZ}
_MIN_LEVELS = {"normal": 2, "two_piece_normal": 3, "skew_t": 4, "mpz": 2}
_DEFAULT_LEVELS = {
    "normal": "all",
    "two_piece_normal": (0.25, 0.5, 0.75),
    "skew_t": (0.05, 0.25, 0.5, 0.75, 0.95),
    "mpz": "all",
}
_SKEW_T_OPTIONS = {"df_grid", "shape_bound", "n_shape_grid", "n_refine_df"}
_LEVEL_WARNING_DISTANCE = 0.01
_GOLD = 0.3819660112501051
_TPN_RATIO_GRID = np.linspace(0.005, 0.995, 199)


def _check_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Use one of {METHODS}.")
    return method


# -----------------------
# Profile least squares engine
# -----------------------


def _ls_rows(Y, Z):
    """Row-wise least squares of Y on loc + scale * Z; sse is inf where scale <= 0 or undefined."""
    Z = np.broadcast_to(Z, Y.shape)
    zm, ym = Z.mean(axis=1), Y.mean(axis=1)
    zc, yc = Z - zm[:, None], Y - ym[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (zc * yc).sum(axis=1) / (zc**2).sum(axis=1)
    loc = ym - scale * zm
    sse = ((Y - loc[:, None] - scale[:, None] * Z) ** 2).sum(axis=1)
    return loc, scale, np.where(scale > 0, sse, np.inf)


def _ls_grid(Y, Zg):
    """Profile sse (n, G) of each row of Y against each candidate row of Zg (G, k)."""
    zc = Zg - Zg.mean(axis=1, keepdims=True)
    yc = Y - Y.mean(axis=1, keepdims=True)
    szy = yc @ zc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        sse = (yc**2).sum(axis=1)[:, None] - szy**2 / (zc**2).sum(axis=1)
    return np.where(szy > 0, np.maximum(sse, 0.0), np.inf)


def _grid_bracket(Y, grid, Zrows, sse, g):
    """Bracket (a, b, c), exact sse values and quantiles Zrows[:, g] around grid index g per row.

    ``Zrows`` is (n, G, k) and ``sse`` (n, G). A minimum at a grid end gives a one-sided bracket
    (b == a or b == c), which ``_refine`` searches by golden steps; invalid rows get zero width.
    """
    G, rows = len(grid), np.arange(len(g))
    gl, gr = np.clip(g - 1, 0, G - 1), np.clip(g + 1, 0, G - 1)
    valid = np.isfinite(sse[rows, g])
    a = np.where(valid, grid[gl], grid[g])
    c = np.where(valid, grid[gr], grid[g])
    f = [_ls_rows(Y, Zrows[rows, i])[2] for i in (gl, g, gr)]
    return a, grid[g], c, f[0], f[1], f[2], Zrows[rows, g]


def _refine(Y, a, b, c, fa, fb, fc, std_q, Zb, tol, max_iter=60):
    """Vectorised safeguarded parabolic minimisation of the profile sse over theta in [a, c].

    ``std_q(theta, rows, x0)`` returns standardised quantiles (len(rows), k); ``x0`` is the current
    best row (a warm start for iterative ppfs). Returns the best theta, sse and quantiles per row.
    """
    a, b, c, fa, fb, fc = (np.array(v, dtype=float) for v in (a, b, c, fa, fb, fc))
    Z = np.array(Zb, dtype=float)
    active = np.flatnonzero((c - a > 2 * tol) & np.isfinite(fb))
    for _ in range(max_iter):
        if active.size == 0:
            break
        A, B, C = a[active], b[active], c[active]
        FA, FB, FC = fa[active], fb[active], fc[active]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = (B - A) ** 2 * (FB - FC) - (B - C) ** 2 * (FB - FA)
            q = (B - A) * (FB - FC) - (B - C) * (FB - FA)
            u = B - 0.5 * p / q
        ok = np.isfinite(u) & (u > A) & (u < C)
        step = ~(ok & (np.abs(u - B) <= tol))
        golden = np.where(B - A > C - B, B - _GOLD * (B - A), B + _GOLD * (C - B))
        u = np.where(ok, u, golden)[step]
        idx = active[step]
        A, B, C, FA, FB, FC = A[step], B[step], C[step], FA[step], FB[step], FC[step]

        Zu = std_q(u, idx, Z[idx])
        fu = _ls_rows(Y[idx], Zu)[2]
        better, left = fu < FB, u < B
        a[idx] = np.where(better, np.where(left, A, B), np.where(left, u, A))
        fa[idx] = np.where(better, np.where(left, FA, FB), np.where(left, fu, FA))
        c[idx] = np.where(better, np.where(left, B, C), np.where(left, C, u))
        fc[idx] = np.where(better, np.where(left, FB, FC), np.where(left, FC, fu))
        b[idx] = np.where(better, u, B)
        fb[idx] = np.where(better, fu, FB)
        Z[idx[better]] = Zu[better]
        active = idx[c[idx] - a[idx] > 2 * tol]
    return b, fb, Z


# -----------------------
# Family fitters: (levels (k,), Y (n, k)) -> (params (n, p), sse (n,))
# -----------------------


def _fit_normal(levels, Y):
    loc, scale, sse = _ls_rows(Y, special.ndtri(levels)[None, :])
    return np.column_stack([loc, scale]), sse


def _tpn_std_q(levels, r):
    """Standardised TPN quantiles (m, k) for ratio r = sigma1 / (sigma1 + sigma2), total scale 1."""
    L, r = np.asarray(levels)[None, :], np.asarray(r, dtype=float)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        low = r * special.ndtri(np.minimum(L / (2 * r), 1.0))
        high = (1 - r) * special.ndtri(np.clip((L - 2 * r + 1) / (2 * (1 - r)), 0.0, 1.0))
    return np.where(L <= r, low, high)


def _fit_tpn(levels, Y):
    grid = _TPN_RATIO_GRID
    Zg = _tpn_std_q(levels, grid)
    sse = _ls_grid(Y, Zg)
    Zrows = np.broadcast_to(Zg, (Y.shape[0], *Zg.shape))
    a, b, c, fa, fb, fc, Zb = _grid_bracket(Y, grid, Zrows, sse, sse.argmin(axis=1))
    r, _, Z = _refine(Y, a, b, c, fa, fb, fc, lambda u, rows, x0: _tpn_std_q(levels, u), Zb, tol=1e-10)
    loc, scale, sse = _ls_rows(Y, Z)
    return np.column_stack([loc, r * scale, (1 - r) * scale]), sse


@functools.lru_cache(maxsize=32)
def _skt_table(levels: tuple, dfs: tuple, deltas: tuple) -> np.ndarray:
    """Standardised skew-t quantiles, shape (len(deltas), len(dfs), len(levels))."""
    d = np.asarray(deltas)
    alpha = (d / np.sqrt(1 - d**2))[:, None, None]
    nu = np.asarray(dfs, dtype=float)[None, :, None]
    q = np.asarray(levels)[None, None, :]
    out = _skt_std_ppf(q, alpha, nu).reshape(len(deltas), len(dfs), len(levels))
    out.setflags(write=False)
    return out


def _skt_std_q(levels, delta, df, x0):
    """Exact standardised skew-t quantiles (m, k) per row (delta, df), Newton warm-started at x0."""
    m, k = x0.shape
    alpha = np.repeat(delta / np.sqrt(1 - delta**2), k)
    nu = np.repeat(df, k)
    q = np.tile(levels, m)
    x = _bracketed_newton(
        q,
        x0.ravel().copy(),
        lambda x, i: _skt_std_cdf(x, alpha[i], nu[i]),
        lambda x, i: _skt_std_pdf(x, alpha[i], nu[i]),
    )
    return x.reshape(m, k)


def _fit_skew_t(levels, Y, df_grid=range(5, 31), shape_bound=30.0, n_shape_grid=41, n_refine_df=None):
    dfs = tuple(float(d) for d in df_grid)
    if not dfs or any(not np.isfinite(d) or d <= 0 for d in dfs):
        raise ValueError("df_grid must be a non-empty sequence of finite values > 0.")
    if not shape_bound > 0:
        raise ValueError("shape_bound must be > 0.")
    if int(n_shape_grid) < 3 or (n_refine_df is not None and int(n_refine_df) < 1):
        raise ValueError("n_shape_grid must be >= 3 and n_refine_df None or >= 1.")

    dmax = shape_bound / np.sqrt(1 + shape_bound**2)
    deltas = np.linspace(-dmax, dmax, int(n_shape_grid))
    T = _skt_table(tuple(levels.tolist()), dfs, tuple(deltas.tolist()))
    G, J, k = T.shape
    n = Y.shape[0]

    sse = _ls_grid(Y, T.reshape(G * J, k)).reshape(n, G, J)
    g_best = sse.argmin(axis=1)
    f_best = np.take_along_axis(sse, g_best[:, None, :], axis=1)[:, 0, :]
    m = J if n_refine_df is None else min(int(n_refine_df), J)
    j_top = np.argsort(f_best, axis=1, kind="stable")[:, :m]

    # Refine the best shape of each of the m best df values per forecast as virtual rows.
    rows = np.repeat(np.arange(n), m)
    j = j_top.ravel()
    Yv = Y[rows]
    bracket = _grid_bracket(Yv, deltas, np.moveaxis(T[:, j, :], 1, 0), sse[rows, :, j], g_best[rows, j])
    dfv = np.asarray(dfs)[j]
    std_q = lambda u, idx, x0: _skt_std_q(levels, u, dfv[idx], x0)  # noqa: E731
    delta, fv, Z = _refine(Yv, *bracket[:6], std_q, bracket[6], tol=1e-8)
    sel = np.arange(n) * m + np.argmin(fv.reshape(n, m), axis=1)
    loc, scale, sse = _ls_rows(Y, Z[sel])
    shape = delta[sel] / np.sqrt(1 - delta[sel] ** 2)
    return np.column_stack([loc, scale, shape, dfv[sel]]), sse


_FITTERS = {"normal": _fit_normal, "two_piece_normal": _fit_tpn, "skew_t": _fit_skew_t}


def fit_quantile_arrays(levels, values, method: str, **options):
    """Fit one density family to quantile values at shared levels.

    Parameters
    ----------
    levels : array-like, shape (k,)
        Distinct quantile levels in (0, 1).
    values : array-like, shape (n, k) or (k,)
        Quantile values, one row per forecast.
    method : {"normal", "two_piece_normal", "skew_t", "mpz"}
        Family to fit. ``"mpz"`` uses the values directly (no fitting).
    **options
        Skew-t only: ``df_grid`` (integer df candidates, default 5..30), ``shape_bound``
        (|shape| bound, default 30), ``n_shape_grid`` (default 41) and ``n_refine_df``
        (refine only this many best grid df values; default None refines all).

    Returns
    -------
    distribution, sse
        Frozen distribution with one row per forecast and the least-squares objective (n,)
        over ``levels``. Rows whose values are missing, constant or decreasing get NaN
        parameters and ``sse = inf``.
    """
    method = _check_method(method)
    lev = np.asarray(levels, dtype=float)
    Y = np.asarray(values, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    if lev.ndim != 1 or Y.ndim != 2 or Y.shape[1] != lev.size:
        raise ValueError("levels must be (k,) and values (n, k).")
    if not np.all(np.isfinite(lev)) or np.any(lev <= 0) or np.any(lev >= 1) or np.any(np.diff(lev) <= 0):
        raise ValueError("levels must be finite, strictly increasing and inside (0, 1).")
    if lev.size < _MIN_LEVELS[method]:
        raise ValueError(f"{method} needs at least {_MIN_LEVELS[method]} distinct quantile levels; got {lev.size}.")
    unknown = set(options) - (_SKEW_T_OPTIONS if method == "skew_t" else set())
    if unknown:
        raise TypeError(f"Unexpected option(s) for method {method!r}: {sorted(unknown)}.")

    if method == "mpz":
        dist = SemiParametricMPZ(lev, Y)
        return dist, np.where(np.all(np.isfinite(Y), axis=1), 0.0, np.inf)

    params, sse = _FITTERS[method](lev, Y, **options)
    params[~np.isfinite(sse)] = np.nan
    return _FAMILIES[method].from_params(params), sse


# -----------------------
# Fitted densities container
# -----------------------


class FittedDensityForecasts:
    """Fitted densities, one row per stored density forecast.

    Row ``i`` of every array output corresponds to row ``i`` of ``keys``/``params``. ``x``
    (and ``q``) broadcast as for the distribution families: scalar -> (N,), (M,) -> (N, M) and
    (N, M) -> (N, M); pass ``x[:, None]`` to evaluate one point per forecast.

    Attributes
    ----------
    keys : pd.DataFrame
        Identifying columns of each density (all density-table columns except quantile/value).
    method : str
        Fitted family.
    diagnostics : pd.DataFrame or None
        Fit diagnostics (``fit_parametric_density`` only): ``levels_used``,
        ``max_level_distance``, ``sse`` (over the levels used), ``max_abs_error`` and
        ``relative_error`` (largest gap to all stored quantiles, absolute and over the fitted
        interquartile range) and ``success``.
    data : DensityForecastData or None
        Source data object (for outturns in evaluation).
    """

    def __init__(self, keys: pd.DataFrame, method: str, blocks: list, *, diagnostics=None, data=None):
        self.keys = keys.reset_index(drop=True)
        self.method = _check_method(method)
        self._blocks = blocks
        self.diagnostics = diagnostics
        self.data = data

    @classmethod
    def from_params(cls, params: pd.DataFrame, method: str, *, data=None) -> FittedDensityForecasts:
        """Build from known parameters (e.g. BoE fan-chart TPN or MATLAB fitted params).

        ``params`` holds the family's parameter columns (``Normal``: loc, scale; ``TwoPieceNormal``:
        mode, sigma1, sigma2; ``AzzaliniSkewT``: loc, scale, shape, df; MPZ: ``levels`` and
        ``values`` sequences per row); every other column is kept as a key.
        """
        method = _check_method(method)
        names = ["levels", "values"] if method == "mpz" else list(_FAMILIES[method].param_names)
        missing = [c for c in names if c not in params.columns]
        if missing:
            raise ValueError(f"params is missing column(s) {missing} for method {method!r}.")
        if params.empty:
            raise ValueError("params is empty.")
        params = params.reset_index(drop=True)
        keys = params.drop(columns=names)
        if method != "mpz":
            dist = _FAMILIES[method].from_params(params[names].to_numpy(dtype=float))
            return cls(keys, method, [(np.arange(len(params)), dist)], data=data)

        by_levels: dict[tuple, list[int]] = {}
        for i, lev in enumerate(params["levels"]):
            by_levels.setdefault(tuple(float(v) for v in lev), []).append(i)
        blocks = []
        for lev, rows in by_levels.items():
            values = np.vstack([np.asarray(params.at[i, "values"], dtype=float) for i in rows])
            blocks.append((np.asarray(rows), SemiParametricMPZ(np.asarray(lev), values)))
        return cls(keys, method, blocks, data=data)

    def __len__(self) -> int:
        return len(self.keys)

    def __repr__(self) -> str:
        return f"FittedDensityForecasts(method={self.method!r}, n_densities={len(self)})"

    @property
    def params(self) -> pd.DataFrame:
        """Keys plus parameter columns (MPZ: ``levels`` and ``values`` tuples)."""
        out = self.keys.copy()
        if self.method != "mpz":
            dist = self._blocks[0][1]
            for name in dist.param_names:
                out[name] = np.atleast_1d(getattr(dist, name))
            return out
        levels, values = [None] * len(self), [None] * len(self)
        for rows, dist in self._blocks:
            lev = tuple(np.atleast_2d(dist.levels)[0].tolist())
            for r, v in zip(rows, np.atleast_2d(dist.values)):
                levels[r], values[r] = lev, tuple(v.tolist())
        return out.assign(levels=levels, values=values)

    def _evaluate(self, name: str, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim > 2 or (x.ndim == 2 and x.shape[0] != len(self)):
            raise ValueError("x must be scalar, (M,) or (N, M) with N the number of densities.")
        out = None
        for rows, dist in self._blocks:
            vals = np.asarray(getattr(dist, name)(x[rows] if x.ndim == 2 else x), dtype=float)
            if out is None:
                out = np.full((len(self),) + vals.shape[1:], np.nan)
            out[rows] = vals
        return out

    def pdf(self, x) -> np.ndarray:
        return self._evaluate("pdf", x)

    def cdf(self, x) -> np.ndarray:
        return self._evaluate("cdf", x)

    def ppf(self, q) -> np.ndarray:
        return self._evaluate("ppf", q)

    def moments(self) -> pd.DataFrame:
        """Keys plus exact ``mean``, ``variance`` and ``skewness`` (NaN where undefined)."""
        cols = {k: np.full(len(self), np.nan) for k in ("mean", "variance", "skewness")}
        for rows, dist in self._blocks:
            for col, val in zip(cols.values(), dist.moments()):
                col[rows] = val
        return self.keys.assign(**cols)

    def _long(self, **columns) -> pd.DataFrame:
        width = next(iter(columns.values())).shape[1]
        out = self.keys.loc[self.keys.index.repeat(width)].reset_index(drop=True)
        for name, arr in columns.items():
            out[name] = np.asarray(arr).ravel()
        return out

    def to_quantile_grid(self, levels: Sequence[float] | None = None) -> pd.DataFrame:
        """Long table of fitted quantiles: keys plus ``quantile`` and ``value``.

        ``levels`` defaults to 0.01, 0.02, ..., 0.99.
        """
        lev = np.round(np.arange(1, 100) / 100, 10) if levels is None else np.asarray(levels, dtype=float).ravel()
        if lev.size == 0 or not np.all((lev > 0) & (lev < 1)):
            raise ValueError("levels must be a non-empty sequence inside (0, 1).")
        values = self.ppf(lev)
        return self._long(quantile=np.broadcast_to(lev, values.shape), value=values)

    def to_value_grid(self, n_points: int = 2000, tail: float = 1e-6) -> pd.DataFrame:
        """Long table on a per-density value grid: keys plus ``value``, ``pdf`` and ``cdf``.

        The grid has ``n_points`` equally spaced values from ``ppf(tail)`` to ``ppf(1 - tail)``.
        """
        if int(n_points) < 2 or not 0 < tail < 0.5:
            raise ValueError("n_points must be >= 2 and tail in (0, 0.5).")
        ends = self.ppf([tail, 1 - tail])
        grid = ends[:, :1] + (ends[:, 1:] - ends[:, :1]) * np.linspace(0, 1, int(n_points))[None, :]
        return self._long(value=grid, pdf=self.pdf(grid), cdf=self.cdf(grid))


# -----------------------
# Fitting stored density tables
# -----------------------


def _resolve_targets(method: str, match_levels) -> np.ndarray | None:
    """Requested levels as a sorted unique array, or None for all stored levels."""
    if match_levels is None:
        match_levels = _DEFAULT_LEVELS[method]
    if isinstance(match_levels, str):
        if match_levels != "all":
            raise ValueError("match_levels must be None, 'all' or a sequence of levels in (0, 1).")
        return None
    arr = np.asarray(match_levels, dtype=float).ravel()
    if arr.size == 0 or not np.all((arr > 0) & (arr < 1)):
        raise ValueError("match_levels must be None, 'all' or a sequence of levels in (0, 1).")
    return np.unique(arr)


def _match_levels(stored: np.ndarray, targets: np.ndarray | None):
    """Nearest stored level per target (ties to the lower level): (used, matched, distance)."""
    if targets is None:
        return stored, stored, np.zeros(stored.size)
    d = np.round(np.abs(stored[None, :] - targets[:, None]), 12)
    idx = d.argmin(axis=1)
    return np.unique(stored[idx]), stored[idx], d[np.arange(targets.size), idx]


def fit_quantile_densities(
    density: pd.DataFrame,
    method: str,
    match_levels: str | Sequence[float] | None = None,
    *,
    approx_tol: float = 0.05,
    data: Any = None,
    **options,
) -> FittedDensityForecasts:
    """Fit a family to every density in a long density table (see ``fit_parametric_density``)."""
    method = _check_method(method)
    targets = _resolve_targets(method, match_levels)
    if density.empty:
        raise ValueError("No density forecasts to fit.")

    density = density.reset_index(drop=True)
    key_cols = [c for c in density.columns if c not in ("quantile", "value")]
    density = density.assign(_gid=density.groupby(key_cols, sort=True, dropna=False).ngroup().to_numpy())
    keys = density.drop_duplicates("_gid").sort_values("_gid")[key_cols].reset_index(drop=True)
    n = len(keys)

    level_sets = density.sort_values("quantile").groupby("_gid")["quantile"].agg(tuple)
    by_set: dict[tuple, list[int]] = {}
    for gid, stored in level_sets.items():
        by_set.setdefault(stored, []).append(gid)

    diag = {
        "levels_used": [None] * n,
        "max_level_distance": np.zeros(n),
        "sse": np.full(n, np.inf),
        "max_abs_error": np.full(n, np.nan),
        "relative_error": np.full(n, np.nan),
    }
    blocks, params, far = [], None, {}
    for stored, gids in by_set.items():
        stored_arr = np.asarray(stored, dtype=float)
        used, matched, dist_to = _match_levels(stored_arr, targets)
        if used.size < _MIN_LEVELS[method]:
            raise ValueError(
                f"{method} needs at least {_MIN_LEVELS[method]} distinct quantile levels, but the requested "
                f"levels match only {used.tolist()} among the stored levels {list(stored)}."
            )
        if targets is not None:
            for t, u, d in zip(targets, matched, dist_to):
                if d > _LEVEL_WARNING_DISTANCE:
                    key = (float(t), float(u), float(d))
                    far[key] = far.get(key, 0) + len(gids)

        gids = np.asarray(gids)
        wide = density[density["_gid"].isin(gids)].pivot(index="_gid", columns="quantile", values="value")
        Y_all = wide.loc[gids].reindex(columns=list(stored)).to_numpy(dtype=float)
        dist, sse = fit_quantile_arrays(used, Y_all[:, np.searchsorted(stored_arr, used)], method, **options)

        if method == "mpz":
            blocks.append((gids, dist))
        else:
            if params is None:
                params = np.full((n, len(dist.param_names)), np.nan)
            params[gids] = np.column_stack([np.atleast_1d(getattr(dist, p)) for p in dist.param_names])

        fitted = dist.ppf(stored_arr)
        iqr = np.diff(dist.ppf([0.25, 0.75]), axis=1)[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            err = np.max(np.abs(fitted - Y_all), axis=1)
            diag["relative_error"][gids] = err / iqr
        diag["max_abs_error"][gids] = err
        diag["sse"][gids] = sse
        diag["max_level_distance"][gids] = dist_to.max()
        for gid in gids:
            diag["levels_used"][gid] = tuple(used.tolist())

    if method != "mpz":
        blocks = [(np.arange(n), _FAMILIES[method].from_params(params))]

    success = np.isfinite(diag["sse"])
    diagnostics = keys.assign(**diag, success=success)
    _warn_fit_issues(method, far, success, diag["relative_error"], approx_tol)
    return FittedDensityForecasts(keys, method, blocks, diagnostics=diagnostics, data=data)


def _warn_fit_issues(method, far, success, relative_error, approx_tol) -> None:
    if far:
        detail = "; ".join(f"{t:g} -> {u:g} (distance {d:.3g}, {k} densities)" for (t, u, d), k in sorted(far.items()))
        warnings.warn(
            f"Requested quantile level(s) not stored; the nearest stored level was used: {detail}.",
            UserWarning,
            stacklevel=3,
        )
    n_failed = int((~success).sum())
    if n_failed:
        warnings.warn(
            f"{n_failed} of {success.size} densities could not be fitted (missing, constant or decreasing "
            "quantile values); their parameters are NaN.",
            UserWarning,
            stacklevel=3,
        )
    with np.errstate(invalid="ignore"):
        poor = relative_error > approx_tol
    if poor.any():
        warnings.warn(
            f"Fitted {method} quantiles differ from the stored quantiles by more than {approx_tol:g} x the fitted "
            f"interquartile range for {int(poor.sum())} of {success.size} densities (largest "
            f"{np.nanmax(relative_error):.3g}); see FittedDensityForecasts.diagnostics.",
            UserWarning,
            stacklevel=3,
        )
