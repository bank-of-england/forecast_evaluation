"""Frozen-parameter density families for density-forecast evaluation.

Parameters are scalars or 1-D arrays of equal length N (one row per forecast). Points ``x`` (and
probabilities ``q``) broadcast as follows: scalar parameters keep the shape of ``x``; parameters of
shape (N,) with scalar ``x`` give (N,), and with ``x`` of shape (M,) or (N, M) give (N, M).
NaN parameters are allowed (missing forecasts) and propagate to NaN outputs.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

__all__ = [
    "AzzaliniSkewT",
    "Normal",
    "SemiParametricMPZ",
    "TwoPieceNormal",
    "boe_moments_to_tpn",
    "mom2g",
    "momtopar",
    "stdtogam",
]


class _Distribution:
    """Shared plumbing: parameter storage/validation, broadcasting and moment accessors."""

    param_names: tuple[str, ...] = ()

    def _store(self, **params) -> None:
        arrays = [np.asarray(v, dtype=float) for v in params.values()]
        if any(a.ndim > 1 for a in arrays):
            raise ValueError("Parameters must be scalars or 1-D arrays (one entry per forecast).")
        try:
            arrays = np.broadcast_arrays(*arrays)
        except ValueError as exc:
            raise ValueError("Parameter arrays must have equal length.") from exc
        for name, arr in zip(params, arrays):
            setattr(self, name, np.array(arr))
        self._ndim = arrays[0].ndim

    def _check_positive(self, *names: str) -> None:
        for name in names:
            if np.any(getattr(self, name) <= 0):
                raise ValueError(f"{name} must be > 0.")

    def _align(self, x, *params):
        x = np.asarray(x, dtype=float)
        if self._ndim == 1 and x.ndim > 0:
            if x.ndim > 2:
                raise ValueError("x must be scalar, (M,) or (N, M) when parameters are 1-D.")
            params = tuple(p[:, None] for p in params)
        return np.broadcast_arrays(x, *params)

    @classmethod
    def from_params(cls, params):
        """Build from an (n_params,) or (N, n_params) array with columns in ``param_names`` order."""
        arr = np.asarray(params, dtype=float)
        if arr.ndim not in (1, 2) or arr.shape[-1] != len(cls.param_names):
            raise ValueError(f"params must have shape ({len(cls.param_names)},) or (N, {len(cls.param_names)}).")
        return cls(*np.moveaxis(arr, -1, 0))

    def moments(self):
        """Return ``(mean, var, skew)``; NaN where undefined."""
        raise NotImplementedError

    def mean(self):
        return self.moments()[0]

    def var(self):
        return self.moments()[1]

    def std(self):
        return np.sqrt(self.var())

    def skew(self):
        return self.moments()[2]

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}={getattr(self, n)!r}" for n in self.param_names)
        return f"{type(self).__name__}({parts})"


def _ppf_edges(q, out):
    """Set q=0/1 to -inf/+inf and q outside [0, 1] to NaN (in place)."""
    out[q == 0] = -np.inf
    out[q == 1] = np.inf
    out[(q < 0) | (q > 1)] = np.nan
    return out


# -----------------------
# Normal
# -----------------------


class Normal(_Distribution):
    """Normal(loc, scale); (N,) params against x (M,) or (N, M) give (N, M)."""

    param_names = ("loc", "scale")

    def __init__(self, loc, scale):
        self._store(loc=loc, scale=scale)
        self._check_positive("scale")

    def pdf(self, x):
        x, m, s = self._align(x, self.loc, self.scale)
        return stats.norm.pdf(x, m, s)[()]

    def cdf(self, x):
        x, m, s = self._align(x, self.loc, self.scale)
        return stats.norm.cdf(x, m, s)[()]

    def ppf(self, q):
        q, m, s = self._align(q, self.loc, self.scale)
        return stats.norm.ppf(q, m, s)[()]

    def moments(self):
        skew = np.where(np.isnan(self.loc) | np.isnan(self.scale), np.nan, 0.0)
        return self.loc[()], (self.scale**2)[()], skew[()]


# -----------------------
# Two-piece normal (BoE / Britton et al.)
# -----------------------


def stdtogam(mean, mode, uncertainty):
    """Port of MATLAB ``stdtogam.m``: skew parameter gamma from BoE mean, mode and uncertainty."""
    mean, mode, unc = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (mean, mode, uncertainty)))
    diff = mean - mode
    c = ((diff**2) * np.pi) / (2 * (unc**2))
    # (sqrt(1+2c)-1)/c rewritten as 2/(sqrt(1+2c)+1): equal to MATLAB, but stable as c -> 0 (gamma -> 0).
    mag = np.sqrt(1 - (2 / (np.sqrt(1 + 2 * c) + 1)) ** 2)
    return np.where(diff < 0, -mag, mag)[()]


def mom2g(s, gam):
    """Port of MATLAB ``mom2g.m``: variance of the TPN implied by uncertainty ``s`` and skew ``gam``."""
    s, gam = np.asarray(s, dtype=float), np.asarray(gam, dtype=float)
    alpha = np.sqrt(1 - gam)
    beta = np.sqrt(1 + gam)
    return (((1 - (2 / np.pi)) * (((s / alpha) - (s / beta)) ** 2)) + ((s**2) / (beta * alpha)))[()]


def momtopar(m, mu, v):
    """Port of MATLAB ``momtopar.m``: (mean, mode, variance) -> (mode, sigma1, sigma2)."""
    m, mu, v = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (m, mu, v)))
    parm = mu.copy()
    alpha = (m - mu) * np.sqrt(np.pi / 2)
    disc = np.sqrt(((1 - 4 * (1 - (2 / np.pi))) * (alpha**2)) + 4 * v)
    sig1 = np.where((disc - alpha) > 0, (disc - alpha) / 2, -1 * (disc + alpha) / 2)
    sig2 = alpha + sig1
    return parm[()], sig1[()], sig2[()]


def boe_moments_to_tpn(mean, mode, uncertainty):
    """BoE (mean, mode, uncertainty) -> TPN (mode, sigma1, sigma2) via stdtogam, mom2g, momtopar."""
    gam = stdtogam(mean, mode, uncertainty)
    return momtopar(mean, mode, mom2g(uncertainty, gam))


class TwoPieceNormal(_Distribution):
    """Two-piece normal (mode, sigma1 below, sigma2 above); (N,) params against x (M,)/(N, M) give (N, M).

    Matches MATLAB ``ftp.m`` (pdf) and ``two_part_normal_inverse_cdf.m`` (ppf).
    """

    param_names = ("mode", "sigma1", "sigma2")

    def __init__(self, mode, sigma1, sigma2):
        self._store(mode=mode, sigma1=sigma1, sigma2=sigma2)
        self._check_positive("sigma1", "sigma2")

    @classmethod
    def from_boe_moments(cls, mean, mode, uncertainty):
        """Build from Bank of England fan-chart mean, mode and uncertainty (MATLAB ``momtopar`` route)."""
        return cls(*boe_moments_to_tpn(mean, mode, uncertainty))

    def pdf(self, x):
        x, m, s1, s2 = self._align(x, self.mode, self.sigma1, self.sigma2)
        fac = ((np.pi * ((s1 + s2) ** 2)) / 2) ** (-0.5)
        s = np.where(x < m, s1, s2)
        return (fac * np.exp((-0.5 * ((x - m) ** 2)) / (s**2)))[()]

    def cdf(self, x):
        x, m, s1, s2 = self._align(x, self.mode, self.sigma1, self.sigma2)
        tot = s1 + s2
        lower = 2 * s1 / tot * special.ndtr((x - m) / s1)
        upper = (s1 - s2) / tot + 2 * s2 / tot * special.ndtr((x - m) / s2)
        return np.where(x < m, lower, upper)[()]

    def ppf(self, q):
        q, m, s1, s2 = self._align(q, self.mode, self.sigma1, self.sigma2)
        cutoff = s1 / (s1 + s2)
        low = q <= cutoff
        low_f = np.where(low, q, cutoff)
        high_f = np.where(low, cutoff, q)
        with np.errstate(invalid="ignore"):
            x_low = m + s1 * special.ndtri((s1 + s2) / (2 * s1) * low_f)
            x_high = m + s2 * special.ndtri((high_f - (s1 - s2) / (s1 + s2)) * (s1 + s2) / (2 * s2))
        return _ppf_edges(q, np.where(low, x_low, x_high))[()]

    def moments(self):
        s1, s2 = self.sigma1, self.sigma2
        d = s2 - s1
        mean = self.mode + np.sqrt(2 / np.pi) * d
        var = (1 - 2 / np.pi) * d**2 + s1 * s2
        mu3 = np.sqrt(2 / np.pi) * d * ((4 / np.pi - 1) * d**2 + s1 * s2)
        return mean[()], var[()], (mu3 / var**1.5)[()]


# -----------------------
# Azzalini-Capitanio skew-t
# -----------------------

# Trapezoid rule in w = log(V/2), V ~ chi2_df, for F(z) = E_V[Phi_SN(z sqrt(V/df); shape)].
# The integrand is analytic in |Im w| < pi/2 uniformly in z, so the step from the strip bound
# (1/cos theta)^k exp(-2 pi theta / h) <= exp(-_SKT_LOG_TOL) gives ~1e-14 accuracy at any |z|.
_SKT_LOG_TOL = 34.0
_SKT_THETA = np.linspace(0.02, 1.56, 155)


def _skt_std_pdf(z, alpha, nu):
    finite = np.isfinite(nu)
    nu_f = np.where(finite, nu, 1.0)
    with np.errstate(invalid="ignore"):
        t_part = 2 * stats.t.pdf(z, nu_f) * special.stdtr(nu_f + 1, alpha * z * np.sqrt((1 + nu_f) / (z**2 + nu_f)))
        sn_part = 2 * stats.norm.pdf(z) * special.ndtr(alpha * z)
    return np.where(np.isinf(z), 0.0, np.where(finite, t_part, sn_part))


def _skt_cdf_finite(z, alpha, nu):
    """Skew-t cdf for flat arrays with finite z and finite df > 0."""
    k = nu / 2
    uk, inv = np.unique(k, return_inverse=True)
    step = (2 * np.pi * _SKT_THETA / (_SKT_LOG_TOL - uk[:, None] * np.log(np.cos(_SKT_THETA)))).max(axis=1)
    w_lo = (special.gammaln(uk + 1) - _SKT_LOG_TOL) / uk
    w_hi = np.log(uk + 2 * _SKT_LOG_TOL + 10 * np.sqrt(uk))
    n_nodes = int(np.ceil(((w_hi - w_lo) / step).max())) + 1
    h = ((w_hi - w_lo) / (n_nodes - 1))[inv]
    w_lo = w_lo[inv]
    log_norm = -special.gammaln(k)
    c = z * np.sqrt(2 / nu)

    acc = np.zeros_like(z)
    wsum = np.zeros_like(z)
    for j in range(n_nodes):
        w = w_lo + j * h
        weight = np.exp(k * w - np.exp(w) + log_norm)
        arg = c * np.exp(w / 2)
        acc += weight * (special.ndtr(arg) - 2 * special.owens_t(arg, alpha))
        wsum += weight
    return acc / wsum


def _skt_std_cdf(z, alpha, nu):
    z, alpha, nu = (a.ravel() for a in np.broadcast_arrays(z, alpha, nu))
    out = np.full(z.shape, np.nan)
    out[z == -np.inf] = 0.0
    out[z == np.inf] = 1.0
    ok = np.isfinite(z) & np.isfinite(alpha) & ~np.isnan(nu)
    sn = ok & np.isinf(nu)
    out[sn] = special.ndtr(z[sn]) - 2 * special.owens_t(z[sn], alpha[sn])
    st = ok & ~sn
    if st.any():
        out[st] = _skt_cdf_finite(z[st], alpha[st], nu[st])
    return np.clip(out, 0.0, 1.0)


def _bracketed_newton(q, x0, cdf, pdf, xtol=1e-13, ftol=1e-15, max_iter=200):
    """Vectorised safeguarded Newton for cdf(x) = q; ``cdf(x, idx)``/``pdf(x, idx)`` act on subsets."""
    n = q.size
    lo, hi = x0 - 1.0, x0 + 1.0
    for bound, sign in ((lo, -1.0), (hi, 1.0)):
        idx, width = np.arange(n), np.full(n, 2.0)
        for _ in range(max_iter):
            f = cdf(bound[idx], idx)
            out = f > q[idx] if sign < 0 else f < q[idx]
            if not out.any():
                break
            idx, width = idx[out], width[out] * 2
            if sign < 0:
                hi[idx] = np.minimum(hi[idx], bound[idx])
            else:
                lo[idx] = np.maximum(lo[idx], bound[idx])
            bound[idx] += sign * width

    x = np.clip(x0, lo, hi)
    active = np.arange(n)
    for _ in range(max_iter):
        xa = x[active]
        resid = cdf(xa, active) - q[active]
        below = resid < 0
        lo[active[below]] = xa[below]
        hi[active[~below]] = xa[~below]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = xa - resid / pdf(xa, active)
        mid = 0.5 * (lo[active] + hi[active])
        bad = ~np.isfinite(xn) | (xn <= lo[active]) | (xn >= hi[active])
        xn[bad] = mid[bad]
        x[active] = xn
        scale = 1.0 + np.abs(xn)
        done = (np.abs(xn - xa) <= xtol * scale) | (np.abs(resid) <= ftol) | (hi[active] - lo[active] <= xtol * scale)
        active = active[~done]
        if active.size == 0:
            break
    return x


def _skt_std_ppf(q, alpha, nu):
    q, alpha, nu = (a.ravel() for a in np.broadcast_arrays(q, alpha, nu))
    out = _ppf_edges(q, np.full(q.shape, np.nan))
    ok = (q > 0) & (q < 1) & np.isfinite(alpha) & ~np.isnan(nu)
    if ok.any():
        qa, aa, na = q[ok], alpha[ok], nu[ok]
        finite = np.isfinite(na)
        x0 = np.where(finite, stats.t.ppf(qa, np.where(finite, na, 1.0)), special.ndtri(qa))
        x0 = x0 + 0.8 * aa / np.sqrt(1 + aa**2)
        out[ok] = _bracketed_newton(
            qa,
            x0,
            lambda x, i: _skt_std_cdf(x, aa[i], na[i]),
            lambda x, i: _skt_std_pdf(x, aa[i], na[i]),
        )
    return out


class AzzaliniSkewT(_Distribution):
    """Azzalini-Capitanio skew-t (loc, scale, shape, df); (N,) params against x (M,)/(N, M) give (N, M).

    pdf matches MATLAB ``dskt``; df may be non-integer and ``df=inf`` gives the skew-normal.
    cdf: trapezoid rule over the chi-square mixing variable (accuracy ~1e-14); ppf: bracketed Newton.
    """

    param_names = ("loc", "scale", "shape", "df")

    def __init__(self, loc, scale, shape, df):
        self._store(loc=loc, scale=scale, shape=shape, df=df)
        self._check_positive("scale", "df")

    def pdf(self, x):
        x, xi, om, al, nu = self._align(x, self.loc, self.scale, self.shape, self.df)
        return (_skt_std_pdf((x - xi) / om, al, nu) / om)[()]

    def cdf(self, x):
        x, xi, om, al, nu = self._align(x, self.loc, self.scale, self.shape, self.df)
        return _skt_std_cdf((x - xi) / om, al, nu).reshape(x.shape)[()]

    def ppf(self, q):
        q, xi, om, al, nu = self._align(q, self.loc, self.scale, self.shape, self.df)
        return (xi + om * _skt_std_ppf(q, al, nu).reshape(q.shape))[()]

    def moments(self):
        nu = self.df
        delta = self.shape / np.sqrt(1 + self.shape**2)
        finite = np.isfinite(nu)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            b = np.where(
                finite,
                np.sqrt(nu / np.pi) * np.exp(special.gammaln((nu - 1) / 2) - special.gammaln(nu / 2)),
                np.sqrt(2 / np.pi),
            )
            mu = delta * b
            r2 = np.where(finite, nu / (nu - 2), 1.0)
            k2 = r2 - mu**2
            r3 = np.where(finite, nu * (3 - delta**2) / (nu - 3), 3 - delta**2)
            k3 = mu * (r3 - 3 * r2 + 2 * mu**2)
            mean = np.where(nu > 1, self.loc + self.scale * mu, np.nan)
            var = np.where(nu > 2, self.scale**2 * k2, np.nan)
            skew = np.where(nu > 3, k3 / k2**1.5, np.nan)
        return mean[()], var[()], skew[()]


# -----------------------
# Semi-parametric MPZ (exact version of MATLAB QR_sm)
# -----------------------


class SemiParametricMPZ(_Distribution):
    """Uniform density between stored quantiles with Gaussian tails from the end pairs (MPZ).

    Equals the distribution MATLAB ``QR_sm`` samples from when levels are j/(K+1); for unequal
    spacing each segment gets its stated level mass (QR_sm would weight segments equally).

    ``levels``/``values`` are (K,) shared or (N, K) per forecast; x broadcasts as for the other
    families. Moments (incl. skewness) are closed form: uniform segments plus truncated-normal tails.
    """

    param_names = ("values",)

    def __init__(self, levels, values):
        lev = np.asarray(levels, dtype=float)
        val = np.asarray(values, dtype=float)
        if lev.ndim not in (1, 2) or val.ndim not in (1, 2):
            raise ValueError("levels and values must be (K,) or (N, K).")
        if lev.shape[-1] != val.shape[-1]:
            raise ValueError("levels and values must have the same number of columns K.")
        if lev.shape[-1] < 2:
            raise ValueError("At least two quantile levels are required.")
        if not np.all(np.isfinite(lev)) or np.any(lev <= 0) or np.any(lev >= 1) or np.any(np.diff(lev, axis=-1) <= 0):
            raise ValueError("levels must be finite, strictly increasing and inside (0, 1).")
        try:
            lev, val = np.broadcast_arrays(lev, val)
        except ValueError as exc:
            raise ValueError("levels and values have incompatible shapes.") from exc
        self._ndim = lev.ndim - 1
        self.levels = np.array(lev)
        self.values = np.sort(val, axis=-1)

        L, V = np.atleast_2d(self.levels), np.atleast_2d(self.values)
        z = special.ndtri(L)
        self._L, self._V, self._z = L, V, z
        self._sig_l = ((V[:, 1] - V[:, 0]) / (z[:, 1] - z[:, 0]))[:, None]
        self._mu_l = V[:, :1] - self._sig_l * z[:, :1]
        self._sig_u = ((V[:, -1] - V[:, -2]) / (z[:, -1] - z[:, -2]))[:, None]
        self._mu_u = V[:, -1:] - self._sig_u * z[:, -1:]
        self._valid = np.all(np.isfinite(V), axis=1)[:, None]

    @classmethod
    def from_params(cls, params, levels):
        """Build from quantile values ``params`` ((K,) or (N, K)) at ``levels``."""
        return cls(levels, params)

    def _rows(self, x):
        """Return x as (R, P) aligned with parameter rows (R = 1 if shared) and the output shape."""
        x = np.asarray(x, dtype=float)
        if self._ndim == 0:
            return x.reshape(1, -1), x.shape
        n = self._V.shape[0]
        if x.ndim == 0:
            return np.broadcast_to(x, (n, 1)), (n,)
        if x.ndim > 2:
            raise ValueError("x must be scalar, (M,) or (N, M) when parameters are 2-D.")
        X = np.broadcast_to(np.atleast_2d(x), (n, x.shape[-1]))
        return X, X.shape

    @staticmethod
    def _segment(X, knots):
        """Index j of the segment [knots_j, knots_{j+1}) containing each X (clipped to 0..K-2)."""
        count = (X[..., None] >= knots[:, None, :]).sum(axis=-1)
        return np.clip(count - 1, 0, knots.shape[1] - 2)

    def _pieces(self, X, knots):
        j = self._segment(X, knots)
        take = lambda A, i: np.take_along_axis(A, i, axis=1)  # noqa: E731
        return take(self._V, j), take(self._V, j + 1), take(self._L, j), take(self._L, j + 1)

    def _finish(self, out, shape):
        return np.where(self._valid, out, np.nan).reshape(shape)[()]

    def pdf(self, x):
        X, shape = self._rows(x)
        q0, q1, t0, t1 = self._pieces(X, self._V)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(q1 > q0, (t1 - t0) / (q1 - q0), 0.0)
            lower = np.where(self._sig_l > 0, stats.norm.pdf(X, self._mu_l, self._sig_l), 0.0)
            upper = np.where(self._sig_u > 0, stats.norm.pdf(X, self._mu_u, self._sig_u), 0.0)
        out = np.where(X < self._V[:, :1], lower, out)
        out = np.where(X >= self._V[:, -1:], upper, out)
        return self._finish(out, shape)

    def cdf(self, x):
        X, shape = self._rows(x)
        q0, q1, t0, t1 = self._pieces(X, self._V)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(q1 > q0, (X - q0) / (q1 - q0), 1.0)
            lower = special.ndtr((X - self._mu_l) / self._sig_l)
            upper = special.ndtr((X - self._mu_u) / self._sig_u)
        out = t0 + (t1 - t0) * frac
        out = np.where(X < self._V[:, :1], lower, out)
        out = np.where(X > self._V[:, -1:], upper, out)
        out = np.where(X == self._V[:, -1:], self._L[:, -1:], out)
        return self._finish(out, shape)

    def ppf(self, q):
        Q, shape = self._rows(q)
        q0, q1, t0, t1 = self._pieces(Q, self._L)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = q0 + (Q - t0) / (t1 - t0) * (q1 - q0)
            lower = np.where(self._sig_l > 0, self._mu_l + self._sig_l * special.ndtri(Q), self._V[:, :1])
            upper = np.where(self._sig_u > 0, self._mu_u + self._sig_u * special.ndtri(Q), self._V[:, -1:])
        out = np.where(Q < self._L[:, :1], lower, out)
        out = np.where(Q > self._L[:, -1:], upper, out)
        out = _ppf_edges(Q, np.array(out))
        return self._finish(out, shape)

    def _raw_moments(self, shift):
        """First three moments of X - shift (rows), exact."""
        L, z = self._L, self._z
        a, b, dt = self._V[:, :-1] - shift[:, None], self._V[:, 1:] - shift[:, None], np.diff(L, axis=1)
        m = [
            (dt * (a + b) / 2).sum(axis=1),
            (dt * (a * a + a * b + b * b) / 3).sum(axis=1),
            (dt * (a + b) * (a * a + b * b) / 4).sum(axis=1),
        ]
        zl, zu = z[:, 0], z[:, -1]
        phl, phu = stats.norm.pdf(zl), stats.norm.pdf(zu)
        tails = (
            # (mu, sigma, partial moments E[Z^r; region] for r = 0..3)
            (self._mu_l[:, 0] - shift, self._sig_l[:, 0], (L[:, 0], -phl, L[:, 0] - zl * phl, -(zl**2 + 2) * phl)),
            (
                self._mu_u[:, 0] - shift,
                self._sig_u[:, 0],
                (1 - L[:, -1], phu, (1 - L[:, -1]) + zu * phu, (zu**2 + 2) * phu),
            ),
        )
        for mu, s, (z0, z1, z2, z3) in tails:
            m[0] = m[0] + mu * z0 + s * z1
            m[1] = m[1] + mu**2 * z0 + 2 * mu * s * z1 + s**2 * z2
            m[2] = m[2] + mu**3 * z0 + 3 * mu**2 * s * z1 + 3 * mu * s**2 * z2 + s**3 * z3
        return m

    def moments(self):
        mean = self._raw_moments(np.zeros(self._V.shape[0]))[0]
        _, var, m3 = self._raw_moments(mean)
        valid = self._valid[:, 0]
        out = [np.where(valid, v, np.nan) for v in (mean, var, m3 / var**1.5)]
        return tuple(o[0] if self._ndim == 0 else o for o in out)
