"""Newey-West HAC covariance: port of MATLAB ``nwX.m`` (Bai and Ng, JBES 2005)."""

from __future__ import annotations

import numpy as np


def nw_hac(u, prewhite: bool = False, lags: float = -1) -> tuple[np.ndarray, float]:
    """HAC covariance of the (non-demeaned) columns of ``u``, as MATLAB ``nwX``.

    Parameters
    ----------
    u : array-like, shape (n,) or (n, m)
        Moment conditions; not demeaned.
    prewhite : bool, optional
        Prewhiten with a VAR(1) (no intercept) and recolour. Default False.
    lags : float, optional
        Bartlett truncation lag; -1 or None selects it by the Andrews (1991) AR(1) rule.
        Lags above n/2 are capped at n/2 (possibly non-integer, as in MATLAB).

    Returns
    -------
    hac : np.ndarray, shape (m, m)
    lags : float
        Truncation lag used.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    n, m = u.shape
    beta = np.zeros((m, m))
    if prewhite:
        reg = u[:-1]
        beta = np.linalg.lstsq(reg, u[1:], rcond=None)[0]
        v = u[1:] - reg @ beta
    else:
        v = u

    if lags is None or lags == -1:
        lagged, lead = v[:-1], v[1:]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rho = (lagged * lead).sum(axis=0) / (lagged**2).sum(axis=0)
            resid = lead - lagged * rho
            sigma = (resid**2).sum(axis=0) / v.shape[0]
            top = np.sum(4 * rho**2 * sigma**2 / ((1 - rho) ** 6 * (1 + rho) ** 2))
            bot = np.sum(sigma**2 / (1 - rho) ** 4)
            lags = float(np.ceil(1.1447 * (top / bot * n) ** (1 / 3)))
        # rho == 1 (e.g. a constant series) makes the rule 0/0; MATLAB's lscov lands just below 1 and hits the cap.
        if not np.isfinite(lags):
            lags = n / 2
    lags = min(float(lags), n / 2)

    vcv = v.T @ v / (n - 1)
    i = 1
    while i <= lags:
        w = 1 - i / (lags + 1)
        cov = v[i:].T @ v[:-i] / (n - 1)
        vcv = vcv + w * (cov + cov.T)
        i += 1
    d = np.linalg.inv(np.eye(m) - beta.T)
    return d @ vcv @ d.T, lags
