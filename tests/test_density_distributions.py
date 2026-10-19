"""Unit tests for the frozen-parameter density families in ``forecast_evaluation.density``."""

import numpy as np
import pytest
from scipy import integrate, special, stats

from forecast_evaluation.density import AzzaliniSkewT, Normal, SemiParametricMPZ, TwoPieceNormal
from forecast_evaluation.density.distributions import boe_moments_to_tpn, mom2g, stdtogam

MPZ_LEVELS = np.array([0.05, 0.1, 0.3, 0.5, 0.9, 0.95])
MPZ_VALUES = np.array([-1.5, -1.0, -0.2, 0.3, 1.4, 2.1])

FAMILIES = {
    "normal": lambda: Normal(0.5, 1.3),
    "tpn": lambda: TwoPieceNormal(1.0, 0.6, 1.4),
    "tpn_left": lambda: TwoPieceNormal(-2.0, 1.5, 0.4),
    "skewt": lambda: AzzaliniSkewT(0.3, 1.2, 3.0, 6.0),
    "skewt_neg": lambda: AzzaliniSkewT(-1.0, 0.7, -2.0, 10.0),
    "skewt_nonint_df": lambda: AzzaliniSkewT(0.0, 1.0, 1.5, 7.5),
    "mpz": lambda: SemiParametricMPZ(MPZ_LEVELS, MPZ_VALUES),
}


def _breakpoints(dist):
    if isinstance(dist, SemiParametricMPZ):
        return list(dist.values)
    if isinstance(dist, TwoPieceNormal):
        return [float(dist.mode)]
    return []


def _integrate(dist, func):
    """Integrate ``func`` over the real line, splitting at the 0.1%/99.9% quantiles and kinks."""
    a, b = float(dist.ppf(1e-3)), float(dist.ppf(1 - 1e-3))
    pts = [p for p in _breakpoints(dist) if a < p < b] or None
    opts = {"epsabs": 1e-13, "epsrel": 1e-12, "limit": 400}
    return (
        integrate.quad(func, -np.inf, a, **opts)[0]
        + integrate.quad(func, a, b, points=pts, **opts)[0]
        + integrate.quad(func, b, np.inf, **opts)[0]
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_ppf_cdf_roundtrip(family):
    dist = FAMILIES[family]()
    q = np.array([1e-4, 1e-3, 0.01, 0.07, 0.2, 0.37, 0.5, 0.63, 0.8, 0.93, 0.99, 0.999, 1 - 1e-4])
    np.testing.assert_allclose(dist.cdf(dist.ppf(q)), q, rtol=0, atol=1e-9)

    x = np.linspace(dist.ppf(1e-3), dist.ppf(1 - 1e-3), 23)
    np.testing.assert_allclose(dist.ppf(dist.cdf(x)), x, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("family", FAMILIES)
def test_pdf_integrates_to_one(family):
    dist = FAMILIES[family]()
    assert abs(_integrate(dist, lambda t: float(dist.pdf(t))) - 1.0) < 1e-8


@pytest.mark.parametrize("family", FAMILIES)
def test_moments_match_numeric(family):
    dist = FAMILIES[family]()
    mean, var, skew = dist.moments()
    num_mean = _integrate(dist, lambda t: t * float(dist.pdf(t)))
    num_var = _integrate(dist, lambda t: (t - num_mean) ** 2 * float(dist.pdf(t)))
    num_m3 = _integrate(dist, lambda t: (t - num_mean) ** 3 * float(dist.pdf(t)))

    assert abs(mean - num_mean) < 1e-7
    assert abs(var - num_var) < 1e-7
    assert abs(skew - num_m3 / num_var**1.5) < 1e-6
    assert dist.mean() == pytest.approx(mean, abs=0)
    assert dist.std() == pytest.approx(np.sqrt(var), abs=0)
    assert dist.skew() == pytest.approx(skew, abs=0)


def test_ppf_edges():
    for dist in (factory() for factory in FAMILIES.values()):
        out = dist.ppf([0.0, 1.0, -0.1, 1.1])
        assert out[0] == -np.inf and out[1] == np.inf
        assert np.isnan(out[2:]).all()


# -----------------------
# Normal / two-piece normal
# -----------------------


def test_normal_matches_scipy():
    x = np.linspace(-4, 6, 11)
    dist = Normal(1.0, 2.0)
    np.testing.assert_allclose(dist.pdf(x), stats.norm.pdf(x, 1.0, 2.0), rtol=1e-15)
    np.testing.assert_allclose(dist.cdf(x), stats.norm.cdf(x, 1.0, 2.0), rtol=1e-15)
    assert dist.moments() == (1.0, 4.0, 0.0)


def test_tpn_equal_sigmas_is_normal():
    x = np.linspace(-5, 7, 25)
    q = np.linspace(0.01, 0.99, 13)
    tpn, norm = TwoPieceNormal(1.0, 1.7, 1.7), Normal(1.0, 1.7)
    np.testing.assert_allclose(tpn.pdf(x), norm.pdf(x), rtol=1e-13)
    np.testing.assert_allclose(tpn.cdf(x), norm.cdf(x), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(tpn.ppf(q), norm.ppf(q), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(tpn.moments(), norm.moments(), atol=1e-14)


def test_tpn_mass_below_mode():
    tpn = TwoPieceNormal(0.5, 0.8, 1.6)
    assert tpn.cdf(0.5) == pytest.approx(0.8 / 2.4, abs=1e-15)
    assert tpn.ppf(0.8 / 2.4) == pytest.approx(0.5, abs=1e-14)


def test_tpn_from_boe_moments():
    sym = TwoPieceNormal.from_boe_moments(2.0, 2.0, 1.3)
    assert float(sym.mode) == 2.0
    assert float(sym.sigma1) == pytest.approx(1.3, rel=1e-14)
    assert float(sym.sigma2) == pytest.approx(1.3, rel=1e-14)
    assert stdtogam(2.0, 2.0, 1.3) == 0.0
    assert abs(stdtogam(0.1 + 0.2, 0.3, 1.0)) < 1e-6

    mean = np.array([2.5, 1.5, 0.3, -1.0])
    mode = np.array([2.0, 2.0, 0.1, -0.5])
    unc = np.array([1.0, 0.8, 2.0, 1.5])
    tpn = TwoPieceNormal.from_boe_moments(mean, mode, unc)
    np.testing.assert_allclose(tpn.mode, mode, rtol=0)
    np.testing.assert_allclose(tpn.mean(), mean, rtol=1e-12)
    np.testing.assert_allclose(tpn.var(), mom2g(unc, stdtogam(mean, mode, unc)), rtol=1e-12)
    assert np.all((tpn.sigma2 > tpn.sigma1) == (mean > mode))

    m, s1, s2 = boe_moments_to_tpn(mean, mode, unc)
    np.testing.assert_array_equal(m, tpn.mode)
    np.testing.assert_array_equal(s1, tpn.sigma1)
    np.testing.assert_array_equal(s2, tpn.sigma2)


# -----------------------
# Skew-t
# -----------------------


@pytest.mark.parametrize("df", [2.0, 5.0, 10.0, 30.0])
def test_skewt_shape_zero_is_student_t(df):
    x = np.linspace(-30, 30, 41)
    dist = AzzaliniSkewT(0.4, 1.7, 0.0, df)
    np.testing.assert_allclose(dist.pdf(x), stats.t.pdf(x, df, 0.4, 1.7), rtol=1e-12)
    np.testing.assert_allclose(dist.cdf(x), stats.t.cdf(x, df, 0.4, 1.7), rtol=0, atol=1e-12)


def _skewt_cdf_quad(z, shape, df):
    log_c = special.gammaln((df + 1) / 2) - special.gammaln(df / 2) - 0.5 * np.log(df * np.pi)

    def pdf(t):
        tail = special.stdtr(df + 1, shape * t * np.sqrt((df + 1) / (t * t + df)))
        return 2 * np.exp(log_c - (df + 1) / 2 * np.log1p(t * t / df)) * tail

    opts = {"epsabs": 1e-14, "epsrel": 1e-13, "limit": 500}
    if z <= 0:
        return integrate.quad(pdf, -np.inf, z, **opts)[0]
    return 1.0 - integrate.quad(pdf, z, np.inf, **opts)[0]


def test_skewt_cdf_matches_quad():
    dfs, shapes = [2.0, 5.0, 10.0, 30.0], [-10.0, -3.0, 0.0, 2.0, 10.0]
    z = np.array([-50.0, -5.0, -1.0, -0.1, 0.0, 0.3, 2.0, 8.0, 50.0])
    grid = np.array([(s, d) for d in dfs for s in shapes])
    dist = AzzaliniSkewT(0.0, 1.0, grid[:, 0], grid[:, 1])
    got = dist.cdf(z)
    ref = np.array([[_skewt_cdf_quad(zz, s, d) for zz in z] for s, d in grid])
    err = np.abs(got - ref).max()
    assert err <= 1e-10, f"max |cdf - quad| = {err:.3e}"
    np.testing.assert_allclose(got[:, 4], 0.5 - np.arctan(grid[:, 0]) / np.pi, atol=1e-13)


def test_skewt_cdf_limits_and_monotone():
    dist = AzzaliniSkewT(0.0, 1.0, 4.0, 3.0)
    x = np.concatenate([[-np.inf], np.linspace(-20, 20, 201), [np.inf]])
    cdf = dist.cdf(x)
    assert cdf[0] == 0.0 and cdf[-1] == 1.0
    assert np.all(np.diff(cdf) >= 0)


def test_skewt_infinite_df_is_skew_normal():
    x = np.linspace(-4, 4, 17)
    dist = AzzaliniSkewT(0.2, 1.3, 2.5, np.inf)
    np.testing.assert_allclose(dist.pdf(x), stats.skewnorm.pdf(x, 2.5, 0.2, 1.3), rtol=1e-12)
    np.testing.assert_allclose(dist.cdf(x), stats.skewnorm.cdf(x, 2.5, 0.2, 1.3), atol=1e-12)
    mean, var, skew = dist.moments()
    s_mean, s_var, s_skew = stats.skewnorm.stats(2.5, 0.2, 1.3, moments="mvs")
    np.testing.assert_allclose([mean, var, skew], [s_mean, s_var, s_skew], rtol=1e-12)


def test_skewt_moments_undefined_for_low_df():
    mean, var, skew = AzzaliniSkewT([0.0, 0.0, 0.0], 1.0, 2.0, [1.0, 2.0, 3.0]).moments()
    assert np.isnan(mean[0]) and np.isfinite(mean[1:]).all()
    assert np.isnan(var[:2]).all() and np.isfinite(var[2])
    assert np.isnan(skew).all()


# -----------------------
# Semi-parametric MPZ
# -----------------------


def test_mpz_unequal_spacing_masses():
    dist = SemiParametricMPZ(MPZ_LEVELS, MPZ_VALUES)
    np.testing.assert_array_equal(dist.cdf(MPZ_VALUES), MPZ_LEVELS)
    np.testing.assert_allclose(np.diff(dist.cdf(MPZ_VALUES)), np.diff(MPZ_LEVELS), rtol=0, atol=1e-15)
    for j in range(len(MPZ_LEVELS) - 1):
        mass = integrate.quad(lambda t: float(dist.pdf(t)), MPZ_VALUES[j], MPZ_VALUES[j + 1])[0]
        assert mass == pytest.approx(MPZ_LEVELS[j + 1] - MPZ_LEVELS[j], abs=1e-13)
        mid = 0.5 * (MPZ_VALUES[j] + MPZ_VALUES[j + 1])
        assert dist.pdf(mid) == pytest.approx(np.diff(MPZ_LEVELS)[j] / np.diff(MPZ_VALUES)[j], rel=1e-14)


def test_mpz_gaussian_tails():
    dist = SemiParametricMPZ(MPZ_LEVELS, MPZ_VALUES)
    z = special.ndtri(MPZ_LEVELS)
    sig_l = (MPZ_VALUES[1] - MPZ_VALUES[0]) / (z[1] - z[0])
    mu_l = MPZ_VALUES[0] - sig_l * z[0]
    sig_u = (MPZ_VALUES[-1] - MPZ_VALUES[-2]) / (z[-1] - z[-2])
    mu_u = MPZ_VALUES[-1] - sig_u * z[-1]
    lo, hi = np.array([-6.0, -3.0, -1.6]), np.array([2.2, 3.5, 8.0])
    np.testing.assert_allclose(dist.cdf(lo), stats.norm.cdf(lo, mu_l, sig_l), rtol=1e-14)
    np.testing.assert_allclose(dist.cdf(hi), stats.norm.cdf(hi, mu_u, sig_u), rtol=1e-14)
    np.testing.assert_allclose(dist.pdf(lo), stats.norm.pdf(lo, mu_l, sig_l), rtol=1e-14)
    np.testing.assert_allclose(dist.pdf(hi), stats.norm.pdf(hi, mu_u, sig_u), rtol=1e-14)
    q = np.array([1e-6, 0.01, 0.04])
    np.testing.assert_allclose(dist.ppf(q), mu_l + sig_l * special.ndtri(q), rtol=1e-14)


def test_mpz_sorts_values_and_handles_ties():
    dist = SemiParametricMPZ([0.1, 0.25, 0.5, 0.75, 0.9], [2.0, -1.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(dist.values, [-1.0, 0.0, 0.0, 1.0, 2.0])
    assert dist.cdf(0.0) == 0.5  # point mass of 0.25 at the tie, right-continuous cdf
    assert dist.cdf(-1e-12) == pytest.approx(0.25, abs=1e-11)
    q = np.array([0.01, 0.2, 0.3, 0.45, 0.6, 0.8, 0.95])
    x = dist.ppf(q)
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) >= 0)
    np.testing.assert_array_equal(dist.ppf([0.3, 0.45]), [0.0, 0.0])
    np.testing.assert_allclose(dist.cdf(x[[0, 1, 4, 5, 6]]), q[[0, 1, 4, 5, 6]], atol=1e-14)
    assert np.isfinite(dist.pdf(np.linspace(-3, 3, 13))).all()


def test_mpz_per_row_parameters():
    values = np.array([MPZ_VALUES, MPZ_VALUES * 2 + 1])
    dist = SemiParametricMPZ(MPZ_LEVELS, values)
    x = np.linspace(-3, 4, 9)
    assert dist.cdf(x).shape == (2, 9)
    np.testing.assert_array_equal(dist.cdf(values), np.tile(MPZ_LEVELS, (2, 1)))
    single = SemiParametricMPZ(MPZ_LEVELS, values[1])
    np.testing.assert_allclose(dist.cdf(x)[1], single.cdf(x), rtol=1e-15)
    np.testing.assert_allclose(dist.moments()[0][1], single.mean(), rtol=1e-14)
    assert SemiParametricMPZ.from_params(values, MPZ_LEVELS).cdf(x).shape == (2, 9)


# -----------------------
# Broadcasting, constructors and validation
# -----------------------

ROW_FACTORIES = {
    "normal": (Normal, np.array([[0.0, 1.0], [1.0, 2.0], [-1.0, 0.5]])),
    "tpn": (TwoPieceNormal, np.array([[0.0, 1.0, 1.0], [1.0, 0.5, 2.0], [-1.0, 1.5, 0.7]])),
    "skewt": (AzzaliniSkewT, np.array([[0.0, 1.0, 0.0, 5.0], [1.0, 2.0, 3.0, 10.0], [-1.0, 0.5, -2.0, 4.0]])),
}


@pytest.mark.parametrize("family", ROW_FACTORIES)
def test_broadcasting_rows_match_scalar(family):
    cls, params = ROW_FACTORIES[family]
    dist = cls.from_params(params)
    x = np.linspace(-3, 3, 7)
    q = np.linspace(0.05, 0.95, 5)
    for method, arg in (("pdf", x), ("cdf", x), ("ppf", q)):
        out = getattr(dist, method)(arg)
        assert out.shape == (3, arg.size)
        np.testing.assert_allclose(getattr(dist, method)(np.tile(arg, (3, 1))), out, rtol=1e-14)
        np.testing.assert_allclose(getattr(dist, method)(arg[2]), out[:, 2], rtol=1e-14)
        for i, row in enumerate(params):
            np.testing.assert_allclose(getattr(cls(*row), method)(arg), out[i], rtol=1e-13, atol=1e-15)
    assert dist.moments()[0].shape == (3,)


def test_scalar_params_keep_x_shape():
    for dist in (factory() for factory in FAMILIES.values()):
        assert dist.cdf(np.zeros(5)).shape == (5,)
        assert dist.pdf(np.zeros((2, 3))).shape == (2, 3)
        assert np.ndim(dist.cdf(0.0)) == 0


def test_from_params_matches_constructor():
    p = np.array([0.5, 1.5, -2.0, 6.0])
    a, b = AzzaliniSkewT.from_params(p), AzzaliniSkewT(*p)
    np.testing.assert_array_equal(a.cdf([-1.0, 0.0, 2.0]), b.cdf([-1.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        AzzaliniSkewT.from_params(np.ones((3, 3)))


@pytest.mark.parametrize(
    "build",
    [
        lambda: Normal(0.0, 0.0),
        lambda: Normal(0.0, [1.0, -1.0]),
        lambda: Normal([0.0, 1.0], [1.0, 1.0, 1.0]),
        lambda: Normal(np.zeros((2, 2)), 1.0),
        lambda: TwoPieceNormal(0.0, -1.0, 1.0),
        lambda: TwoPieceNormal(0.0, 1.0, 0.0),
        lambda: AzzaliniSkewT(0.0, -1.0, 0.0, 5.0),
        lambda: AzzaliniSkewT(0.0, 1.0, 0.0, 0.0),
        lambda: SemiParametricMPZ([0.5], [1.0]),
        lambda: SemiParametricMPZ([0.5, 0.25], [0.0, 1.0]),
        lambda: SemiParametricMPZ([0.0, 0.5], [0.0, 1.0]),
        lambda: SemiParametricMPZ([0.5, 1.0], [0.0, 1.0]),
        lambda: SemiParametricMPZ([0.25, 0.5], [0.0, 1.0, 2.0]),
    ],
)
def test_invalid_params_raise(build):
    with pytest.raises(ValueError):
        build()
