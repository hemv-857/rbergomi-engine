import numpy as np
import pytest

from rbergomi.calibrate import SurfaceQuote, calibrate
from rbergomi.implied_vol import implied_vol
from rbergomi.pricing import black_scholes, price_european, smile
from rbergomi.volterra import joint_covariance, simulate_rbergomi, simulate_volterra


# ---------------------------------------------------------------- covariance
def test_var_y_matches_t_2h():
    """E[Y_t^2] = t^{2H}: quadrature covariance is exact at grid points."""
    times = np.linspace(0.05, 1.0, 12)
    for H in (0.05, 0.15, 0.35):
        C = joint_covariance(times, H)
        assert np.allclose(np.diag(C)[: len(times)], times ** (2 * H), rtol=1e-6)


def test_cov_matrix_is_psd_and_symmetric():
    C = joint_covariance(np.linspace(0.1, 0.9, 20), H=0.1)
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-10


def test_simulated_moments():
    Y, W = simulate_volterra(T=0.5, n_steps=32, H=0.1, n_paths=40_000, seed=1)
    t = np.arange(1, 33) * 0.5 / 32
    emp_var = Y.var(axis=0)
    assert np.allclose(emp_var, t**0.2, rtol=0.06)  # 2H = 0.2
    assert np.allclose(W[:, -1].var(), 0.5, rtol=0.03)  # W_T var == T


def test_antithetic_paths_exact_mirror():
    Y, _ = simulate_volterra(1.0, 16, 0.1, n_paths=100, seed=3)
    assert Y.shape[0] % 2 == 0
    assert np.allclose(Y[:50] + Y[50:], 0)


# ---------------------------------------------------------------- pricing
def test_h_half_recovers_black_scholes():
    """H -> 0.5 must collapse to flat-vol BS within MC error."""
    sigma = float(np.sqrt(0.18))
    bs = black_scholes(1.0, 1.0, 0.5, 0.0, sigma)["call"]
    mc = price_european("call", 1.0, s0=1.0, xi0=0.18, eta=0.0, rho=0.0,
                        H=0.4999, T=0.5, n_steps=16, n_paths=60_000, seed=5)
    # eta=0 kills stochastic vol entirely; remaining roughness error ~ O((0.5-H))
    assert abs(mc["price"] - bs) < 4 * mc["stderr"] + 1e-4


def test_put_call_parity():
    c = price_european("call", 1.1, s0=1.0, xi0=0.18, eta=1.7, rho=-0.9,
                       H=0.08, T=0.5, n_paths=80_000, seed=2)
    p = price_european("put", 1.1, s0=1.0, xi0=0.18, eta=1.7, rho=-0.9,
                       H=0.08, T=0.5, n_paths=80_000, seed=2)
    # parity under zero rates: C - P = S0 - K e^{-rT}; floor guards the
    # near-zero stderr case where parity holds to machine precision
    assert abs((c["price"] - p["price"]) - (1.0 - 1.1)) < max(
        4 * max(c["stderr"], p["stderr"]), 1e-8
    )


def test_control_variate_reduces_variance():
    kw = dict(s0=1.0, xi0=0.18, eta=1.7, rho=-0.9, H=0.08, T=1.0,
              n_steps=64, n_paths=20_000, seed=11)
    with_cv = price_european("call", 1.2, control_variate=True, **kw)
    without = price_european("call", 1.2, control_variate=False, **kw)
    assert with_cv["variance_reduction"] > 1.5  # OTM: terminal-value CV helps less than ATM


def test_control_variate_unbiased():
    kw = dict(s0=1.0, xi0=0.18, eta=1.7, rho=-0.9, H=0.08, T=0.5,
              n_steps=64, n_paths=40_000, seed=13)
    cv = price_european("call", 1.0, control_variate=True, **kw)["price"]
    raw = price_european("call", 1.0, control_variate=False, **kw)
    assert abs(cv - raw["price"]) < 6 * raw["stderr"]


# ---------------------------------------------------------------- implied vol
def test_iv_roundtrip():
    from scipy.stats import norm

    s0, k, T, sig = 100.0, 105.0, 0.75, 0.31
    d1 = (np.log(s0 / k) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    px = s0 * norm.cdf(d1) - k * norm.cdf(d1 - sig * np.sqrt(T))
    assert abs(implied_vol(px, s0, k, T) - sig) < 1e-8


def test_iv_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        implied_vol(200.0, s0=100.0, k=90.0, T=1.0)  # above spot


# ---------------------------------------------------------------- calibration
@pytest.mark.slow
def test_calibration_recovers_synthetic_params():
    true = {"eta": 1.7, "rho": -0.85, "H": 0.12}
    quotes = []
    for T in (0.25, 0.75):
        ks = np.linspace(0.9, 1.1, 5)
        r = smile(ks, s0=1.0, xi0=0.18, T=T, seed=99, n_paths=30_000, **true)
        quotes += [SurfaceQuote(T=T, K=float(k), iv=float(v)) for k, v in zip(ks, r["iv"])]

    fit = calibrate(quotes, s0=1.0, seed=42, maxiter_de=25)
    assert abs(fit["eta"] - true["eta"]) < 0.45
    assert abs(fit["rho"] - true["rho"]) < 0.2
    assert abs(fit["H"] - true["H"]) < 0.08
    assert fit["rmse_iv"] < 0.01


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
