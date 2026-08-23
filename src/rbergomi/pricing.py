"""European option pricing under rBergomi via MC with BS control variate."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .volterra import simulate_rbergomi

__all__ = ["black_scholes", "price_european", "smile"]


def black_scholes(s0: float, k: float, T: float, r: float, sigma: float) -> dict[str, float]:
    """Analytic European call/put/vega (r = 0 default usage; kept explicit)."""
    if T <= 0 or sigma <= 0:
        call = max(s0 - k, 0.0)
        return {"call": call, "put": call + k * np.exp(-r * T) - s0, "vega": 0.0}
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    call = s0 * norm.cdf(d1) - k * disc * norm.cdf(d2)
    put = k * disc * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    vega = s0 * norm.pdf(d1) * np.sqrt(T)
    return {"call": call, "put": put, "vega": vega}


def price_european(
    kind: str,
    strike: float,
    *,
    s0: float,
    xi0: float,
    eta: float,
    rho: float,
    H: float,
    T: float,
    n_steps: int = 128,
    n_paths: int = 50_000,
    seed: int = 42,
    control_variate: bool = True,
) -> dict[str, float]:
    """MC price of a European call/put with antithetics + optional BS control variate.

    The control variate prices the same payoff assuming constant vol equal to the
    terminal expected variance sqrt(xi0); its expectation is analytic, its sample
    error is highly correlated with the rBergomi sample error.
    """
    paths = simulate_rbergomi(s0, xi0, eta, rho, H, T, n_steps, n_paths, seed)["S"][:, -1]
    st = paths
    if kind == "call":
        payoff = np.maximum(st - strike, 0.0)
    elif kind == "put":
        payoff = np.maximum(strike - st, 0.0)
    else:
        raise ValueError(f"kind must be call|put, got {kind!r}")

    raw = float(np.mean(payoff))
    var_raw = float(np.var(payoff) / n_paths)

    if not control_variate:
        return {"price": raw, "stderr": var_raw**0.5, "n_paths": n_paths}

    if not control_variate:
        return {"price": raw, "stderr": var_raw**0.5, "n_paths": n_paths}

    # Terminal value as control variate: E[S_T] = S0 under the pricing measure
    # (exact), and S_T is strongly correlated with any vanilla payoff.
    beta = float(np.cov(payoff, st, ddof=1)[0, 1] / np.var(st, ddof=1))
    adjusted = payoff - beta * (st - s0)
    return {
        "price": float(np.mean(adjusted)),
        "stderr": float(np.std(adjusted, ddof=1) / np.sqrt(n_paths)),
        "raw_price": raw,
        "raw_stderr": var_raw**0.5,
        "variance_reduction": var_raw / max(float(np.var(adjusted) / n_paths), 1e-300),
        "n_paths": n_paths,
    }


def smile(
    strikes: np.ndarray,
    *,
    s0: float,
    xi0: float,
    eta: float,
    rho: float,
    H: float,
    T: float,
    n_steps: int = 128,
    n_paths: int = 50_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Model implied-vol smile at one expiry (same paths reused across strikes)."""
    from .implied_vol import implied_vol

    paths = simulate_rbergomi(s0, xi0, eta, rho, H, T, n_steps, n_paths, seed)["S"][:, -1]
    ivs, prices = [], []
    for k in np.atleast_1d(strikes):
        p = price_from_paths(paths, k)
        prices.append(p)
        ivs.append(implied_vol(p, s0, k, T))
    return {"strikes": np.atleast_1d(strikes), "iv": np.array(ivs), "price": np.array(prices)}


def price_from_paths(st: np.ndarray, strike: float) -> float:
    return float(np.mean(np.maximum(st - strike, 0.0)))
