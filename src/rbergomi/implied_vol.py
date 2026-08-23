"""Implied volatility: Newton with analytic vega, bracketed Brent fallback."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = ["implied_vol", "implied_vol_batch"]

_MAX_ITER = 100
_TOL = 1e-10


def _bs_call(sigma: float, s0: float, k: float, T: float) -> float:
    if sigma <= 0:
        return max(s0 - k, 0.0)
    d1 = (np.log(s0 / k) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return s0 * norm.cdf(d1) - k * norm.cdf(d2)


def _vega(sigma: float, s0: float, k: float, T: float) -> float:
    d1 = (np.log(s0 / k) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return s0 * norm.pdf(d1) * np.sqrt(T)


def implied_vol(
    price: float, s0: float, k: float, T: float, lo: float = 1e-4, hi: float = 10.0
) -> float:
    """Call IV from price. Newton first; falls back to Brent on the bracket."""
    intrinsic = max(s0 - k, 0.0)
    if not intrinsic <= price <= s0:
        raise ValueError(f"price {price:.6f} outside no-arbitrage bounds [{intrinsic:.6f}, {s0}]")

    sigma = 0.3
    for _ in range(_MAX_ITER):
        diff = _bs_call(sigma, s0, k, T) - price
        v = _vega(sigma, s0, k, T)
        if abs(diff) < _TOL:
            return float(sigma)
        if v < 1e-12:
            break
        step = diff / v
        nxt = sigma - step
        if not lo < nxt < hi:  # left the sane region -> switch to bracketed solve
            break
        sigma = nxt

    f = lambda s: _bs_call(s, s0, k, T) - price
    try:
        return float(brentq(f, lo, hi, xtol=1e-12))
    except ValueError as e:  # pragma: no cover - guarded by bounds check above
        raise ValueError("IV bracket failed; price likely violates arbitrage bounds") from e


def implied_vol_batch(prices: np.ndarray, s0: float, strikes: np.ndarray, T: float) -> np.ndarray:
    return np.array([implied_vol(p, s0, k, T) for p, k in zip(prices, strikes)])
