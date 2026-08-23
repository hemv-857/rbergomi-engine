"""Calibrate (xi0, eta, rho, H) to an implied-vol surface by vega-weighted RMSE."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .pricing import smile

__all__ = ["SurfaceQuote", "calibrate", "load_surface_csv"]


@dataclass(frozen=True)
class SurfaceQuote:
    T: float
    K: float
    iv: float
    vega: float = 1.0  # weight; set from market data when available


def load_surface_csv(path: str) -> list[SurfaceQuote]:
    """CSV columns: T,K,iv[,vega]. One row per quote."""
    quotes = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            quotes.append(
                SurfaceQuote(
                    float(row["T"]),
                    float(row["K"]),
                    float(row["iv"]),
                    float(row.get("vega") or 1.0),
                )
            )
    if not quotes:
        raise ValueError(f"no quotes in {path}")
    return quotes


def _model_ivs(params: np.ndarray, s0: float, quotes: list[SurfaceQuote], seed: int) -> np.ndarray:
    xi0, eta, rho, H = params
    out = []
    for T in sorted({q.T for q in quotes}):
        ks = np.array([q.K for q in quotes if q.T == T])
        res = smile(
            ks, s0=s0, xi0=xi0, eta=eta, rho=rho, H=H, T=T,
            n_steps=max(64, int(64 * T)), n_paths=20_000, seed=seed,
        )
        out.extend(res["iv"])
    return np.array(out)


def calibrate(
    quotes: list[SurfaceQuote],
    s0: float,
    seed: int = 7,
    maxiter_de: int = 40,
    verbose: bool = False,
) -> dict:
    """Fit flat forward-variance rBergomi to `quotes`.

    ponytail: flat xi0 and coarse MC budget inside the loss (20k paths).
    Upgrade path: xi0(t) piecewise-flat per expiry + more paths near the optimum.
    """
    targets = np.array([q.iv for q in quotes])
    weights = np.array([q.vega for q in quotes])
    weights = weights / weights.mean()

    def loss(p):
        xi0, _eta, _rho, H = p
        if not (1e-4 < xi0 < 4.0 and 0.05 < H < 0.5):
            return 1e6
        try:
            ivs = _model_ivs(p, s0, quotes, seed)
        except (ValueError, np.linalg.LinAlgError):
            return 1e6
        w_rmse = np.sqrt(np.mean(weights * (ivs - targets) ** 2))
        return w_rmse if np.isfinite(w_rmse) else 1e6

    bounds = [(0.01, 1.0), (0.3, 3.5), (-0.99, -0.05), (0.03, 0.49)]
    de = differential_evolution(loss, bounds, seed=seed, maxiter=maxiter_de, tol=1e-3, polish=False)
    nm = minimize(loss, de.x, method="Nelder-Mead",
                  options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300})

    best = nm.x if nm.fun <= de.fun else de.x
    fitted_ivs = _model_ivs(best, s0, quotes, seed)
    resid = fitted_ivs - targets
    result = {
        "xi0": float(best[0]),
        "eta": float(best[1]),
        "rho": float(best[2]),
        "H": float(best[3]),
        "rmse_iv": float(np.sqrt(np.mean(resid**2))),
        "max_abs_err": float(np.max(np.abs(resid))),
        "n_quotes": len(quotes),
        "fitted_ivs": fitted_ivs.tolist(),
    }
    if verbose:
        print(f"xi0={result['xi0']:.4f} eta={result['eta']:.4f} "
              f"rho={result['rho']:.4f} H={result['H']:.4f}")
        print(f"IV RMSE={result['rmse_iv']:.2e} max|err|={result['max_abs_err']:.2e}")
    return result
