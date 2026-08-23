"""Calibrate rBergomi to the fetched SPX surface; save results JSON + fit plot.

Usage:
    python scripts/fetch_spx.py
    python scripts/calibrate_spx.py            # vega-weighted subsample, full fit
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rbergomi.calibrate import SurfaceQuote, calibrate
from rbergomi.implied_vol import implied_vol
from rbergomi.pricing import smile


def load_quotes(path: str) -> tuple[list[SurfaceQuote], float]:
    quotes, spot = [], None
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            quotes.append(SurfaceQuote(float(row["T"]), float(row["K"]),
                                       float(row["iv"]), float(row["vega"])))
            spot = float(row["spot"])
    if not quotes:
        raise SystemExit(f"no quotes in {path}")
    return quotes, spot


def subsample(quotes: list[SurfaceQuote], spot: float, strikes_per_expiry: int = 8,
              max_expiries: int = 5, k_lo: float = -0.15, k_hi: float = 0.10) -> list[SurfaceQuote]:
    """Spread calibration points across the smile: even quantiles in log-moneyness.

    Vega-clustering collapses everything onto ATM where eta/rho/H are unidentifiable;
    the vega weights still enter the loss, so liquid strikes dominate naturally.
    """
    by_T: dict[float, list[SurfaceQuote]] = defaultdict(list)
    for q in quotes:
        by_T[q.T].append(q)
    expiries = sorted(by_T)
    if len(expiries) > max_expiries:
        idx = np.linspace(0, len(expiries) - 1, max_expiries).round().astype(int)
        chosen = [expiries[i] for i in dict.fromkeys(idx)]
    else:
        chosen = expiries
    out = []
    for T in chosen:
        rows = sorted(by_T[T], key=lambda q: q.K)
        ks = np.log([q.K / spot for q in rows])
        band = [(q, k) for q, k in zip(rows, ks) if k_lo <= k <= k_hi]
        if len(band) < strikes_per_expiry:
            band = list(zip(rows, ks))
        band.sort(key=lambda qk: qk[1])
        kk = np.array([k for _, k in band])
        take_idx = np.linspace(0, len(band) - 1, strikes_per_expiry).round().astype(int)
        out += [band[i][0] for i in take_idx]
    return out


def main() -> None:
    quotes, spot = load_quotes("data/spx_surface.csv")
    sub = subsample(quotes, spot)
    print(f"{len(quotes)} quotes -> {len(sub)} calibration points "
          f"({len({q.T for q in sub})} expiries), spot={spot:.1f}")

    t0 = time.time()
    res = calibrate(sub, s0=spot, maxiter_de=30, verbose=True)
    res["seconds"] = round(time.time() - t0, 1)
    res["n_points"] = len(sub)
    res["spot"] = spot
    res["fetched_from"] = "CBOE delayed quotes (_SPX.json)"
    print(f"fit done in {res['seconds']}s")

    # model smiles on the SAME subsampled grid for the plot
    fig, axes = plt.subplots(1, len({q.T for q in sub}), figsize=(4 * len({q.T for q in sub}), 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, T in zip(axes, sorted({q.T for q in sub})):
        rows = [q for q in sub if q.T == T]
        ks = np.array([np.log(q.K / spot) for q in rows])
        model_iv = []
        for q in rows:
            p = smile(np.array([q.K]), s0=spot, xi0=res["xi0"], eta=res["eta"],
                      rho=res["rho"], H=res["H"], T=T, n_paths=40_000,
                      n_steps=max(64, int(64 * T)), seed=11)["price"][0]
            model_iv.append(implied_vol(p, spot, q.K, T))
        ax.plot(ks * 100, [q.iv for q in rows], "o-", label="market", ms=5)
        ax.plot(ks * 100, model_iv, "s--", label="rBergomi", ms=5)
        ax.set_title(f"T={T*365:.0f}d")
        ax.set_xlabel("log-moneyness %")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("implied vol")
    axes[0].legend()
    fig.suptitle(
        f"rBergomi fit: eta={res['eta']:.2f} rho={res['rho']:.2f} H={res['H']:.3f} "
        f"xi0={res['xi0']:.4f} | IV RMSE={res['rmse_iv']*100:.2f} vol pts")
    fig.tight_layout()
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/spx_calibration.png", dpi=130)

    Path("docs").mkdir(exist_ok=True)
    with open("docs/calibration_spx.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("saved docs/calibration_spx.json + docs/spx_calibration.png")


if __name__ == "__main__":
    main()
