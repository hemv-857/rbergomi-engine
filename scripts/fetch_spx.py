"""Fetch live SPX option quotes from CBOE delayed feed -> calibration surface CSV.

Usage: python scripts/fetch_spx.py [--out data/spx_surface.csv]

Filters: 14 <= DTE <= 180 days, |log-moneyness| <= 0.20, positive two-sided quotes.
Call/put IVs at the same (T, K) are averaged; vega weights come from BS at mid IV.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict

import httpx

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"


def parse_option_symbol(sym: str) -> tuple[str, str, float] | None:
    # e.g. SPXW260918C05000000 / SPX260118P04500000
    if len(sym) < 15 or not sym.startswith("SPX"):
        return None
    date_part = sym[3:9]
    cp = sym[9]
    try:
        strike = int(sym[10:]) / 1000.0
        dt.datetime.strptime(date_part, "%y%m%d")
    except ValueError:
        return None
    return date_part, cp, strike


def bs_vega(s: float, k: float, T: float, sigma: float) -> float:
    import math

    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return s * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T)


def main(out_path: str) -> None:
    r = httpx.get(URL, timeout=30, headers={"User-Agent": "rbergomi-research/0.1"})
    r.raise_for_status()
    data = r.json()["data"]
    spot = float(data["current_price"])
    now = dt.datetime.now(dt.UTC)

    per_tk: dict[tuple[str, float], dict] = defaultdict(lambda: {"ivs": [], "bid_ask_ok": False})
    for opt in data.get("options", []):
        parsed = parse_option_symbol(opt["option"])
        if parsed is None:
            continue
        date_str, _cp, strike = parsed
        iv = opt.get("iv")
        bid, ask = opt.get("bid", 0), opt.get("ask", 0)
        if not iv or iv <= 0 or bid <= 0 or ask <= 0:
            continue
        expiry = dt.datetime.strptime(date_str, "%y%m%d").replace(tzinfo=dt.UTC)
        dte = (expiry - now).total_seconds() / 86400
        if not (14 <= dte <= 180):
            continue
        k_ratio = strike / spot
        if abs(__import__("math").log(k_ratio)) > 0.20:
            continue
        entry = per_tk[(date_str, round(strike, 1))]
        entry["ivs"].append(iv)
        entry.update({"dte": dte, "strike": strike})

    rows = []
    for (date_str, strike), e in sorted(per_tk.items()):
        if len(e["ivs"]) == 0:
            continue
        iv_mid = sum(e["ivs"]) / len(e["ivs"])  # call+put average when both present
        T = e["dte"] / 365.0
        vega = bs_vega(spot, strike, T, iv_mid)
        rows.append({
            "T": round(T, 6), "K": strike, "iv": round(iv_mid, 6),
            "vega": round(vega, 8), "spot": spot,
        })

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["T", "K", "iv", "vega", "spot"])
        w.writeheader()
        w.writerows(rows)
    print(f"spot={spot:.2f}; wrote {len(rows)} quotes -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/spx_surface.csv")
    args = ap.parse_args()
    main(args.out)
