"""CLI: `rbergomi demo` prices a smile; `rbergomi calibrate` fits a surface."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .calibrate import SurfaceQuote, calibrate, load_surface_csv
from .pricing import price_european, smile


def _demo(args: argparse.Namespace) -> None:
    strikes = np.linspace(0.8, 1.2, 9)
    res = smile(
        strikes,
        s0=1.0,
        xi0=args.xi0,
        eta=args.eta,
        rho=-0.9,
        H=args.H,
        T=args.T,
        n_paths=args.paths,
    )
    print(f"rBergomi smile  T={args.T}  xi0={args.xi0}  eta={args.eta}  rho=-0.9  H={args.H}")
    print(f"{'strike':>8} {'call':>10} {'implied vol':>12}")
    for k, p, iv in zip(res["strikes"], res["price"], res["iv"]):
        print(f"{k:8.3f} {p:10.5f} {iv:12.4f}")


def _calibrate_cmd(args: argparse.Namespace) -> None:
    if args.surface:
        quotes = load_surface_csv(args.surface)
    else:
        # self-contained default: synthetic surface generated from known params
        true = {"eta": 1.7, "rho": -0.85, "H": 0.08}
        quotes = []
        for T in (0.25, 0.5, 1.0):
            ks = np.linspace(0.85, 1.15, 7)
            r = smile(ks, s0=1.0, xi0=0.18, seed=123, T=T, n_paths=30_000, **true)
            quotes += [SurfaceQuote(T=T, K=float(k), iv=float(iv))
                       for k, iv in zip(ks, r["iv"])]
        print(f"target: synthetic surface from eta={true['eta']} rho={true['rho']} H={true['H']}")
    out = calibrate(quotes, s0=1.0, verbose=True)
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_ivs")}, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rbergomi", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="price an OTM/ITM call smile")
    d.add_argument("--xi0", type=float, default=0.18)
    d.add_argument("--eta", type=float, default=1.7)
    d.add_argument("--H", type=float, default=0.08)
    d.add_argument("--T", type=float, default=0.5)
    d.add_argument("--paths", type=int, default=50_000)
    d.set_defaults(func=_demo)

    c = sub.add_parser("calibrate", help="fit (xi0, eta, rho, H) to an IV surface")
    c.add_argument("--surface", help="CSV with columns T,K,iv[,vega]; omit for synthetic demo")
    c.set_defaults(func=_calibrate_cmd)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
