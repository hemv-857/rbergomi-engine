# rbergomi-engine

Simulation, pricing and calibration under the **rough Bergomi** model:

```
v_t = xi0 * exp( eta * Y_t - 0.5 * eta^2 * t^{2H} )
Y_t = sqrt(2H) * ∫_0^t (t-s)^{H-1/2} dW_s        (Riemann–Liouville, alpha = H - 1/2)
dS_t = sqrt(v_t) S_t ( rho dW1 + sqrt(1-rho^2) dW2 )
```

## What's inside

| Module | Contents |
|---|---|
| `volterra.py` | Exact Gaussian simulation of the Volterra process: covariance by Gauss–Legendre quadrature + Cholesky; antithetic support |
| `pricing.py` | European MC pricer with Black–Scholes control variate; analytic BS reference |
| `implied_vol.py` | Newton implied vol with bracketed Brent fallback |
| `calibrate.py` | Vega-weighted IV RMSE fit of `(xi0, eta, rho, H)` — differential evolution global init + Nelder–Mead polish |
| `cli.py` | `rbergomi demo`, `rbergomi calibrate` |

## Quickstart

```bash
pip install -e ".[dev]"
make test          # property tests + calibration roundtrip
make demo          # price a smile under rBergomi
make calibrate     # fit params to a synthetic target surface, print recovered table
```

## Correctness gates (enforced in tests)

- `E[Y_t^2] = t^{2H}` exactly (quadrature covariance is unbiased)
- `H -> 0.5` recovers flat smile / BS prices within MC error
- Put–call parity holds to MC tolerance
- Price -> IV -> price roundtrip is exact
- Calibration on a synthetic surface recovers the true `(eta, rho, H)` inside loose bounds

## Honest scope

- `ponytail:` forward variance is flat (`xi0` const); term structure of `xi0(t)` is the natural next step.
- Market-data ingestion is a CSV loader (`--surface quotes.csv`: `T,K,iv_mid,vega`) — wire your data vendor in one function.
