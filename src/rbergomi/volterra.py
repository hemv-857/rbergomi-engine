"""Exact Gaussian simulation of the rough Bergomi Volterra process.

Y_t = sqrt(2H) * int_0^t (t-s)^{H-1/2} dW_s   (alpha = H - 1/2 in (-1/2, 0))

Cov(Y_s, Y_t) = 2H * int_0^{min(s,t)} (s-u)^a (t-u)^a du
Cov(Y_s, W_t) = sqrt(2H) * int_0^{min(s,t)} (s-u)^a du
Var(W_t)      = t

The Y-Y kernel has an algebraic endpoint singularity u -> min(s,t), so plain
Gauss-Legendre converges at O(N^{2a+2}) -- unusable for small H. Substituting
u = m(1+x)/2 turns the singular factor into the Jacobi weight (1-x)^a, which a
Gauss-Jacobi rule absorbs with fast convergence; smooth integrands (Y-W block)
use plain Gauss-Legendre. Diagonal entries have exact closed forms. The stacked
Gaussian vector (Y_1..Y_n, W_1..W_n) is then sampled exactly via one Cholesky.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi

__all__ = ["joint_covariance", "simulate_rbergomi", "simulate_volterra"]


def joint_covariance(times: np.ndarray, H: float, nodes: int = 64) -> np.ndarray:
    """Covariance of the stacked Gaussian vector [Y_{t1..tn}, W_{t1..tn}]."""
    times = np.asarray(times, dtype=float)
    n = times.size
    alpha = H - 0.5
    sqrt2H = np.sqrt(2 * H)

    xj, wj = roots_jacobi(nodes, alpha, 0.0)  # absorbs weight (1-x)^alpha
    xg, wg = leggauss(nodes)  # plain rule for smooth integrands

    def yy_offdiag(s: float, m: float) -> float:
        """int_0^m (s-u)^a (m-u)^a du, s > m."""
        a = 0.5 * m * (xj + 1.0)
        return (0.5 * m) ** (alpha + 1) * np.sum(wj * (s - a) ** alpha)

    def yw_smooth(s: float, m: float) -> float:
        """int_0^m (s-u)^a du, s > m (integrand smooth)."""
        a = 0.5 * m * (xg + 1.0)
        return 0.5 * m * np.sum(wg * (s - a) ** alpha)

    C = np.zeros((2 * n, 2 * n))
    for i, ti in enumerate(times):
        for j in range(i):  # Y-Y off-diagonal
            c = 2 * H * yy_offdiag(ti, times[j])
            C[i, j] = C[j, i] = c
        C[i, i] = ti ** (2 * H)  # Var(Y_t) = t^{2H}, exact
        # Y-W upper-right block: R[i,j] = Cov(Y_i, W_j) = g(t_i, t_j)
        #   g(s,t) = sqrt(2H) int_0^{min(s,t)} (s-u)^a du
        C[i, n : n + i] = [sqrt2H * yw_smooth(ti, tm) for tm in times[:i]]  # min = t_j < t_i
        yw_diag = sqrt2H * ti ** (alpha + 1) / (alpha + 1)  # j >= i: min = t_i -> exact
        C[i, n + i :] = yw_diag
        # Lower-left block is the transpose (covariance matrices are symmetric);
        # g is NOT symmetric in its arguments, so mirror the filled values.
        C[n:, i] = C[i, n:]
        C[n + i, n + i] = ti
    C[n:, n:] = np.minimum.outer(times, times)  # Var/cov of W levels
    return C


def _chol_psd(C: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(C)
        return np.linalg.cholesky(evecs @ np.diag(np.maximum(evals, jitter)) @ evecs.T)


def simulate_volterra(
    T: float,
    n_steps: int,
    H: float,
    n_paths: int,
    seed: int,
    antithetic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly simulate (Y, W) levels on grid t_k = k*T/n_steps, k=1..n_steps."""
    times = np.arange(1, n_steps + 1) * (T / n_steps)
    L = _chol_psd(joint_covariance(times, H))

    half = (n_paths + 1) // 2 if antithetic else n_paths
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((half, 2 * n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])[:n_paths]
    X = Z @ L.T
    return X[:, :n_steps], X[:, n_steps:]


def simulate_rbergomi(
    s0: float,
    xi0: float,
    eta: float,
    rho: float,
    H: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    antithetic: bool = True,
) -> dict[str, np.ndarray]:
    """Simulate rBergomi paths. Returns dict with t, Y, W, v, S (S includes t=0 column)."""
    Y, W = simulate_volterra(T, n_steps, H, n_paths, seed, antithetic)
    dt = T / n_steps
    times = np.arange(1, n_steps + 1) * dt

    # v_t = xi0 * exp(eta*Y_t - 0.5*eta^2 * t^{2H}): with Var(Y_t)=t^{2H} this is
    # an exponential martingale in Y, so E[v_t] = xi0 exactly.
    v = xi0 * np.exp(eta * Y - 0.5 * eta**2 * times[np.newaxis, :] ** (2 * H))
    sigma = np.sqrt(v)

    W0 = np.hstack([np.zeros((W.shape[0], 1)), W])
    dW = np.diff(W0, axis=1)
    rng = np.random.default_rng(seed + 1)  # ponytail: second stream, uncorrelated by construction
    dZ = rng.standard_normal(dW.shape) * np.sqrt(dt)

    # Left-point Euler: coefficient over [t_{k-1}, t_k] must be known at t_{k-1};
    # pairing sigma(t_k) with dW_k leaks future info into the driver and breaks
    # the martingale property (biased spot drift).
    s0_col = np.full((sigma.shape[0], 1), np.sqrt(xi0))
    sig_l = np.hstack([s0_col, sigma[:, :-1]])
    v_l = np.hstack([np.full((v.shape[0], 1), xi0), v[:, :-1]])

    log_ret = sig_l * (rho * dW + np.sqrt(1 - rho**2) * dZ) - 0.5 * v_l * dt
    S = s0 * np.exp(np.cumsum(log_ret, axis=1))
    return {
        "t": times,
        "Y": Y,
        "W": W,
        "v": v,
        "S": np.hstack([np.full((S.shape[0], 1), s0), S]),
    }
