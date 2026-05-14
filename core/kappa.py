import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


def _pow_safe(x, p):
    return np.power(np.clip(np.asarray(x, float), 1e-300, np.inf), p)


def _A_from_u(u, h):
    u = np.asarray(u, float)
    if np.abs(h) < _EPS:
        return -np.log(u)
    return (1.0 - _pow_safe(u, h)) / h


def kappa_ppf(u, k, h, xi=0.0, alpha=1.0):
    """Kappa distribution percent-point function (quantile function)."""
    u = np.asarray(u, float)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    A = _A_from_u(u, h)
    if np.abs(k) < _EPS:
        return xi - alpha * np.log(A)
    return xi + (alpha / k) * (1.0 - _pow_safe(A, k))


def tau3tau4_kappa(k, h, n=2001):
    """Compute (τ3, τ4) for the kappa distribution with parameters (k, h)."""
    u = np.linspace(1e-10, 1 - 1e-10, n)
    Q = kappa_ppf(u, k, h)
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    b0 = np.dot(Q, w)
    b1 = np.dot(Q * u, w)
    b2 = np.dot(Q * u ** 2, w)
    b3 = np.dot(Q * u ** 3, w)
    L2 = 2 * b1 - b0
    L3 = 6 * b2 - 6 * b1 + b0
    L4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    if np.abs(L2) < _EPS:
        return np.nan, np.nan
    return L3 / L2, L4 / L2


def fit_kappa(t3_ref, t4_ref):
    """Fit kappa (k, h) to minimise distance to (t3_ref, t4_ref) in τ3–τ4 space."""
    def objective(params):
        k, h = params
        if abs(k) >= 0.99 or abs(h) >= 0.99:
            return 1e6
        t3m, t4m = tau3tau4_kappa(k, h)
        if not (np.isfinite(t3m) and np.isfinite(t4m)):
            return 1e6
        return (t3m - t3_ref) ** 2 + (t4m - t4_ref) ** 2

    res = minimize(objective, x0=[0.1, 0.1], bounds=[(-0.99, 0.99), (-0.99, 0.99)])
    return float(res.x[0]), float(res.x[1])


def kappa_curve(h, ks=None, n=2001):
    """Generate τ3–τ4 curve for kappa family with fixed h, varying k."""
    if ks is None:
        ks = np.linspace(0, 0.99, 300)
    t3c, t4c = [], []
    for k in ks:
        t3, t4 = tau3tau4_kappa(k, h, n=n)
        if np.isfinite(t3) and np.isfinite(t4):
            t3c.append(t3)
            t4c.append(t4)
    return np.array(t3c), np.array(t4c)
