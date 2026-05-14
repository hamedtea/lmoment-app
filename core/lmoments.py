import numpy as np


def lmoments_columns(X):
    """Compute λ1–λ4, τ3, τ4 per column using unbiased PWM estimators (Hosking)."""
    n, d = X.shape
    xs = np.sort(X, axis=0)
    k = np.arange(1, n + 1)[:, None].astype(float)

    w0 = np.ones_like(k)
    w1 = (k - 1) / (n - 1)
    w2 = ((k - 1) * (k - 2)) / ((n - 1) * (n - 2)) if n >= 3 else np.zeros_like(k)
    w3 = ((k - 1) * (k - 2) * (k - 3)) / ((n - 1) * (n - 2) * (n - 3)) if n >= 4 else np.zeros_like(k)

    b0 = (w0 * xs).sum(axis=0) / n
    b1 = (w1 * xs).sum(axis=0) / n
    b2 = (w2 * xs).sum(axis=0) / n
    b3 = (w3 * xs).sum(axis=0) / n

    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    eps = np.finfo(float).eps
    safe_l2 = np.where(np.abs(l2) < eps, np.nan, l2)
    tau3 = l3 / safe_l2
    tau4 = l4 / safe_l2

    return l1, l2, l3, l4, tau3, tau4


def bootstrap_tau(X, B=100, seed=1):
    """Bootstrap mean τ3 and τ4 across columns for B resamples."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    T3, T4 = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        _, _, _, _, t3, t4 = lmoments_columns(X[idx])
        T3.append(np.nanmean(t3))
        T4.append(np.nanmean(t4))
    return np.array(T3), np.array(T4)


def lmr_from_ppf(Q, n=2000):
    """Compute (τ3, τ4) for a distribution given its PPF via numerical integration."""
    u = np.linspace(1e-12, 1 - 1e-12, n)
    q = Q(u)
    w = np.ones_like(u)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    b0 = np.dot(q, w)
    b1 = np.dot(q * u, w)
    b2 = np.dot(q * u ** 2, w)
    b3 = np.dot(q * u ** 3, w)
    L2 = 2 * b1 - b0
    L3 = 6 * b2 - 6 * b1 + b0
    L4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return L3 / L2, L4 / L2
