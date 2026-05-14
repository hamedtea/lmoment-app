import numpy as np
from scipy.stats import rayleigh, norm, rice


def _shifted_legendre(u):
    P0 = np.ones_like(u)
    P1 = 2 * u - 1
    P2 = 6 * u ** 2 - 6 * u + 1
    P3 = 20 * u ** 3 - 30 * u ** 2 + 12 * u - 1
    return P0, P1, P2, P3


def _lmr_gauss(Q_vals, u, w):
    P0, P1, P2, P3 = _shifted_legendre(u)
    l2 = np.sum(Q_vals * P1 * w)
    l3 = np.sum(Q_vals * P2 * w)
    l4 = np.sum(Q_vals * P3 * w)
    return float(l3 / l2), float(l4 / l2)


def _gauss_quad(m=500):
    x, w = np.polynomial.legendre.leggauss(m)
    u = 0.5 * (x + 1.0)
    w = 0.5 * w
    return u, w


def rayleigh_tau(m=500):
    u, w = _gauss_quad(m)
    Q = rayleigh.ppf(u, scale=1.0)
    return _lmr_gauss(Q, u, w)


def normal_tau(m=500):
    u, w = _gauss_quad(m)
    Q = norm.ppf(u)
    return _lmr_gauss(Q, u, w)


def rice_tau(KdB=10.0, m=240):
    K_lin = 10 ** (KdB / 10.0)
    b = np.sqrt(2 * K_lin)
    u, w = _gauss_quad(m)
    Q = rice.ppf(u, b=b, scale=1.0)
    return _lmr_gauss(Q, u, w)


def lognormal_tau_curve(sigmas, m=200, mu=0.0):
    u, w = _gauss_quad(m)
    z = norm.ppf(u)
    P0, P1, P2, P3 = _shifted_legendre(u)
    tau3, tau4 = [], []
    for sigma in sigmas:
        Q = np.exp(mu + sigma * z)
        l2 = np.sum(Q * P1 * w)
        l3 = np.sum(Q * P2 * w)
        l4 = np.sum(Q * P3 * w)
        tau3.append(l3 / l2)
        tau4.append(l4 / l2)
    return np.array(tau3), np.array(tau4)


def gev_curve():
    """GEV τ3–τ4 curve (closed-form)."""
    xi = np.linspace(-0.99, 20, 600)
    with np.errstate(divide="ignore", invalid="ignore"):
        t3 = 2 * (1 - 3 ** (-xi)) / (1 - 2 ** (-xi)) - 3
        t4 = (1 - 6 * 2 ** (-xi) + 10 * 3 ** (-xi) - 5 * 4 ** (-xi)) / (1 - 2 ** (-xi))
    m = np.isfinite(t3) & np.isfinite(t4)
    return t3[m], t4[m]


def glo_curve():
    """Generalised Logistic τ3–τ4 curve (closed-form)."""
    xi = np.linspace(-0.99, 0.99, 600)
    t3 = -xi
    t4 = (1 + 5 * xi ** 2) / 6
    return t3, t4


def gpa_curve():
    """Generalised Pareto τ3–τ4 curve (closed-form)."""
    xi = np.linspace(-0.99, 0.99, 600)
    t3 = (1 - xi) / (3 + xi)
    t4 = ((1 - xi) * (2 - xi)) / ((3 + xi) * (4 + xi))
    return t3, t4


def lower_bound_curve():
    """L-moment lower bound: τ4 = (5τ3² − 1) / 4."""
    t3 = np.linspace(-1, 1, 600)
    t4 = (5 * t3 ** 2 - 1) / 4
    return t3, t4
