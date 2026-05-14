import numpy as np
from scipy.stats import norm, expon, gumbel_r, logistic, uniform
from scipy.stats import pearson3

# Common τ3 domain applied to every curve
T3_MIN, T3_MAX = -1.0, 1.0


# ── Quadrature helpers ────────────────────────────────────────────────────────

def _gauss_quad(m=400):
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (x + 1.0), 0.5 * w


def _shifted_legendre(u):
    P1 = 2 * u - 1
    P2 = 6 * u ** 2 - 6 * u + 1
    P3 = 20 * u ** 3 - 30 * u ** 2 + 12 * u - 1
    return P1, P2, P3


def _lm_ratios(Q_vals, P1, P2, P3, w):
    l2 = np.sum(Q_vals * P1 * w)
    l3 = np.sum(Q_vals * P2 * w)
    l4 = np.sum(Q_vals * P3 * w)
    if abs(l2) < 1e-12 or not (np.isfinite(l3) and np.isfinite(l4)):
        return None, None
    return l3 / l2, l4 / l2


def _filter_sort(t3s, t4s):
    """Keep finite points with τ3 ∈ [T3_MIN, T3_MAX], sorted by τ3."""
    t3 = np.asarray(t3s, float)
    t4 = np.asarray(t4s, float)
    mask = np.isfinite(t3) & np.isfinite(t4) & (t3 >= T3_MIN) & (t3 <= T3_MAX)
    idx = np.argsort(t3[mask])
    return t3[mask][idx], t4[mask][idx]


# ── PPF-based helper ──────────────────────────────────────────────────────────

def lmr_from_ppf(Q, n=2000):
    """Compute (τ3, τ4) for any distribution given its PPF."""
    u = np.linspace(1e-12, 1 - 1e-12, n)
    q = Q(u)
    w = np.ones(n); w[0] = w[-1] = 0.5; w /= w.sum()
    b0 = np.dot(q, w);        b1 = np.dot(q * u, w)
    b2 = np.dot(q * u**2, w); b3 = np.dot(q * u**3, w)
    L2 = 2*b1 - b0
    L3 = 6*b2 - 6*b1 + b0
    L4 = 20*b3 - 30*b2 + 12*b1 - b0
    return float(L3 / L2), float(L4 / L2)


def _single_tau(ppf_fn, m=400):
    u, w = _gauss_quad(m)
    P1, P2, P3 = _shifted_legendre(u)
    t3, t4 = _lm_ratios(ppf_fn(u), P1, P2, P3, w)
    return float(t3), float(t4)


# ── Single-point theoretical values ──────────────────────────────────────────

def normal_tau(m=400):
    return _single_tau(norm.ppf, m)

def uniform_tau(m=400):
    return _single_tau(uniform.ppf, m)

def exponential_tau(m=400):
    return _single_tau(lambda u: expon.ppf(u, scale=1.0), m)

def logistic_tau(m=400):
    return _single_tau(logistic.ppf, m)

def gumbel_tau(m=400):
    return _single_tau(gumbel_r.ppf, m)


# ── Curves — all filtered to τ3 ∈ [−1, 1] ────────────────────────────────────

def gev_curve():
    """GEV family (Hosking 1990 closed-form).
    k→−1 gives τ3→1; k→+∞ gives τ3→−1.
    """
    k = np.concatenate([
        np.linspace(-0.9999, -0.001,  500),
        np.linspace( 0.001,   30.0, 1500),
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        t3 = 2 * (1 - 3**(-k)) / (1 - 2**(-k)) - 3
        t4 = (1 - 6*2**(-k) + 10*3**(-k) - 5*4**(-k)) / (1 - 2**(-k))
    return _filter_sort(t3, t4)


def glo_curve():
    """Generalised Logistic: τ3 = −ξ spans (−1, 1) exactly."""
    xi = np.linspace(-0.9999, 0.9999, 2000)
    t3 = -xi
    t4 = (1 + 5 * xi**2) / 6
    return _filter_sort(t3, t4)


def gpa_curve():
    """Generalised Pareto family.
    k→−1 gives τ3→1; k→+∞ gives τ3→−1.
    Uses logspace for large k so the negative-τ3 tail is smooth.
    """
    k = np.concatenate([
        np.linspace(-0.9999,  0.0,    1000),        # negative k side
        np.logspace(  -3,     3,      4000),         # 0.001 → 1000 logspaced
    ])
    t3 = (1 - k) / (3 + k)
    t4 = ((1 - k) * (2 - k)) / ((3 + k) * (4 + k))
    return _filter_sort(t3, t4)


def normal3_tau_curve(m=400):
    """3-parameter Normal (Pearson Type III): spans τ3 ∈ (−1, 1).
    Sweeps Fisher skewness γ from −large to +large.
    Symmetric around zero so it covers both sides.
    """
    neg_skew = -np.logspace( 2, -3, 1500)   # −100  →  −0.001
    pos_skew =  np.logspace(-3,  2, 1500)   #  0.001 →  100
    skews = np.concatenate([neg_skew, [0.0], pos_skew])

    u, w = _gauss_quad(m)
    P1, P2, P3 = _shifted_legendre(u)
    t3s, t4s = [], []
    for s in skews:
        try:
            Q = pearson3.ppf(u, skew=s)
            t3, t4 = _lm_ratios(Q, P1, P2, P3, w)
            if t3 is not None:
                t3s.append(t3); t4s.append(t4)
        except Exception:
            pass
    return _filter_sort(t3s, t4s)


def lower_bound_curve():
    """Theoretical lower bound: τ4 = (5τ3² − 1) / 4 for τ3 ∈ [−1, 1]."""
    t3 = np.linspace(-1, 1, 800)
    t4 = (5 * t3**2 - 1) / 4
    return t3, t4


# ── Reference point collection ────────────────────────────────────────────────

def all_theoretical_points():
    return {
        "Normal":      normal_tau(),
        "Uniform":     uniform_tau(),
        "Exponential": exponential_tau(),
        "Logistic":    logistic_tau(),
        "Gumbel":      gumbel_tau(),
    }
