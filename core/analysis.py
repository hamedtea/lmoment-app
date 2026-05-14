import numpy as np
from functools import lru_cache

from core.lmoments import lmoments_columns, bootstrap_tau
from core.ordinary_moments import ordinary_moments_columns
from core.kappa import fit_kappa, tau3tau4_kappa, kappa_curve
from core.distributions import gev_curve, glo_curve, gpa_curve, lower_bound_curve


@lru_cache(maxsize=1)
def ref_curves():
    """Theoretical reference curves — computed once and cached."""
    t3_gev, t4_gev = gev_curve()
    t3_glo, t4_glo = glo_curve()
    t3_gpa, t4_gpa = gpa_curve()
    t3_lb,  t4_lb  = lower_bound_curve()
    return {
        "GEV":         (t3_gev, t4_gev),
        "GLO":         (t3_glo, t4_glo),
        "GPA":         (t3_gpa, t4_gpa),
        "Lower bound": (t3_lb,  t4_lb),
    }


def run_analysis(X, B=50):
    """Run full L-moment analysis on matrix X."""
    om = ordinary_moments_columns(X)
    l1, l2, l3, l4, tau3, tau4 = lmoments_columns(X)
    t3_boot, t4_boot = bootstrap_tau(X, B=B)
    t3_mean = float(np.nanmean(tau3))
    t4_mean = float(np.nanmean(tau4))
    k_fit, h_fit = fit_kappa(t3_mean, t4_mean)
    t3_kappa, t4_kappa = tau3tau4_kappa(k_fit, h_fit)
    t3_kappa_curve, t4_kappa_curve = kappa_curve(h_fit)
    return {
        "X": X,
        "om": om,
        "lm": {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "tau3": tau3, "tau4": tau4},
        "boot": {"t3": t3_boot, "t4": t4_boot},
        "kappa": {"k": k_fit, "h": h_fit,
                  "t3": t3_kappa, "t4": t4_kappa,
                  "curve_t3": t3_kappa_curve, "curve_t4": t4_kappa_curve},
        "means": {"t3": t3_mean, "t4": t4_mean},
        "ref_curves": ref_curves(),
    }
