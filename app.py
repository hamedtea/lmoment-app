import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import rice as rice_dist, rayleigh as rayleigh_dist

from core.lmoments import lmoments_columns, bootstrap_tau
from core.ordinary_moments import ordinary_moments_columns
from core.kappa import fit_kappa, tau3tau4_kappa, kappa_curve
from core.distributions import (
    rayleigh_tau, normal_tau, rice_tau, lognormal_tau_curve,
    gev_curve, glo_curve, gpa_curve, lower_bound_curve,
)

st.set_page_config(page_title="L-Moment Analyser", layout="wide")
st.title("L-Moment Analyser")
st.markdown(
    "A tool for computing ordinary moments and L-moments of a data matrix, "
    "fitting the Kappa distribution family in τ₃–τ₄ space, and comparing "
    "with theoretical distributions."
)

# ── Mode selection ────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state["mode"] = "offline"

col_off, col_on = st.columns(2)
with col_off:
    if st.button("Offline Mode", use_container_width=True,
                 type="primary" if st.session_state["mode"] == "offline" else "secondary"):
        st.session_state["mode"] = "offline"
        st.rerun()
with col_on:
    if st.button("Online Mode", use_container_width=True,
                 type="primary" if st.session_state["mode"] == "online" else "secondary"):
        st.session_state["mode"] = "online"
        st.rerun()

st.markdown("---")

# ── Theoretical Kappa diagram (static, always shown) ─────────────────────────
st.subheader("Kappa Distribution Family — L-Moment Ratio Diagram")

@st.cache_data
def build_kappa_diagram():
    ks = np.linspace(-0.999, 0.999, 600)

    h_configs = [
        (-0.999, "GLO  (h → −1)", "black",     "dash",    3),
        (-0.5,   "h = −0.5",      "steelblue", "dot",     2),
        ( 0.0005,"GEV  (h → 0)",  "green",     "dashdot", 3),
        ( 0.5,   "h = +0.5",      "orange",    "dot",     2),
        ( 0.999, "GPA  (h → +1)", "purple",    "dash",    3),
    ]

    curves = []
    for h_val, label, color, dash, width in h_configs:
        t3c, t4c = [], []
        for k in ks:
            t3, t4 = tau3tau4_kappa(k, h_val)
            if np.isfinite(t3) and np.isfinite(t4):
                t3c.append(t3)
                t4c.append(t4)
        curves.append((np.array(t3c), np.array(t4c), label, color, dash, width))

    special = {
        "Logistic (k=0, h→−1)":    tau3tau4_kappa(0.0, -0.999),
        "Gumbel   (k=0, h→0)":     tau3tau4_kappa(0.0,  0.0005),
        "Exponential (k=0, h→+1)": tau3tau4_kappa(0.0,  0.999),
    }

    from scipy.stats import norm, rayleigh
    from core.lmoments import lmr_from_ppf
    single = {
        "Normal":   lmr_from_ppf(lambda u: norm.ppf(u)),
        "Rayleigh": lmr_from_ppf(lambda u: rayleigh.ppf(u, scale=1.0)),
    }

    return curves, special, single


curves, special, single = build_kappa_diagram()

fig = go.Figure()

for t3c, t4c, label, color, dash, width in curves:
    fig.add_trace(go.Scatter(
        x=t3c, y=t4c,
        mode="lines", name=label,
        line=dict(color=color, dash=dash, width=width),
    ))

marker_symbols = ["square", "circle", "triangle-up"]
marker_colors  = ["black", "green", "purple"]
short_labels   = ["L", "G", "E"]
for i, (name, (t3p, t4p)) in enumerate(special.items()):
    if not (np.isfinite(t3p) and np.isfinite(t4p)):
        continue
    fig.add_trace(go.Scatter(
        x=[t3p], y=[t4p],
        mode="markers+text",
        name=name,
        text=[short_labels[i]],
        textposition="top right",
        textfont=dict(size=13, color=marker_colors[i]),
        marker=dict(color=marker_colors[i], size=11,
                    symbol=marker_symbols[i],
                    line=dict(color="black", width=1)),
        showlegend=True,
    ))

point_cfg = {
    "Normal":   ("royalblue", "circle-open"),
    "Rayleigh": ("firebrick", "diamond-open"),
}
for name, (color, symbol) in point_cfg.items():
    t3p, t4p = single[name]
    fig.add_trace(go.Scatter(
        x=[t3p], y=[t4p],
        mode="markers+text",
        name=f"{name} ({t3p:.3f}, {t4p:.3f})",
        text=[name], textposition="top right",
        marker=dict(color=color, size=13, symbol=symbol,
                    line=dict(color=color, width=2)),
    ))

fig.update_layout(
    xaxis_title="L-skewness  τ₃ = λ₃ / λ₂",
    yaxis_title="L-kurtosis  τ₄ = λ₄ / λ₂",
    xaxis=dict(range=[-0.6, 0.6]),
    yaxis=dict(range=[-0.1, 0.5]),
    legend=dict(x=1.01, y=1, xanchor="left"),
    height=560,
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── OFFLINE MODE ──────────────────────────────────────────────────────────────
if st.session_state["mode"] == "offline":
    st.subheader("Upload your matrix")
    st.markdown("Rows = samples, columns = variables. Accepted formats: `.csv`, `.npy`")

    uploaded = st.file_uploader("Choose a file", type=["csv", "npy"])

    if uploaded is not None:
        if uploaded.name.endswith(".npy"):
            X = np.load(io.BytesIO(uploaded.read()))
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        else:
            X = pd.read_csv(uploaded, header=None).values

        st.success(f"Loaded matrix: **{X.shape[0]} rows × {X.shape[1]} columns**")
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(pd.DataFrame(X).head())

        col1, col2 = st.columns(2)
        with col1:
            B = st.number_input("Bootstrap resamples (B)", min_value=10, max_value=1000, value=100, step=10)
        with col2:
            rice_KdB = st.number_input("Rice K-factor (dB) for reference point", min_value=0.0, max_value=30.0, value=10.0, step=1.0)

        if st.button("Compute", type="primary"):
            with st.spinner("Computing moments and fitting distributions…"):
                om = ordinary_moments_columns(X)
                l1, l2, l3, l4, tau3, tau4 = lmoments_columns(X)
                t3_boot, t4_boot = bootstrap_tau(X, B=int(B))
                t3_mean = float(np.nanmean(tau3))
                t4_mean = float(np.nanmean(tau4))

                k_fit, h_fit = fit_kappa(t3_mean, t4_mean)
                t3_kappa, t4_kappa = tau3tau4_kappa(k_fit, h_fit)
                t3_kappa_curve, t4_kappa_curve = kappa_curve(h_fit)

                t3_ray, t4_ray   = rayleigh_tau()
                t3_norm, t4_norm = normal_tau()
                t3_rice, t4_rice = rice_tau(KdB=rice_KdB)

                t3_ln, t4_ln   = lognormal_tau_curve(np.linspace(0.01, 4.0, 100), m=300)
                t3_gev, t4_gev = gev_curve()
                t3_glo, t4_glo = glo_curve()
                t3_gpa, t4_gpa = gpa_curve()
                t3_lb,  t4_lb  = lower_bound_curve()

                t4_std = float(np.nanstd(tau4, ddof=1))

                def z_stat(t4_dist):
                    return (t4_dist - t4_mean) / t4_std if t4_std > 0 else np.nan

                def d_stat(t3_dist, t4_dist):
                    return float(np.sqrt((t3_dist - t3_mean) ** 2 + (t4_dist - t4_mean) ** 2))

                def closest(t3c, t4c):
                    i = np.argmin((t3c - t3_mean) ** 2 + (t4c - t4_mean) ** 2)
                    return t3c[i], t4c[i]

                t3g,    t4g    = closest(t3_gev, t4_gev)
                t3l,    t4l    = closest(t3_glo, t4_glo)
                t3p,    t4p    = closest(t3_gpa, t4_gpa)
                t3ln_c, t4ln_c = closest(t3_ln,  t4_ln)

                gof = {
                    "Kappa":     {"Z": z_stat(t4_kappa), "D": d_stat(t3_kappa, t4_kappa)},
                    "GEV":       {"Z": z_stat(t4g),      "D": d_stat(t3g, t4g)},
                    "GLO":       {"Z": z_stat(t4l),      "D": d_stat(t3l, t4l)},
                    "GPA":       {"Z": z_stat(t4p),      "D": d_stat(t3p, t4p)},
                    "Lognormal": {"Z": z_stat(t4ln_c),   "D": d_stat(t3ln_c, t4ln_c)},
                    "Rayleigh":  {"Z": z_stat(t4_ray),   "D": d_stat(t3_ray, t4_ray)},
                    "Normal":    {"Z": z_stat(t4_norm),  "D": d_stat(t3_norm, t4_norm)},
                    f"Rice({rice_KdB:.0f}dB)": {"Z": z_stat(t4_rice), "D": d_stat(t3_rice, t4_rice)},
                }

                st.session_state["X"]    = X
                st.session_state["om"]   = om
                st.session_state["lm"]   = {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "tau3": tau3, "tau4": tau4}
                st.session_state["boot"] = {"t3": t3_boot, "t4": t4_boot}
                st.session_state["kappa"] = {
                    "k": k_fit, "h": h_fit,
                    "t3": t3_kappa, "t4": t4_kappa,
                    "curve_t3": t3_kappa_curve, "curve_t4": t4_kappa_curve,
                }
                st.session_state["ref_points"] = {
                    "Rayleigh": (t3_ray, t4_ray),
                    "Normal":   (t3_norm, t4_norm),
                    f"Rice({rice_KdB:.0f}dB)": (t3_rice, t4_rice),
                }
                st.session_state["ref_curves"] = {
                    "GEV":         (t3_gev, t4_gev),
                    "GLO":         (t3_glo, t4_glo),
                    "GPA":         (t3_gpa, t4_gpa),
                    "Lognormal":   (t3_ln,  t4_ln),
                    "Lower bound": (t3_lb,  t4_lb),
                }
                st.session_state["gof"]      = gof
                st.session_state["means"]    = {"t3": t3_mean, "t4": t4_mean}
                st.session_state["rice_KdB"] = rice_KdB

            st.success("Done! Navigate to the sidebar pages to explore results.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Columns",  X.shape[1])
            c2.metric("Rows",     X.shape[0])
            c3.metric("Mean τ₃",  f"{t3_mean:.4f}")
            c4.metric("Mean τ₄",  f"{t4_mean:.4f}")
            c5, c6 = st.columns(2)
            c5.metric("Fitted k", f"{k_fit:.4f}")
            c6.metric("Fitted h", f"{h_fit:.4f}")

    elif "X" in st.session_state:
        st.info("Matrix already loaded — navigate to the sidebar pages or upload a new file.")
        X       = st.session_state["X"]
        t3_mean = st.session_state["means"]["t3"]
        t4_mean = st.session_state["means"]["t4"]
        k_fit   = st.session_state["kappa"]["k"]
        h_fit   = st.session_state["kappa"]["h"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Columns",  X.shape[1])
        c2.metric("Rows",     X.shape[0])
        c3.metric("Mean τ₃",  f"{t3_mean:.4f}")
        c4.metric("Mean τ₄",  f"{t4_mean:.4f}")
        c5, c6 = st.columns(2)
        c5.metric("Fitted k", f"{k_fit:.4f}")
        c6.metric("Fitted h", f"{h_fit:.4f}")
    else:
        st.info("Upload a matrix file to compute moments and fit distributions.")

# ── ONLINE MODE ───────────────────────────────────────────────────────────────
else:
    st.subheader("Online Mode — LOS / NLOS Streaming")
    st.markdown(
        "**LOS** (Line of Sight) sampled from a **Rice** distribution.  "
        "**NLOS** (Non-Line of Sight) sampled from a **Rayleigh** distribution.  "
        "Matrix size: **1 000 rows × 64 columns** each stream."
    )

    col1, col2 = st.columns(2)
    with col1:
        refresh_interval = st.slider("Refresh interval (s)", 5, 120, 15, step=5)
    with col2:
        rice_KdB_on = st.number_input("Rice K-factor for LOS (dB)",
                                      min_value=0.0, max_value=30.0,
                                      value=10.0, step=1.0)

    st.session_state["online_config"] = {
        "refresh_interval": int(refresh_interval),
        "rice_KdB":         float(rice_KdB_on),
    }

    if "los" in st.session_state and "nlos" in st.session_state:
        los_data  = st.session_state["los"]
        nlos_data = st.session_state["nlos"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LOS Mean τ₃",  f"{los_data['means']['t3']:.4f}")
        c2.metric("LOS Mean τ₄",  f"{los_data['means']['t4']:.4f}")
        c3.metric("NLOS Mean τ₃", f"{nlos_data['means']['t3']:.4f}")
        c4.metric("NLOS Mean τ₄", f"{nlos_data['means']['t4']:.4f}")
    else:
        st.info("Navigate to the **Moments** page to start the live stream.")
