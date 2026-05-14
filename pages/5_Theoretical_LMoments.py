import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.tLmoments import (
    gev_curve, glo_curve, gpa_curve,
    normal3_tau_curve,
    lower_bound_curve, all_theoretical_points,
)

st.set_page_config(page_title="Theoretical Distributions", layout="wide")
st.title("Theoretical Distributions — L-Moment Ratio Diagram")
st.caption(
    "Static reference diagram — no data required. "
    "All (τ₃, τ₄) values are computed from each distribution's PPF "
    "via Gauss–Legendre quadrature. All curves span τ₃ ∈ [−1, 1]."
)

# ── Sidebar display options ───────────────────────────────────────────────────
st.sidebar.header("Display options")
show_gev   = st.sidebar.checkbox("GEV curve",              value=True)
show_glo   = st.sidebar.checkbox("GLO curve",              value=True)
show_gpa   = st.sidebar.checkbox("Gen. Pareto curve",      value=True)
show_n3    = st.sidebar.checkbox("3-par. Normal (P-III)",  value=True)
show_lb    = st.sidebar.checkbox("Lower bound",            value=True)
show_norm  = st.sidebar.checkbox("Normal point",           value=True)
show_unif  = st.sidebar.checkbox("Uniform point",          value=True)
show_exp   = st.sidebar.checkbox("Exponential point",      value=True)
show_logi  = st.sidebar.checkbox("Logistic point",         value=True)
show_gumb  = st.sidebar.checkbox("Gumbel point",           value=True)

# ── Compute (cached) ──────────────────────────────────────────────────────────
@st.cache_data
def compute_all():
    return {
        "gev": gev_curve(),
        "glo": glo_curve(),
        "gpa": gpa_curve(),
        "n3":  normal3_tau_curve(),
        "lb":  lower_bound_curve(),
        "pts": all_theoretical_points(),
    }

with st.spinner("Computing theoretical curves…"):
    data = compute_all()

# ── Build figure ──────────────────────────────────────────────────────────────
fig = go.Figure()

curve_cfg = [
    ("gev", show_gev, "GEV",               "green",      "dash",     2.5),
    ("glo", show_glo, "GLO",               "black",      "dot",      2.5),
    ("gpa", show_gpa, "Gen. Pareto",       "purple",     "dashdot",  2.5),
    ("n3",  show_n3,  "3-par. Normal (P-III)", "steelblue","longdash",2.0),
    ("lb",  show_lb,  "Lower bound",       "gray",       "solid",    1.5),
]
for key, visible, label, color, dash, width in curve_cfg:
    if not visible:
        continue
    t3c, t4c = data[key]
    fig.add_trace(go.Scatter(
        x=t3c, y=t4c,
        mode="lines", name=label,
        line=dict(color=color, dash=dash, width=width),
    ))

point_cfg = {
    "Normal":      (show_norm, "royalblue",  "circle",      "top right"),
    "Uniform":     (show_unif, "steelblue",  "square",      "top right"),
    "Exponential": (show_exp,  "firebrick",  "triangle-up", "top right"),
    "Logistic":    (show_logi, "darkgreen",  "cross",       "top right"),
    "Gumbel":      (show_gumb, "teal",       "diamond",     "top right"),
}
for name, (visible, color, symbol, tpos) in point_cfg.items():
    if not visible or name not in data["pts"]:
        continue
    t3p, t4p = data["pts"][name]
    fig.add_trace(go.Scatter(
        x=[t3p], y=[t4p],
        mode="markers+text",
        name=f"{name} ({t3p:.3f}, {t4p:.3f})",
        text=[name], textposition=tpos,
        marker=dict(color=color, size=12, symbol=symbol,
                    line=dict(color="black", width=1)),
    ))

fig.update_layout(
    xaxis_title="L-skewness  τ₃ = λ₃ / λ₂",
    yaxis_title="L-kurtosis  τ₄ = λ₄ / λ₂",
    xaxis=dict(autorange=True),
    yaxis=dict(autorange=True),
    legend=dict(x=1.01, y=1, xanchor="left"),
    height=640,
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

# ── Reference table ───────────────────────────────────────────────────────────
st.subheader("Reference values")
import pandas as pd
rows = [{"Distribution": k, "τ₃": round(v[0], 6), "τ₄": round(v[1], 6)}
        for k, v in data["pts"].items()]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
