import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Kappa Fit", layout="wide")
st.title("Kappa Distribution Goodness of Fit (GoF)")

if "X" not in st.session_state:
    st.warning("No data loaded. Go to the Home page and upload a matrix first.")
    st.stop()

kappa = st.session_state["kappa"]
gof = st.session_state["gof"]
means = st.session_state["means"]
ref_points = st.session_state["ref_points"]

# ── Fitted parameters ─────────────────────────────────────────────────────────
st.subheader("Fitted Kappa parameters")
c1, c2, c3, c4 = st.columns(4)
c1.metric("k", f"{kappa['k']:.6f}")
c2.metric("h", f"{kappa['h']:.6f}")
c3.metric("Fitted τ₃", f"{kappa['t3']:.6f}")
c4.metric("Fitted τ₄", f"{kappa['t4']:.6f}")

st.caption(
    "Parameters k and h are found by minimising (τ₃_model − τ̄₃)² + (τ₄_model − τ̄₄)² "
    "where τ̄₃, τ̄₄ are the column-mean L-moment ratios of the uploaded matrix."
)

# ── GOF table ─────────────────────────────────────────────────────────────────
st.subheader("GoF metrics")
st.markdown(
    "**Z**: standardised distance in τ₄ — |Z| < 1.64 indicates acceptable fit at 10% level.  \n"
    "**D**: Euclidean distance in τ₃–τ₄ space — smaller is better."
)

df_gof = pd.DataFrame(gof).T.reset_index()
df_gof.columns = ["Distribution", "Z-statistic", "Euclidean D"]
df_gof["|Z|"] = df_gof["Z-statistic"].abs()
df_gof = df_gof.sort_values("Euclidean D").reset_index(drop=True)

def highlight_best(s):
    styles = [""] * len(s)
    styles[0] = "background-color: #d4edda; font-weight: bold"
    return styles

st.dataframe(
    df_gof.style
        .format({"Z-statistic": "{:.4f}", "Euclidean D": "{:.6f}", "|Z|": "{:.4f}"})
        .apply(highlight_best, axis=0),
    use_container_width=True,
    hide_index=True,
)
st.download_button("Download GOF table", df_gof.to_csv(index=False).encode(), "gof.csv", "text/csv")

# ── Kappa curve plot ──────────────────────────────────────────────────────────
st.subheader("Kappa τ₃–τ₄ curve (fixed h, varying k)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=kappa["curve_t3"], y=kappa["curve_t4"],
    mode="lines", name=f"Kappa (h={kappa['h']:.3f})",
    line=dict(color="red", width=2.5),
))
fig.add_trace(go.Scatter(
    x=[kappa["t3"]], y=[kappa["t4"]], mode="markers",
    name=f"Fitted point (k={kappa['k']:.3f}, h={kappa['h']:.3f})",
    marker=dict(color="red", size=12, symbol="diamond", line=dict(color="black", width=1)),
))
fig.add_trace(go.Scatter(
    x=[means["t3"]], y=[means["t4"]], mode="markers",
    name=f"Sample mean ({means['t3']:.3f}, {means['t4']:.3f})",
    marker=dict(color="blue", size=14, symbol="star", line=dict(color="black", width=1)),
))

# Add all reference points for comparison
point_colors = {"Rayleigh": "purple", "Normal": "brown"}
for name, (t3p, t4p) in ref_points.items():
    color = point_colors.get(name, "darkgreen")
    fig.add_trace(go.Scatter(
        x=[t3p], y=[t4p], mode="markers+text",
        name=f"{name} ({t3p:.3f}, {t4p:.3f})",
        text=[name], textposition="top right",
        marker=dict(color=color, size=11, symbol="circle", line=dict(color="black", width=1)),
    ))

fig.update_layout(
    xaxis_title="τ₃ (L-skewness)",
    yaxis_title="τ₄ (L-kurtosis)",
    height=500,
    template="plotly_white",
    legend=dict(x=1.01, y=1, xanchor="left"),
)
st.plotly_chart(fig, use_container_width=True)
