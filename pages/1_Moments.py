import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

from core.ellipse import ellipse_traces

st.set_page_config(page_title="Moments", layout="wide")

mode = st.session_state.get("mode", "offline")

# ── Data access ───────────────────────────────────────────────────────────────
if mode == "offline":
    if "X" not in st.session_state:
        st.warning("No data loaded. Go to the Home page and upload a matrix first.")
        st.stop()
    datasets = {
        "Data": {
            "X":          st.session_state["X"],
            "om":         st.session_state["om"],
            "lm":         st.session_state["lm"],
            "boot":       st.session_state["boot"],
            "kappa":      st.session_state["kappa"],
            "means":      st.session_state["means"],
            "ref_curves": st.session_state["ref_curves"],
        }
    }
else:
    if "los" not in st.session_state or "nlos" not in st.session_state:
        st.warning("Online mode not active. Go to the Home page and click Online Mode.")
        st.stop()
    datasets = {
        "LOS":  st.session_state["los"],
        "NLOS": st.session_state["nlos"],
    }

first      = next(iter(datasets.values()))
n_cols     = first["X"].shape[1]
col_labels = [f"Col {i}" for i in range(n_cols)]

# Colour palette per stream
COLORS        = {"Data": "steelblue", "LOS": "steelblue",  "NLOS": "firebrick"}
BOOT_COLORS   = {"Data": "orange",    "LOS": "orange",      "NLOS": "plum"}
STAR_COLORS   = {"Data": "gold",      "LOS": "gold",        "NLOS": "silver"}
KAPPA_DASH    = {"Data": "solid",     "LOS": "solid",       "NLOS": "dash"}
CONTOUR_SCALE = {"Data": "Blues",     "LOS": "Blues",       "NLOS": "Reds"}

tab1, tab2, tab3 = st.tabs(["Ordinary Moments", "L-Moments", "History"])

# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Mean figure ───────────────────────────────────────────────────────────
    fig_mean = go.Figure()
    if mode == "offline":
        d = datasets["Data"]
        for label, col_data in zip(col_labels, d["X"].T):
            fig_mean.add_trace(go.Box(
                y=col_data, name=label, boxmean=True,
                marker_color="steelblue", line_color="steelblue",
                showlegend=False,
            ))
        fig_mean.add_trace(go.Scatter(
            x=col_labels, y=d["om"]["mean"],
            mode="lines+markers", name="Mean",
            line=dict(color="firebrick", dash="dash", width=2),
            marker=dict(symbol="circle", size=10, color="firebrick",
                        line=dict(color="black", width=1)),
        ))
    else:
        for dname, d in datasets.items():
            fig_mean.add_trace(go.Scatter(
                x=col_labels, y=d["om"]["mean"],
                mode="lines+markers", name=f"{dname} Mean",
                line=dict(color=COLORS[dname], dash=KAPPA_DASH[dname], width=2),
                marker=dict(symbol="circle", size=8, color=COLORS[dname],
                            line=dict(color="black", width=1)),
            ))
    fig_mean.update_layout(
        title="Mean" + (" — LOS vs NLOS" if mode == "online" else ""),
        xaxis_title="Column", yaxis_title="Mean",
        height=420, template="plotly_white",
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    st.plotly_chart(fig_mean, use_container_width=True)

    # ── Skewness vs Excess Kurtosis ───────────────────────────────────────────
    fig_sk = go.Figure()
    for dname, d in datasets.items():
        skew = np.asarray(d["om"]["skewness"])
        kurt = np.asarray(d["om"]["kurtosis"])
        if len(skew) >= 2:
            kde = gaussian_kde(np.vstack([skew, kurt]))
            pad_x = (skew.max() - skew.min()) * 0.3 or 0.5
            pad_y = (kurt.max() - kurt.min()) * 0.3 or 0.5
            xg = np.linspace(skew.min() - pad_x, skew.max() + pad_x, 120)
            yg = np.linspace(kurt.min() - pad_y, kurt.max() + pad_y, 120)
            XX, YY = np.meshgrid(xg, yg)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            fig_sk.add_trace(go.Contour(
                x=xg, y=yg, z=ZZ,
                colorscale=CONTOUR_SCALE[dname],
                contours=dict(showlabels=False),
                line=dict(width=1), showscale=False,
                name=f"KDE {dname}", showlegend=False,
            ))
        fig_sk.add_trace(go.Scatter(
            x=skew, y=kurt, mode="markers",
            marker=dict(symbol="circle", size=10, color=COLORS[dname],
                        line=dict(color="black", width=1)),
            name=dname,
        ))
    fig_sk.update_layout(
        title="Skewness vs Excess Kurtosis" + (" — LOS vs NLOS" if mode == "online" else ""),
        xaxis_title="Skewness", yaxis_title="Excess Kurtosis",
        xaxis=dict(autorange=True), yaxis=dict(autorange=True),
        height=420, template="plotly_white",
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    st.plotly_chart(fig_sk, use_container_width=True)

    # ── Heatmap(s) ────────────────────────────────────────────────────────────
    moment_names = ["Mean", "Variance", "Skewness", "Kurtosis (excess)"]
    for dname, d in datasets.items():
        raw = np.column_stack([d["om"]["mean"], d["om"]["variance"],
                               d["om"]["skewness"], d["om"]["kurtosis"]])
        mu  = raw.mean(axis=0)
        std = raw.std(axis=0, ddof=1)
        std[std == 0] = 1
        z = (raw - mu) / std
        fig_hm = go.Figure(go.Heatmap(
            z=z.T, x=col_labels, y=moment_names,
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="z-score"),
            text=np.round(raw.T, 3),
            hovertemplate=(
                "Variable: %{x}<br>Moment: %{y}<br>"
                "Raw: %{text}<br>z: %{z:.2f}<extra></extra>"
            ),
        ))
        suffix = f" — {dname}" if mode == "online" else ""
        fig_hm.update_layout(
            title=f"Moments heatmap (z-score normalised){suffix}",
            xaxis_title="Variable", yaxis_title="Moment",
            height=300, template="plotly_white",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    if mode == "offline":
        d = datasets["Data"]
        df_ord = pd.DataFrame({
            "Column": col_labels,
            "Mean": d["om"]["mean"], "Variance": d["om"]["variance"],
            "Skewness": d["om"]["skewness"], "Kurtosis (excess)": d["om"]["kurtosis"],
        })
        st.download_button("Download CSV", df_ord.to_csv(index=False).encode(),
                           "ordinary_moments.csv", "text/csv")

# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── τ3 vs L-CV ────────────────────────────────────────────────────────────
    fig_lcv = go.Figure()
    dash_styles = ["solid", "dash", "dot"]
    for dname, d in datasets.items():
        tau3 = np.asarray(d["lm"]["tau3"])
        lcv  = np.asarray(d["lm"]["l2"]) / np.asarray(d["lm"]["l1"])
        color = COLORS[dname]
        for i, tr in enumerate(ellipse_traces(tau3, lcv, radii=(1, 2, 3))):
            fig_lcv.add_trace(go.Scatter(
                x=tr["x"], y=tr["y"], mode="lines",
                name=f"{dname} {tr['name']}",
                legendgroup=dname,
                showlegend=tr["showlegend"],
                line=dict(color=color, width=1.5, dash=dash_styles[i]),
            ))
        fig_lcv.add_trace(go.Scatter(
            x=tau3, y=lcv, mode="markers",
            marker=dict(symbol="circle", size=10, color=color,
                        line=dict(color="black", width=1)),
            name=dname, legendgroup=dname,
        ))
    fig_lcv.update_layout(
        title="L-skewness vs L-CV" + (" — LOS vs NLOS" if mode == "online" else ""),
        xaxis_title="L-skewness  τ₃ = λ₃ / λ₂",
        yaxis_title="L-CV  τ₂ = λ₂ / λ₁",
        xaxis=dict(autorange=True), yaxis=dict(autorange=True),
        height=420, template="plotly_white",
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    st.plotly_chart(fig_lcv, use_container_width=True)

    # ── τ3 vs τ4 ─────────────────────────────────────────────────────────────
    fig_t34 = go.Figure()

    # Theoretical reference curves (shared)
    ref_curves = first["ref_curves"]
    theory_cfg = [
        ("GEV", "green",  "dash"),
        ("GLO", "black",  "dot"),
        ("GPA", "purple", "dashdot"),
    ]
    for name, color, dash in theory_cfg:
        if name not in ref_curves:
            continue
        ct3 = np.asarray(ref_curves[name][0])
        ct4 = np.asarray(ref_curves[name][1])
        m = (ct3 >= 0) & (ct3 <= 1)
        fig_t34.add_trace(go.Scatter(
            x=ct3[m], y=ct4[m], mode="lines", name=name,
            line=dict(color=color, dash=dash, width=1.5),
        ))

    for dname, d in datasets.items():
        tau3    = np.asarray(d["lm"]["tau3"])
        tau4    = np.asarray(d["lm"]["tau4"])
        t3_mean = d["means"]["t3"]
        t4_mean = d["means"]["t4"]
        kappa   = d["kappa"]
        bt3     = np.asarray(d["boot"]["t3"])
        bt4     = np.asarray(d["boot"]["t4"])
        color   = COLORS[dname]

        kt3  = np.asarray(kappa["curve_t3"])
        kt4  = np.asarray(kappa["curve_t4"])
        mask = kt3 >= 0
        fig_t34.add_trace(go.Scatter(
            x=kt3[mask], y=kt4[mask], mode="lines",
            name=f"Kappa {dname} (k={kappa['k']:.3f}, h={kappa['h']:.3f})",
            legendgroup=dname,
            line=dict(color=color, width=2, dash=KAPPA_DASH[dname]),
        ))
        fig_t34.add_trace(go.Scatter(
            x=tau3, y=tau4, mode="markers",
            marker=dict(symbol="circle", size=8, color=color,
                        line=dict(color="black", width=1)),
            name=f"{dname} columns", legendgroup=dname,
        ))
        fig_t34.add_trace(go.Scatter(
            x=bt3, y=bt4, mode="markers", opacity=0.7,
            marker=dict(symbol="star", size=10, color=BOOT_COLORS[dname],
                        line=dict(color="black", width=0.5)),
            name=f"{dname} bootstrap (B={len(bt3)})", legendgroup=dname,
        ))
        fig_t34.add_trace(go.Scatter(
            x=[t3_mean], y=[t4_mean], mode="markers",
            marker=dict(symbol="star", size=18, color=STAR_COLORS[dname],
                        line=dict(color="black", width=1.5)),
            name=f"{dname} mean (τ₃={t3_mean:.3f}, τ₄={t4_mean:.3f})",
            legendgroup=dname,
        ))

    x_max = max(float(np.nanmax(np.asarray(d["lm"]["tau3"]))) for d in datasets.values())
    fig_t34.update_layout(
        title="L-skewness vs L-kurtosis" + (" — LOS vs NLOS" if mode == "online" else ""),
        xaxis_title="L-skewness  τ₃ = λ₃ / λ₂",
        yaxis_title="L-kurtosis  τ₄ = λ₄ / λ₂",
        xaxis=dict(range=[0, x_max * 1.05]),
        yaxis=dict(autorange=True),
        height=480, template="plotly_white",
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    st.plotly_chart(fig_t34, use_container_width=True)

    # ── Zoomed bootstrap region ───────────────────────────────────────────────
    all_bt3 = np.concatenate([np.asarray(d["boot"]["t3"]) for d in datasets.values()])
    all_bt4 = np.concatenate([np.asarray(d["boot"]["t4"]) for d in datasets.values()])
    pad_y   = (all_bt4.max() - all_bt4.min()) * 0.3 or 0.05
    x0, x1  = 0.0, all_bt3.max() + all_bt3.std()
    y0, y1  = all_bt4.min() - pad_y, all_bt4.max() + pad_y

    fig_zoom = go.Figure()

    for name, color, dash in theory_cfg:
        if name not in ref_curves:
            continue
        ct3 = np.asarray(ref_curves[name][0])
        ct4 = np.asarray(ref_curves[name][1])
        m = (ct3 >= x0) & (ct3 <= x1)
        fig_zoom.add_trace(go.Scatter(
            x=ct3[m], y=ct4[m], mode="lines", name=name,
            line=dict(color=color, dash=dash, width=1.5),
        ))

    for dname, d in datasets.items():
        t3_mean = d["means"]["t3"]
        t4_mean = d["means"]["t4"]
        kappa   = d["kappa"]
        bt3     = np.asarray(d["boot"]["t3"])
        bt4     = np.asarray(d["boot"]["t4"])
        color   = COLORS[dname]

        kt3 = np.asarray(kappa["curve_t3"])
        kt4 = np.asarray(kappa["curve_t4"])
        m_k = (kt3 >= x0) & (kt3 <= x1)
        fig_zoom.add_trace(go.Scatter(
            x=kt3[m_k], y=kt4[m_k], mode="lines",
            name=f"Kappa {dname}", legendgroup=dname,
            line=dict(color=color, width=2, dash=KAPPA_DASH[dname]),
        ))
        fig_zoom.add_trace(go.Scatter(
            x=bt3, y=bt4, mode="markers", opacity=0.7,
            marker=dict(symbol="star", size=10, color=BOOT_COLORS[dname],
                        line=dict(color="black", width=0.5)),
            name=f"{dname} bootstrap", legendgroup=dname,
        ))
        fig_zoom.add_trace(go.Scatter(
            x=[t3_mean], y=[t4_mean], mode="markers",
            marker=dict(symbol="star", size=18, color=STAR_COLORS[dname],
                        line=dict(color="black", width=1.5)),
            name=f"{dname} mean (τ₃={t3_mean:.3f})", legendgroup=dname,
        ))

    fig_zoom.update_layout(
        title="Zoomed — Bootstrap region with fitted distributions"
              + (" — LOS vs NLOS" if mode == "online" else ""),
        xaxis_title="L-skewness  τ₃ = λ₃ / λ₂",
        yaxis_title="L-kurtosis  τ₄ = λ₄ / λ₂",
        xaxis=dict(range=[x0, x1]), yaxis=dict(range=[y0, y1]),
        height=480, template="plotly_white",
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    st.plotly_chart(fig_zoom, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Summary across columns" + (" — LOS" if mode == "online" else ""))
    d_summary = datasets["LOS"] if mode == "online" else datasets["Data"]
    df_lm = pd.DataFrame({
        "Column":           col_labels,
        "λ1 (L-mean)":      d_summary["lm"]["l1"],
        "λ2 (L-scale)":     d_summary["lm"]["l2"],
        "λ3":               d_summary["lm"]["l3"],
        "λ4":               d_summary["lm"]["l4"],
        "τ3 (L-skewness)":  d_summary["lm"]["tau3"],
        "τ4 (L-kurtosis)":  d_summary["lm"]["tau4"],
    })
    st.dataframe(df_lm.drop(columns="Column").describe().style.format("{:.4f}"),
                 use_container_width=True)
    st.download_button("Download CSV", df_lm.to_csv(index=False).encode(),
                       "lmoments.csv", "text/csv")

# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    history = st.session_state.get("history", [])

    if mode != "online" or len(history) == 0:
        st.info("History is only available in Online Mode. "
                "Start Online Mode and wait for at least one refresh.")
    else:
        df_h = pd.DataFrame(history)
        n    = df_h["refresh"].values

        # ── Time series: mean τ3 and τ4 over refreshes ────────────────────────
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=n, y=df_h["los_t3"], mode="lines+markers",
            name="LOS τ₃", line=dict(color="steelblue", width=2),
            marker=dict(size=7),
        ))
        fig_ts.add_trace(go.Scatter(
            x=n, y=df_h["nlos_t3"], mode="lines+markers",
            name="NLOS τ₃", line=dict(color="firebrick", width=2, dash="dash"),
            marker=dict(size=7),
        ))
        fig_ts.add_trace(go.Scatter(
            x=n, y=df_h["los_t4"], mode="lines+markers",
            name="LOS τ₄", line=dict(color="steelblue", width=2, dash="dot"),
            marker=dict(symbol="square", size=7),
        ))
        fig_ts.add_trace(go.Scatter(
            x=n, y=df_h["nlos_t4"], mode="lines+markers",
            name="NLOS τ₄", line=dict(color="firebrick", width=2, dash="dashdot"),
            marker=dict(symbol="square", size=7),
        ))
        fig_ts.update_layout(
            title="Mean τ₃ and τ₄ over refreshes",
            xaxis_title="Refresh #",
            yaxis_title="L-moment ratio",
            height=380, template="plotly_white",
            legend=dict(x=1.01, y=1, xanchor="left"),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── Trail scatter: τ3 vs τ4 trajectory ───────────────────────────────
        N    = len(df_h)
        opacities = np.linspace(0.15, 1.0, N)

        fig_trail = go.Figure()

        # Reference curves
        ref_curves = first["ref_curves"]
        for name, color, dash in [("GEV","green","dash"),
                                   ("GLO","black","dot"),
                                   ("GPA","purple","dashdot")]:
            if name not in ref_curves:
                continue
            ct3 = np.asarray(ref_curves[name][0])
            ct4 = np.asarray(ref_curves[name][1])
            m = (ct3 >= 0) & (ct3 <= 1)
            fig_trail.add_trace(go.Scatter(
                x=ct3[m], y=ct4[m], mode="lines", name=name,
                line=dict(color=color, dash=dash, width=1.5),
            ))

        # Trail: one marker per refresh, colour-coded by time index
        for stream, col in [("los", "steelblue"), ("nlos", "firebrick")]:
            fig_trail.add_trace(go.Scatter(
                x=df_h[f"{stream}_t3"].values,
                y=df_h[f"{stream}_t4"].values,
                mode="lines+markers",
                name=f"{'LOS' if stream=='los' else 'NLOS'} trail",
                line=dict(color=col, width=1, dash="dot"),
                marker=dict(
                    symbol="circle", size=10,
                    color=list(range(N)),
                    colorscale="Blues" if stream == "los" else "Reds",
                    showscale=False,
                    line=dict(color="black", width=0.5),
                ),
            ))
            # Highlight most recent point
            fig_trail.add_trace(go.Scatter(
                x=[df_h[f"{stream}_t3"].iloc[-1]],
                y=[df_h[f"{stream}_t4"].iloc[-1]],
                mode="markers",
                name=f"{'LOS' if stream=='los' else 'NLOS'} latest",
                marker=dict(symbol="star", size=16, color=col,
                            line=dict(color="black", width=1.5)),
                showlegend=True,
            ))

        fig_trail.update_layout(
            title=f"τ₃ vs τ₄ trajectory over {N} refreshes",
            xaxis_title="L-skewness  τ₃ = λ₃ / λ₂",
            yaxis_title="L-kurtosis  τ₄ = λ₄ / λ₂",
            xaxis=dict(autorange=True), yaxis=dict(autorange=True),
            height=480, template="plotly_white",
            legend=dict(x=1.01, y=1, xanchor="left"),
        )
        st.plotly_chart(fig_trail, use_container_width=True)

        st.caption(f"Sliding window of last {N}/20 refreshes. Oldest refresh drops out as new data arrives. Markers darken from oldest (light) to newest (dark).")
