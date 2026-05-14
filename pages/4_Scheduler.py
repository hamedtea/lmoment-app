import datetime
import streamlit as st

st.set_page_config(page_title="Scheduler", layout="wide")
st.title("Task Scheduler")
st.caption("Schedule periodic recomputation of moments using APScheduler.")

# ── Lazy scheduler init (once per session) ────────────────────────────────────
if "scheduler" not in st.session_state:
    from apscheduler.schedulers.background import BackgroundScheduler
    sched = BackgroundScheduler()
    sched.start()
    st.session_state["scheduler"] = sched
    st.session_state["job_log"] = []

scheduler = st.session_state["scheduler"]

# ── Add a job ─────────────────────────────────────────────────────────────────
st.subheader("Add a recompute job")

if "X" not in st.session_state:
    st.warning("No matrix loaded yet. Load data on the Home page before scheduling jobs.")
else:
    with st.form("add_job"):
        interval_sec = st.number_input("Run every (seconds)", min_value=10, max_value=3600, value=60, step=10)
        job_id = st.text_input("Job ID (unique name)", value="recompute_1")
        submitted = st.form_submit_button("Add job")

    if submitted:
        X = st.session_state["X"]
        log = st.session_state["job_log"]

        def recompute_job():
            from core.lmoments import lmoments_columns, bootstrap_tau
            from core.ordinary_moments import ordinary_moments_columns
            from core.kappa import fit_kappa, tau3tau4_kappa, kappa_curve
            import numpy as np

            om = ordinary_moments_columns(X)
            l1, l2, l3, l4, tau3, tau4 = lmoments_columns(X)
            t3_boot, t4_boot = bootstrap_tau(X, B=50)
            t3_mean = float(np.nanmean(tau3))
            t4_mean = float(np.nanmean(tau4))
            k_fit, h_fit = fit_kappa(t3_mean, t4_mean)
            t3_kappa, t4_kappa = tau3tau4_kappa(k_fit, h_fit)
            t3_kc, t4_kc = kappa_curve(h_fit)

            st.session_state["om"] = om
            st.session_state["lm"] = {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "tau3": tau3, "tau4": tau4}
            st.session_state["boot"] = {"t3": t3_boot, "t4": t4_boot}
            st.session_state["kappa"] = {
                "k": k_fit, "h": h_fit,
                "t3": t3_kappa, "t4": t4_kappa,
                "curve_t3": t3_kc, "curve_t4": t4_kc,
            }
            st.session_state["means"] = {"t3": t3_mean, "t4": t4_mean}
            log.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "job": job_id, "status": "OK"})

        # Remove existing job with same ID if present
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

        scheduler.add_job(recompute_job, "interval", seconds=int(interval_sec), id=job_id)
        st.success(f"Job **{job_id}** scheduled every {interval_sec}s.")

# ── Active jobs ───────────────────────────────────────────────────────────────
st.subheader("Active jobs")
jobs = scheduler.get_jobs()
if jobs:
    for job in jobs:
        c1, c2, c3 = st.columns([3, 3, 1])
        c1.write(f"**{job.id}**")
        c2.write(f"Next run: {job.next_run_time}")
        if c3.button("Remove", key=f"rm_{job.id}"):
            scheduler.remove_job(job.id)
            st.rerun()
else:
    st.info("No active jobs.")

# ── Run log ───────────────────────────────────────────────────────────────────
st.subheader("Run log")
log = st.session_state.get("job_log", [])
if log:
    import pandas as pd
    st.dataframe(pd.DataFrame(log[::-1]), use_container_width=True, hide_index=True)
else:
    st.info("No runs yet.")

if st.button("Clear log"):
    st.session_state["job_log"] = []
    st.rerun()
