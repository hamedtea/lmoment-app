# L-Moment Analyser — Development Notes

## v1.0 — First Release

### What the app does
- Computes ordinary moments (mean, variance, skewness, kurtosis) and L-moments (λ1–λ4, τ3, τ4) for an uploaded n×m matrix
- Fits a Kappa distribution in τ3–τ4 space
- Compares data against theoretical distributions (GEV, GLO, GPA, Normal, Rayleigh, Rice, Lognormal)
- Online mode simulates LOS (Rice) and NLOS (Rayleigh) streams with automatic refresh and 20-step sliding window history

---

### Key technical decisions (important for v2)

**Distributions**
- `scipy.stats.gengamma` cannot reach τ3 < −0.17 — it is bounded by construction (support on (0,∞)). Use `scipy.stats.pearson3` (Pearson Type III / 3-parameter gamma) for any curve that must span τ3 ∈ (−1, 1).
- GPA curve requires logspace k up to 1000+ to smoothly reach τ3 = −1. Linear spacing is not sufficient.
- GEV curve requires k up to ~30 to cover the full τ3 range.

**L-moment ratio diagram**
- All theoretical curves are computed via Gauss-Legendre quadrature (400 points) using shifted Legendre polynomials.
- `_filter_sort` clips everything to τ3 ∈ [−1, 1] and sorts by τ3 before plotting.

**Bootstrap**
- Offline mode: B=100 (user-configurable up to 1000)
- Online mode: B=50 (fixed, for speed with 1000×64 matrices)

**Online mode auto-refresh**
- Only `app.py` auto-reruns via `time.sleep()` + `st.rerun()`.
- The Moments page does NOT auto-refresh on its own — users must navigate there manually to see updated results.
- History is a 20-entry sliding window stored in `st.session_state["history"]`.

**Mahalanobis ellipses**
- Implemented in `core/ellipse.py` using eigendecomposition of the covariance matrix.
- Draws MD = 1, 2, 3 ellipses parametrically (not using matplotlib Ellipse patch — adapted for Plotly).

**Session state structure**
- Offline mode keys: `X`, `om`, `lm`, `boot`, `kappa`, `means`, `ref_curves`, `ref_points`, `gof`, `rice_KdB`
- Online mode keys: `los`, `nlos` (each a dict with `X`, `om`, `lm`, `boot`, `kappa`, `means`, `ref_curves`), `history`, `mode`

---

### Known limitations

- Online mode Moments page requires manual navigation — it does not auto-refresh.
- Kappa fitting uses `scipy.optimize.minimize` with bounds (−0.99, 0.99) for both k and h — may not converge for extreme τ3/τ4 values.
- Scheduler page (APScheduler) is a placeholder — no scheduled jobs are wired to the analysis pipeline yet.
- GOF Z-statistic is normalised by `std(τ4)` across columns — unreliable when number of columns is small (< 10).

---

### Ideas for v2

- [ ] Wire the Scheduler to periodically reload a file from disk and recompute (true online file-watching mode)
- [ ] Add confidence intervals to bootstrap scatter (convex hull or KDE contour)
- [ ] Add GEV/GLO/GPA parameter estimation (not just GOF distance)
- [ ] Export full report as PDF
- [ ] Add column selection / filtering before computing moments
- [ ] Support multi-sheet Excel upload
- [ ] Add L-moment ratio diagram to the Moments page (overlay data on theoretical curves)
- [ ] Make online mode stream visible on Moments page without manual navigation (requires `st.fragment` with `run_every`, available in Streamlit >= 1.37)

---

### Deployment
- GitHub: https://github.com/hamedtea/lmoment-app
- Streamlit Cloud: deployed from `main` branch, entry point `app.py`
- Tagged release: `v1.0`
