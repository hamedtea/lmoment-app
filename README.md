# L-Moment Analyser

A Streamlit web application for computing ordinary moments and L-moments of a data matrix, fitting the Kappa distribution family in τ₃–τ₄ space, and comparing with theoretical distributions.

## Live App

[https://lmoment.streamlit.app](https://lmoment.streamlit.app)

---

## Features

### Offline Mode
- Upload a matrix as `.csv` or `.npy` (rows = samples, columns = variables)
- Compute ordinary moments: mean, variance, skewness, excess kurtosis
- Compute L-moments: λ₁, λ₂, λ₃, λ₄, L-skewness τ₃, L-kurtosis τ₄
- Bootstrap resampling (B = 10–1000) for stability of τ₃, τ₄ estimates
- Fit a Kappa distribution to the mean (τ₃, τ₄) point
- Goodness-of-fit table comparing Kappa, GEV, GLO, GPA, Lognormal, Rayleigh, Normal, Rice

### Online Mode
- Simulates two data streams automatically:
  - **LOS** (Line of Sight): Rice distribution, configurable K-factor
  - **NLOS** (Non-Line of Sight): Rayleigh distribution
  - Matrix size: 1000 rows × 64 columns per stream
- Configurable refresh interval (5–120 seconds)
- 20-refresh sliding window history with time series and trajectory plots

### Visualisations
- L-moment ratio diagram with GEV, GLO, GPA, Kappa curves
- Mahalanobis distance ellipses (MD = 1, 2, 3) in τ₃–τ₄ and τ₃–L-CV space
- Bootstrap mean trajectory (trail scatter)
- Z-score heatmap of moments across columns
- Skewness vs kurtosis scatter with 2D KDE contours
- Theoretical distributions page: GEV, GLO, GPA, Pearson Type III, lower bound

---

## Installation

```bash
git clone https://github.com/hamedtea/lmoment-app.git
cd lmoment-app
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
lmoment-app/
├── app.py                        # Home page — mode selection, Kappa diagram
├── requirements.txt
├── pages/
│   ├── 1_Moments.py              # Ordinary moments and L-moments visualisations
│   ├── 3_Kappa_Fit.py            # Kappa fit results and GOF table
│   └── 5_Theoretical_LMoments.py # Static theoretical L-moment ratio diagram
└── core/
    ├── lmoments.py               # L-moment estimators, bootstrap
    ├── ordinary_moments.py       # Mean, variance, skewness, kurtosis
    ├── kappa.py                  # Kappa distribution PPF, fitting, curve
    ├── distributions.py          # Reference curves and points (GEV, GLO, GPA, Rice, Rayleigh...)
    ├── tLmoments.py              # Theoretical curves for page 5
    └── ellipse.py                # Mahalanobis ellipse traces for Plotly
```

---

## Dependencies

| Package | Version |
|---|---|
| streamlit | ≥ 1.32.0 |
| numpy | ≥ 1.26.0 |
| pandas | ≥ 2.0.0 |
| scipy | ≥ 1.12.0 |
| plotly | ≥ 5.20.0 |

---

## Background

**L-moments** are linear combinations of order statistics that provide robust alternatives to conventional moments for characterising probability distributions. They are defined as:

- λ₁ = mean
- λ₂ = L-scale (measure of spread)
- τ₃ = λ₃/λ₂ (L-skewness, analogous to skewness)
- τ₄ = λ₄/λ₂ (L-kurtosis, analogous to kurtosis)

The **Kappa distribution** is a four-parameter family that includes GEV (h→0), GLO (h→−1), and GPA (h→+1) as special cases, making it a flexible fit in τ₃–τ₄ space.

Reference: Hosking, J.R.M. (1990). *L-moments: analysis and estimation of distributions using linear combinations of order statistics.* Journal of the Royal Statistical Society B, 52(1), 105–124.

---

## Author

Hamed Talebian — Mid Sweden University
