import numpy as np


def md2_per_feature(t3, t):
    Z = np.vstack([t3, t]).T
    mask = np.all(np.isfinite(Z), axis=1)
    Z = Z[mask]
    mu = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False)
    invcov = np.linalg.pinv(cov)
    diffs = np.vstack([t3, t]).T - mu
    md2 = np.einsum('ij,jk,ik->i', diffs, invcov, diffs)
    return md2


def ellipse_traces(x, y, center=None, radii=(1, 2, 3), color="black", n_pts=200):
    """Return a list of Plotly Scatter traces for Mahalanobis ellipses.

    Parameters
    ----------
    x, y   : 1-D arrays of data points
    center : (cx, cy) override; defaults to data mean
    radii  : Mahalanobis radii to draw
    color  : line colour for all ellipses
    n_pts  : points per ellipse
    """
    data = np.vstack([x, y]).T
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] < 2:
        return []

    cov = np.cov(data, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)       # ascending eigenvalues
    order = np.argsort(vals)[::-1]         # sort descending
    vals, vecs = vals[order], vecs[:, order]

    if center is None:
        center = data.mean(axis=0)

    theta = np.linspace(0, 2 * np.pi, n_pts)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    traces = []
    for i, r in enumerate(radii):
        # Parametric ellipse in eigenvector space, rotated back to data space
        ex = r * np.sqrt(vals[0]) * cos_t
        ey = r * np.sqrt(vals[1]) * sin_t
        pts = np.column_stack([ex, ey]) @ vecs.T + center
        traces.append({
            "x": pts[:, 0].tolist(),
            "y": pts[:, 1].tolist(),
            "name": f"MD = {r}",
            "showlegend": True,
        })
    return traces
