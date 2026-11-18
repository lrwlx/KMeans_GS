# -*- coding: utf-8 -*-
"""Complete-algorithm baselines.

These cannot be split into "seeding + shared Lloyd", so they are timed end to
end.  Hyperparameters follow label-free rules and are recorded in the result
row so that the paper can report them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import DBSCAN, Birch, KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .seeding import DEFAULT_CHUNK, sqdist_to_points


@dataclass
class FullResult:
    labels: np.ndarray
    centers: np.ndarray
    labels_eval: Optional[np.ndarray] = None
    n_iter: int = 0
    note: str = ""
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.labels_eval is None:
            self.labels_eval = self.labels


def _centers_and_eval_labels(X, labels, chunk=DEFAULT_CHUNK):
    labels = np.asarray(labels)
    uniq = [int(u) for u in np.unique(labels) if u >= 0]
    if not uniq:
        raise ValueError("no cluster produced")
    centers = np.empty((len(uniq), X.shape[1]), dtype=np.float64)
    for j, u in enumerate(uniq):
        centers[j] = X[labels == u].mean(axis=0)
    remap = {u: j for j, u in enumerate(uniq)}
    eval_labels = np.array([remap.get(int(v), -1) for v in labels], dtype=np.int64)
    noise = np.nonzero(eval_labels < 0)[0]
    if noise.size:
        for s in range(0, noise.size, chunk):
            idx = noise[s:s + chunk]
            Xb = X[idx]
            d2 = (np.einsum("ij,ij->i", Xb, Xb)[:, None]
                  - 2.0 * (Xb @ centers.T)
                  + np.einsum("ij,ij->i", centers, centers)[None, :])
            eval_labels[idx] = np.argmin(d2, axis=1)
    return centers, eval_labels, len(uniq), int(noise.size)


# --------------------------------------------------------------------------
# DBSCAN
# --------------------------------------------------------------------------
def _kdist_curve(X, min_samples, sample, seed):
    n = X.shape[0]
    m = min(n, int(sample))
    rng = np.random.default_rng(seed)
    sub = np.ascontiguousarray(X[rng.choice(n, size=m, replace=False)]) if m < n else X
    nn = NearestNeighbors(n_neighbors=min(int(min_samples) + 1, m)).fit(sub)
    d, _ = nn.kneighbors(sub)
    return sub, np.sort(d[:, -1])


def _knee(sorted_vals):
    v = np.asarray(sorted_vals, dtype=np.float64)
    n = v.size
    if n < 3:
        return float(v[-1])
    x = np.linspace(0.0, 1.0, n)
    y = (v - v.min()) / (v.max() - v.min() + 1e-12)
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y
               + x[-1] * y[0] - y[-1] * x[0])
    return float(v[int(np.argmax(d))])


def choose_dbscan_eps(X, *, min_samples=5, sample=3000, n_grid=8, seed=42,
                      min_clusters=2, max_noise=0.6, target_clusters=None,
                      band=2.0, verbose=False, **kwargs):
    """Grid-search eps on a k-distance curve, scoring every candidate on the
    full data by silhouette.  Labels are never used.

    If ``target_clusters`` is given, the silhouette is maximised only among
    candidates whose cluster count lies in ``[K/band, K*band]``.  K is an input
    to every method in the study (BIRCH and the mixture use it directly), so
    this does not give DBSCAN information that the other methods lack; it only
    prevents the silhouette from selecting a degenerate granularity (e.g. 214
    clusters for a 2-class problem).
    """
    sub, kdist = _kdist_curve(X, min_samples, sample, seed)
    knee = _knee(kdist)
    qs = np.linspace(0.02, 0.90, int(n_grid))
    cands = np.concatenate([np.quantile(kdist, qs), [knee, 0.5 * knee, 0.25 * knee]])
    cands = np.unique(cands[np.isfinite(cands) & (cands > 0)])
    n = X.shape[0]
    grid = []
    for eps in cands:
        lab = DBSCAN(eps=float(eps), min_samples=int(min_samples),
                     n_jobs=-1).fit_predict(X)
        try:
            _, ev, n_found, n_noise = _centers_and_eval_labels(X, lab)
        except ValueError:
            n_found, n_noise, ev = 0, int(n), None
        sil = float("nan")
        if ev is not None and n_found >= 2:
            try:
                sil = float(silhouette_score(X, ev, sample_size=min(n, 5000),
                                             random_state=seed))
            except Exception:
                sil = float("nan")
        grid.append({"eps": float(eps), "n_clusters": int(n_found),
                     "noise_frac": n_noise / float(n), "silhouette": sil})
        if verbose:
            print("    eps=%.4g clusters=%d noise=%.2f sil=%.3f"
                  % (eps, n_found, n_noise / float(n), sil), flush=True)
    ok = [g for g in grid if g["n_clusters"] >= int(min_clusters)
          and g["noise_frac"] <= float(max_noise) and np.isfinite(g["silhouette"])]
    if target_clusters:
        lo = max(int(min_clusters), int(np.ceil(float(target_clusters) / band)))
        hi = max(lo, int(np.floor(float(target_clusters) * band)))
        banded = [g for g in ok if lo <= g["n_clusters"] <= hi]
        if banded:
            ok = banded
        elif ok:
            # no candidate in the band: take the closest granularity to K
            ok = [min(ok, key=lambda g: abs(g["n_clusters"] - int(target_clusters)))]
    if ok:
        best = max(ok, key=lambda g: g["silhouette"])
    else:
        multi = [g for g in grid if g["n_clusters"] >= int(min_clusters)]
        best = max(multi, key=lambda g: g["n_clusters"]) if multi else grid[-1]
    best = dict(best)
    best["grid"] = grid
    return best


def run_dbscan(X, K, rng, *, min_samples=5, eps=None, knn_sample=2000,
               max_points=50000, eps_source="k-distance knee", **kwargs):
    n = X.shape[0]
    if n > int(max_points):
        raise ValueError("DBSCAN infeasible: N=%d > %d" % (n, int(max_points)))
    if eps is None:
        _, kdist = _kdist_curve(X, min_samples, knn_sample, 0)
        eps = _knee(kdist)
    db = DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=1).fit(X)
    labels = db.labels_
    centers, eval_labels, n_found, n_noise = _centers_and_eval_labels(X, labels)
    return FullResult(labels=labels, centers=centers, labels_eval=eval_labels,
                      note="DBSCAN(eps=%.4g [%s], min_samples=%d)"
                           % (float(eps), eps_source, int(min_samples)),
                      extra={"eps": float(eps), "n_clusters_found": int(n_found),
                             "noise_frac": n_noise / float(n)})


# --------------------------------------------------------------------------
# BIRCH
# --------------------------------------------------------------------------
def _median_nn_dist(X, rng, sample=1000, chunk=DEFAULT_CHUNK):
    n = X.shape[0]
    m = int(min(n, max(2, sample)))
    sub = np.ascontiguousarray(X[rng.choice(n, size=m, replace=False)]) if m < n else X
    d2 = sqdist_to_points(sub, sub, chunk)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(axis=1))
    med = float(np.median(nn))
    if not (med > 0.0):
        nz = nn[nn > 0.0]
        med = float(np.median(nz)) if nz.size else 1e-6
    return float(max(med, 1e-12))


def run_birch(X, K, rng, *, threshold=None, branching_factor=50, **kwargs):
    if threshold is None:
        threshold = _median_nn_dist(X, rng)
    br = Birch(n_clusters=int(K), threshold=float(threshold),
               branching_factor=int(branching_factor))
    labels = br.fit_predict(X)
    centers, eval_labels, n_found, n_noise = _centers_and_eval_labels(X, labels)
    return FullResult(labels=labels, centers=centers, labels_eval=eval_labels,
                      note="BIRCH(n_clusters=%d, threshold=%.4g, branching_factor=%d)"
                           % (int(K), float(threshold), int(branching_factor)),
                      extra={"threshold": float(threshold),
                             "n_clusters_found": int(n_found)})


# --------------------------------------------------------------------------
# Gaussian mixture
# --------------------------------------------------------------------------
def run_gmm(X, K, rng, *, covariance=None, max_iter=100, tol=1e-3,
            reg_covar=1e-6, **kwargs):
    n, d = X.shape
    cov = covariance or ("diag" if d > 50 else "full")
    gm = GaussianMixture(n_components=int(K), covariance_type=cov,
                         max_iter=int(max_iter), tol=float(tol),
                         reg_covar=float(reg_covar), n_init=1,
                         init_params="k-means++",
                         random_state=int(rng.integers(2 ** 31 - 1)))
    labels = gm.fit_predict(X)
    return FullResult(labels=labels,
                      centers=np.asarray(gm.means_, dtype=np.float64),
                      n_iter=int(gm.n_iter_),
                      note="GMM(covariance_type=%s, max_iter=%d, init=k-means++)"
                           % (cov, int(max_iter)),
                      extra={"lower_bound": float(gm.lower_bound_)})


# --------------------------------------------------------------------------
# MiniBatchKMeans
# --------------------------------------------------------------------------
def run_minibatch_kmeans(X, K, rng, *, batch_size=None, restarts=3,
                         max_iter=100, tol=1e-3, **kwargs):
    bs = int(batch_size) if batch_size else max(1024, 20 * int(K))
    mb = MiniBatchKMeans(n_clusters=int(K), init="k-means++", n_init=int(restarts),
                         batch_size=bs, max_iter=int(max_iter), tol=float(tol),
                         reassignment_ratio=0.01,
                         random_state=int(rng.integers(2 ** 31 - 1)))
    labels = mb.fit_predict(X)
    return FullResult(labels=labels,
                      centers=np.asarray(mb.cluster_centers_, dtype=np.float64),
                      n_iter=int(mb.n_iter_),
                      note="MiniBatchKMeans(batch_size=%d, n_init=%d, max_iter=%d)"
                           % (bs, int(restarts), int(max_iter)),
                      extra={"sse": float(mb.inertia_), "batch_size": bs})


# --------------------------------------------------------------------------
# scikit-learn KMeans as an end-to-end reference (own Lloyd implementation)
# --------------------------------------------------------------------------
def run_sklearn_kmeans(X, K, rng, *, n_init=1, max_iter=300, tol=1e-4, **kwargs):
    km = KMeans(n_clusters=int(K), init="k-means++", n_init=int(n_init),
                max_iter=int(max_iter), tol=float(tol), algorithm="lloyd",
                random_state=int(rng.integers(2 ** 31 - 1)))
    labels = km.fit_predict(X)
    return FullResult(labels=labels,
                      centers=np.asarray(km.cluster_centers_, dtype=np.float64),
                      n_iter=int(km.n_iter_),
                      note="sklearn KMeans(n_init=%d, algorithm=lloyd, tol=%g)"
                           % (int(n_init), float(tol)),
                      extra={"inertia": float(km.inertia_)})


FULL_METHODS = {
    "mbkmeans": dict(label="MiniBatchKMeans", fn=run_minibatch_kmeans),
    "dbscan": dict(label="DBSCAN", fn=run_dbscan),
    "birch": dict(label="BIRCH", fn=run_birch),
    "gmm": dict(label="Gaussian mixture", fn=run_gmm),
    "sklearn1": dict(label="scikit-learn KMeans (1 restart)", fn=run_sklearn_kmeans,
                     defaults={"n_init": 1}),
    "sklearn10": dict(label="scikit-learn KMeans (10 restarts)", fn=run_sklearn_kmeans,
                      defaults={"n_init": 10}),
}

FULL_LABELS = {k: v["label"] for k, v in FULL_METHODS.items()}
