# -*- coding: utf-8 -*-
"""Seeding rules for the KMeans-GS v2 study.

All seeders return a :class:`SeedResult` with the selected centres, their row
indices in ``X``, and (where relevant) diagnostics used in the paper:
per-round trial counts, fallbacks, thresholds and the estimated scale.

Design notes
------------
* ``X`` is expected to be a contiguous float64 array of shape (N, d).
* Every seeder takes an explicit ``numpy.random.Generator`` so that all
  methods can be run with the same repetition seed.
* The scale estimator of KMeans-GS is the main engineering change relative to
  v1.  v1 computed two column-wise percentiles over an (m, d) subsample with
  ``numpy.percentile``, whose strided access makes the cost grow with d and
  dominate the seeding time.  v2 uses a coordinate-subsampled, block-wise
  quantile estimator (``scale_mode="coord"``) whose cost is independent of d
  for d > ``n_coords`` and an order of magnitude smaller for large d.  The
  full estimator is kept as ``scale_mode="full"`` for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import kmeans_plusplus

DEFAULT_CHUNK = 65536


@dataclass
class SeedResult:
    centers: np.ndarray
    method: str = ""
    indices: Optional[np.ndarray] = None
    trials: Optional[np.ndarray] = None       # per round, length K-1
    fallbacks: int = 0
    taus: Optional[np.ndarray] = None
    d0: float = float("nan")
    scale_mode: str = ""
    scale_n_points: int = 0
    scale_n_coords: int = 0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# distance helpers
# --------------------------------------------------------------------------
def sqdist_to_point(X, c, chunk=DEFAULT_CHUNK):
    """Squared Euclidean distance from every row of X to the point c."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = X[s:e] - c
        out[s:e] = np.einsum("ij,ij->i", diff, diff)
    return out


def sqdist_to_points(X, P, chunk=DEFAULT_CHUNK):
    """(N, m) squared distance matrix between X and P, computed in chunks."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    n = X.shape[0]
    m = P.shape[0]
    out = np.empty((n, m), dtype=np.float64)
    pn = np.einsum("ij,ij->i", P, P)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xb = X[s:e]
        blk = (np.einsum("ij,ij->i", Xb, Xb)[:, None]
               - 2.0 * (Xb @ P.T) + pn[None, :])
        np.maximum(blk, 0.0, out=blk)
        out[s:e] = blk
    return out


def _pair_quantiles(sub, qs=(25.0, 75.0)):
    """Per-column quantiles of ``sub`` (m, d), cache-friendly.

    ``numpy.percentile(..., axis=0)`` walks columns with a stride of d, which
    is slow for large d.  Transposing blocks of columns makes the rows
    contiguous, so the percentile is computed over contiguous vectors.
    """
    m, d = sub.shape
    q1 = np.empty(d, dtype=np.float64)
    q3 = np.empty(d, dtype=np.float64)
    block = 64
    for s in range(0, d, block):
        e = min(s + block, d)
        blk = np.ascontiguousarray(sub[:, s:e].T)   # (b, m)
        q = np.percentile(blk, qs, axis=1)          # (2, b)
        q1[s:e] = q[0]
        q3[s:e] = q[1]
    return q1, q3


def robust_scale(X, rng, *, mode="coord", n_points=512, n_coords=128, alpha=0.5):
    """Estimate ``d0 = alpha * ||IQR||_2`` with cost independent of N.

    mode="coord"
        Draw ``n_points`` rows, keep ``min(d, n_coords)`` randomly chosen
        coordinates, and estimate ``||IQR||^2`` by ``(d/s) * sum_{k in S} IQR_k^2``.
        This is the Horvitz-Thompson estimator of the squared norm; its cost is
        O(min(N, m) * min(d, s)) rather than O(N d) or O(m d) for large d.
    mode="full"
        The v1 estimator: all coordinates, all ``m`` points.

    Returns (d0, n_points_used, n_coords_used).
    """
    n, d = X.shape
    if mode not in ("coord", "full"):
        raise ValueError("unknown scale_mode: %r" % (mode,))
    s = d if mode == "full" else min(d, int(n_coords))
    m = min(n, int(n_points))
    if s < d:
        cols = np.sort(rng.choice(d, size=s, replace=False))
        rows = rng.integers(0, n, size=m) if m < n else np.arange(n)
        sub = X[np.ix_(rows, cols)]
    else:
        cols = np.arange(d)
        sub = X[rng.integers(0, n, size=m)] if m < n else X[:m]
    q1, q3 = _pair_quantiles(sub, (25.0, 75.0))
    iqr = q3 - q1
    total = float(np.sum(iqr ** 2))
    if s < d:
        total *= d / float(s)
    d0 = float(alpha) * float(np.sqrt(total))
    if not np.isfinite(d0) or d0 <= 0.0:
        raise ValueError("non-positive scale estimate (degenerate data?)")
    return d0, int(m), int(s)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def seed_random(X, K, rng, **kwargs):
    n = X.shape[0]
    if K > n:
        raise ValueError("K > N")
    idx = rng.choice(n, size=K, replace=False)
    return SeedResult(centers=np.array(X[idx], dtype=np.float64, copy=True),
                      method="kmeans", indices=np.asarray(idx))


def _sample_d2(closest, n_trials, rng):
    """Sample indices with probability proportional to ``closest``."""
    total = float(closest.sum())
    if not np.isfinite(total) or total <= 0.0:
        return rng.integers(0, closest.size, size=n_trials)
    cum = np.cumsum(closest)
    r = rng.random(n_trials) * float(cum[-1])
    return np.searchsorted(cum, r, side="left").clip(0, closest.size - 1)


def seed_kmeanspp(X, K, rng, *, chunk=DEFAULT_CHUNK, **kwargs):
    """Vanilla k-means++ (Arthur & Vassilvitskii, 2007): one candidate per round."""
    n, d = X.shape
    centers = np.empty((K, d), dtype=np.float64)
    idx = np.empty(K, dtype=np.int64)
    idx[0] = int(rng.integers(n))
    centers[0] = X[idx[0]]
    closest = sqdist_to_point(X, centers[0], chunk)
    for k in range(1, K):
        j = int(_sample_d2(closest, 1, rng)[0])
        idx[k] = j
        centers[k] = X[j]
        np.minimum(closest, sqdist_to_point(X, centers[k], chunk), out=closest)
    return SeedResult(centers=centers, method="kmeanspp", indices=idx)


def seed_kmeanspp_greedy(X, K, rng, *, n_local_trials=None, chunk=DEFAULT_CHUNK,
                         **kwargs):
    """Greedy k-means++: draws ``2 + ln K`` candidates per round and keeps the
    one that minimises the resulting potential (the scikit-learn default)."""
    n, d = X.shape
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(max(K, 2)))
    centers = np.empty((K, d), dtype=np.float64)
    idx = np.empty(K, dtype=np.int64)
    idx[0] = int(rng.integers(n))
    centers[0] = X[idx[0]]
    closest = sqdist_to_point(X, centers[0], chunk)
    for k in range(1, K):
        cand = _sample_d2(closest, n_local_trials, rng)
        d2 = sqdist_to_points(X, X[cand], chunk)            # (n, n_trials)
        pot = np.minimum(d2, closest[:, None]).sum(axis=0)  # potential of each
        best = int(np.argmin(pot))
        j = int(cand[best])
        idx[k] = j
        centers[k] = X[j]
        np.minimum(closest, d2[:, best], out=closest)
    return SeedResult(centers=centers, method="kmeanspp_greedy", indices=idx)


def seed_kmeanspp_sklearn(X, K, rng, *, n_local_trials=1, **kwargs):
    """Compiled k-means++ from scikit-learn (Cython implementation).

    ``n_local_trials=1`` gives the compiled *vanilla* rule, isolating the
    implementation from the algorithm: comparing it with our numpy
    ``seed_kmeanspp`` shows what the Python implementation costs.
    """
    seed = int(rng.integers(2 ** 31 - 1))
    centers, idx = kmeans_plusplus(np.asarray(X, dtype=np.float64), n_clusters=K,
                                   n_local_trials=int(n_local_trials),
                                   random_state=seed)
    return SeedResult(centers=np.asarray(centers, dtype=np.float64),
                      method="kmeanspp_sklearn",
                      indices=np.asarray(idx, dtype=np.int64))


def seed_kmeans_parallel(X, K, rng, *, oversample=None, rounds=5,
                         chunk=DEFAULT_CHUNK, **kwargs):
    """k-means|| (Bahmani et al., 2012): l = 2K oversampling, 5 rounds, then
    k-means++ on the candidate pool."""
    n, d = X.shape
    ell = int(oversample) if oversample else max(2 * K, K)
    centers0 = X[int(rng.integers(n))]
    psi = sqdist_to_point(X, centers0, chunk)
    picks = []
    for _ in range(max(1, int(rounds))):
        total = float(psi.sum())
        if not np.isfinite(total) or total <= 0.0:
            break
        p = np.clip(ell * psi / total, 0.0, 1.0)
        sel = np.nonzero(rng.random(n) < p)[0]
        if sel.size == 0:
            continue
        picks.append(sel)
        np.minimum(psi, sqdist_to_points(X, X[sel], chunk).min(axis=1), out=psi)
    if picks:
        pool = np.concatenate(picks)
        sub = seed_kmeanspp(np.ascontiguousarray(X[pool]), K, rng, chunk=chunk)
        out = SeedResult(centers=sub.centers, method="kmeans_par")
        out.extra["pool_size"] = int(pool.size)
    else:
        out = seed_kmeanspp(X, K, rng, chunk=chunk)
        out.method = "kmeans_par"
        out.extra["pool_size"] = 0
    return out


def seed_afkmc2(X, K, rng, *, chain=200, max_trials=100, accept_slack=2.0,
                chunk=DEFAULT_CHUNK, **kwargs):
    """AFK-MC2 (Bachem et al., 2016), with the same trial cap as KMeans-GS."""
    n, d = X.shape
    centers = np.empty((K, d), dtype=np.float64)
    centers[0] = X[int(rng.integers(n))]
    cur = X[int(rng.integers(n))]
    d2_cur = float(np.dot(cur - centers[0], cur - centers[0]))
    total = 0.0
    for _ in range(max(1, int(chain))):
        prop = X[int(rng.integers(n))]
        d2_prop = float(np.dot(prop - centers[0], prop - centers[0]))
        if d2_cur <= 0.0 or rng.random() < min(1.0, d2_prop / d2_cur):
            cur, d2_cur = prop, d2_prop
        total += d2_cur
    psi = total / max(1, int(chain))
    if not np.isfinite(psi) or psi <= 0.0:
        raise ValueError("AFK-MC2 constant estimate is not positive")

    trials = np.zeros(max(K - 1, 0), dtype=np.int64)
    fallbacks = 0
    for k in range(1, K):
        best, best_min2, t, accepted = None, -1.0, 0, False
        for t in range(1, int(max_trials) + 1):
            cand = X[int(rng.integers(n))]
            diff = centers[:k] - cand
            dmin2 = float(np.einsum("ij,ij->i", diff, diff).min())
            if dmin2 > best_min2:
                best_min2, best = dmin2, cand
            if rng.random() < dmin2 / (accept_slack * psi):
                accepted = True
                break
        trials[k - 1] = t
        if not accepted:
            fallbacks += 1
        centers[k] = best
    return SeedResult(centers=centers, method="afkmc2", trials=trials,
                      fallbacks=fallbacks, d0=float(psi),
                      scale_mode="mcmc", scale_n_points=int(chain))


# --------------------------------------------------------------------------
# KMeans-GS
# --------------------------------------------------------------------------
def seed_kmeans_gs(X, K, rng, *, batch=1, select="first", alpha=0.5,
                   decay="inv_sqrt", decay_p=0.5, first="random",
                   max_trials=100, scale_mode="coord", n_points=512,
                   n_coords=128, first_sample=2000, **kwargs):
    """Geometric rejection seeding.

    Parameters
    ----------
    batch : int
        Candidates drawn per round.  ``1`` for KMeans-GS, ``8`` for KMeans-GS-G.
    select : {"first", "farthest"}
        ``first`` accepts the first admissible candidate; ``farthest`` keeps the
        admissible candidate with the largest minimum distance.
    scale_mode : {"coord", "full"}
        See :func:`robust_scale`.
    """
    n, d = X.shape
    if K > n:
        raise ValueError("K > N")
    d0, m_used, s_used = robust_scale(X, rng, mode=scale_mode, n_points=n_points,
                                      n_coords=n_coords, alpha=alpha)
    centers = np.empty((K, d), dtype=np.float64)
    idx = np.empty(K, dtype=np.int64)

    if first == "random":
        idx[0] = int(rng.integers(n))
    elif first in ("median", "median_sub"):
        if first == "median_sub":
            mm = int(min(n, max(2, first_sample)))
            pick = rng.choice(n, size=mm, replace=False) if mm < n else np.arange(n)
            sub = np.ascontiguousarray(X[pick])
            target = np.median(sub, axis=0)
            j = int(np.argmin(sqdist_to_point(sub, target)))
            idx[0] = int(pick[j])
        else:
            target = np.median(X, axis=0)
            idx[0] = int(np.argmin(sqdist_to_point(X, target)))
    else:
        raise ValueError("unknown first-centre rule: %r" % (first,))
    centers[0] = X[idx[0]]

    trials = np.zeros(max(K - 1, 0), dtype=np.int64)
    taus = np.zeros(max(K - 1, 0), dtype=np.float64)
    fallbacks = 0
    b = int(max(1, batch))
    for k in range(1, K):
        i = k + 1
        if decay == "inv_sqrt":
            tau = d0 / np.sqrt(i)
        elif decay == "power":
            tau = d0 / (i ** float(decay_p))
        elif decay == "log":
            tau = d0 / (3.0 + np.log(i))
        else:
            raise ValueError("unknown decay: %r" % (decay,))
        taus[k - 1] = tau
        best = None
        best_idx = -1
        best_min = -1.0
        pick = -1
        t = 0
        accepted = False
        for t in range(1, max_trials + 1):
            cand_idx = rng.integers(0, n, size=b) if b > 1 else np.array(
                [rng.integers(0, n)], dtype=np.int64)
            diff = centers[None, :k, :] - X[cand_idx][:, None, :]      # (b, k, d)
            dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))       # (b, k)
            dmin = dist.min(axis=1)                                    # (b,)
            jbest = int(np.argmax(dmin))
            if dmin[jbest] > best_min:
                best_min = float(dmin[jbest])
                best = X[cand_idx[jbest]]
                best_idx = int(cand_idx[jbest])
            if dmin[jbest] > tau:
                if select == "first":
                    # the first admissible candidate in draw order
                    ok = np.nonzero(dmin > tau)[0]
                    pick = int(cand_idx[ok[0]])
                else:
                    pick = int(cand_idx[jbest])
                accepted = True
                break
        if not accepted:
            fallbacks += 1
            pick = best_idx
        trials[k - 1] = t
        idx[k] = pick
        centers[k] = X[pick]
    method = "kmeans_gs_g" if select == "farthest" else "kmeans_gs"
    return SeedResult(centers=centers, method=method, indices=idx, trials=trials,
                      fallbacks=fallbacks, taus=taus, d0=d0, scale_mode=scale_mode,
                      scale_n_points=m_used, scale_n_coords=s_used)


def seed_kmeans_gs_g(X, K, rng, **kwargs):
    kwargs.setdefault("batch", 8)
    kwargs.setdefault("select", "farthest")
    return seed_kmeans_gs(X, K, rng, **kwargs)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
SEEDERS = {
    "kmeans": seed_random,
    "kmeanspp": seed_kmeanspp,
    "kmeanspp_greedy": seed_kmeanspp_greedy,
    "kmeanspp_sklearn": seed_kmeanspp_sklearn,
    "kmeans_par": seed_kmeans_parallel,
    "afkmc2": seed_afkmc2,
    "kmeans_gs": seed_kmeans_gs,
    "kmeans_gs_g": seed_kmeans_gs_g,
}

METHOD_LABELS = {
    "kmeans": "K-means (random)",
    "kmeanspp": "k-means++",
    "kmeanspp_greedy": "k-means++ (greedy)",
    "kmeanspp_sklearn": "k-means++ (compiled)",
    "kmeans_par": "k-means||",
    "afkmc2": "AFK-MC2",
    "kmeans_gs": "KMeans-GS",
    "kmeans_gs_g": "KMeans-GS-G",
}
