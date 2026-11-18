# -*- coding: utf-8 -*-
"""Metrics used in the KMeans-GS v2 study."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             davies_bouldin_score, fowlkes_mallows_score,
                             silhouette_score)


def supervised_scores(y, labels):
    return {
        "ari": float(adjusted_rand_score(y, labels)),
        "ami": float(adjusted_mutual_info_score(y, labels)),
        "fmi": float(fowlkes_mallows_score(y, labels)),
    }


def mse_to_centers(X, labels, centers):
    """Mean squared distance from each point to its assigned centre."""
    C = np.asarray(centers, dtype=np.float64)
    lab = np.asarray(labels)
    n = X.shape[0]
    tot = 0.0
    chunk = 65536
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = X[s:e] - C[lab[s:e]]
        tot += float(np.einsum("ij,ij->", diff, diff))
    return tot / float(n)


def safe_silhouette(X, labels, sample=10000, seed=0):
    try:
        n = X.shape[0]
        k = len(np.unique(labels))
        if k < 2 or k >= n:
            return float("nan")
        return float(silhouette_score(X, labels, sample_size=min(n, int(sample)),
                                      random_state=int(seed)))
    except Exception:
        return float("nan")


def safe_davies_bouldin(X, labels):
    try:
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(davies_bouldin_score(X, labels))
    except Exception:
        return float("nan")


def seed_separation(centers):
    """Minimum and mean pairwise distance of a seed set."""
    C = np.asarray(centers, dtype=np.float64)
    K = C.shape[0]
    if K < 2:
        return {"seed_min_dist": float("nan"), "seed_mean_dist": float("nan")}
    d = 0.0
    mn = float("inf")
    cnt = 0
    for i in range(K - 1):
        diff = C[i + 1:] - C[i]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        mn = min(mn, float(dist.min()))
        d += float(dist.sum())
        cnt += dist.size
    return {"seed_min_dist": mn, "seed_mean_dist": d / cnt}


def theorem_bound(taus, K):
    """Theorem 2 bound: 2/(K(K-1)) * sum_{i=2..K} (i-1) tau_i."""
    if taus is None or K < 2:
        return float("nan")
    idx = np.arange(2, K + 1, dtype=np.float64)
    return float(2.0 / (K * (K - 1)) * np.sum((idx - 1.0) * taus))
