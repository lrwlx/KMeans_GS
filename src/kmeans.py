# -*- coding: utf-8 -*-
"""Shared Lloyd refinement.

Every seeder in :mod:`src.seeding` is followed by this exact code path, so
differences in iteration count and final quality can be attributed to the
initial centres alone.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _default_chunk(n_features, K):
    return max(1024, int(2 ** 22 // max(1, K)))


def assign_labels(X, centers, chunk=0):
    """Assign every point to its nearest centre.

    Returns (labels, squared distance to the assigned centre).
    """
    X = np.asarray(X)
    n = X.shape[0]
    C = np.asarray(centers, dtype=np.float64)
    if chunk <= 0:
        chunk = _default_chunk(X.shape[1], C.shape[0])
    Cn = np.einsum("ij,ij->i", C, C)
    labels = np.empty(n, dtype=np.int32)
    mind = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xb = X[s:e]
        d2 = np.einsum("ij,ij->i", Xb, Xb)[:, None] - 2.0 * (Xb @ C.T) + Cn[None, :]
        j = np.argmin(d2, axis=1)
        labels[s:e] = j
        mind[s:e] = d2[np.arange(e - s), j]
    return labels, mind


def lloyd(X, centers0, max_iter=300, tol=1e-4, chunk=0):
    """Standard Lloyd iteration.

    Stops when the largest centre displacement is at most ``tol`` (absolute,
    in data units; the tabular data are standardised) or after ``max_iter``
    updates.  Returns (labels, centers, n_iter); ``n_iter`` counts the update
    steps actually performed.
    """
    X = np.asarray(X)
    n, d = X.shape
    K = int(centers0.shape[0])
    C = np.array(centers0, dtype=np.float64, copy=True)
    if chunk <= 0:
        chunk = _default_chunk(d, K)
    idx_all = np.arange(n)
    labels, mind = assign_labels(X, C, chunk)
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        counts = np.bincount(labels, minlength=K)
        M = sparse.csr_matrix(
            (np.ones(n, dtype=np.float64), (labels, idx_all)), shape=(K, n)
        )
        sums = np.asarray(M @ X, dtype=np.float64)
        newC = C.copy()
        nz = counts > 0
        newC[nz] = sums[nz] / counts[nz, None]
        for k in np.where(~nz)[0]:
            j = int(np.argmax(mind))
            newC[k] = X[j]
            mind[j] = -1.0
        shift = float(np.max(np.sqrt(np.einsum("ij,ij->i", newC - C, newC - C))))
        C = newC
        if shift <= tol:
            break
        labels, mind = assign_labels(X, C, chunk)
    labels, _ = assign_labels(X, C, chunk)
    return labels, C, n_iter
