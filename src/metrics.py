# -*- coding: utf-8 -*-
"""聚类质量指标与初始中心诊断量。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    silhouette_score,
)


def mse_to_centers(X, labels, centers, chunk=65536):
    """均方误差：每个样本到其所属中心的平方距离的均值。"""
    X = np.asarray(X)
    C = np.asarray(centers, dtype=np.float64)
    n = X.shape[0]
    total = 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = X[s:e] - C[labels[s:e]]
        total += float(np.einsum("ij,ij->i", diff, diff).sum())
    return total / n


def safe_silhouette(X, labels, sample=10000, seed=0):
    """轮廓系数；样本量超过 sample 时按 sklearn 的方式抽样估计（O(N^2) 不可承受）。"""
    if len(np.unique(labels)) < 2:
        return float("nan")
    n = X.shape[0]
    try:
        if n > sample:
            return float(silhouette_score(X, labels, sample_size=int(sample),
                                          random_state=seed))
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


def safe_davies_bouldin(X, labels):
    """Davies-Bouldin 指数（越小越好）。"""
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(X, labels))
    except Exception:
        return float("nan")


def supervised_scores(y_true, labels):
    """有标签数据上的三项指标（ARI / AMI / FMI），均越接近 1 越好。"""
    return {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "ami": float(adjusted_mutual_info_score(y_true, labels)),
        "fmi": float(fowlkes_mallows_score(y_true, labels)),
    }


def seed_separation(centers):
    """初始中心的最小/平均成对欧氏距离，用于与定理 2 的下界对照。"""
    C = np.asarray(centers, dtype=np.float64)
    K = C.shape[0]
    if K < 2:
        return {"seed_min_dist": float("nan"), "seed_mean_dist": float("nan")}
    diff = C[:, None, :] - C[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    iu = np.triu_indices(K, k=1)
    vals = dist[iu]
    return {"seed_min_dist": float(vals.min()), "seed_mean_dist": float(vals.mean())}
