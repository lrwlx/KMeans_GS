# -*- coding: utf-8 -*-
"""共享的 Lloyd 迭代实现。

三种初始化方法（random / k-means++ / KMeans-GS）在选出初始中心后走**完全相同的这条代码路径**，
因此迭代时间与最终聚类质量的差异只能归因于初始中心。

设计要点
--------
* 距离计算使用 ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 的分块矩阵乘法，
  峰值内存与样本数 N 无关（旧实现用 X[:, None] - centroids，在 N=1e5 以上会爆内存）。
* 中心更新用稀疏指示矩阵乘法（等价于按簇求和），避免 np.add.at 的性能陷阱。
* 空簇用"离自身中心最远的点"补齐（确定性），而不是旧实现的随机重取。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _default_chunk(n_features, K):
    return max(1024, int(2 ** 22 // max(1, K)))


def assign_labels(X, centers, chunk=0):
    """把每个样本分配给最近的簇中心，返回 (labels, mind)，mind 为到所属中心的平方距离。"""
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


def lloyd(X, centers0, max_iter=300, tol=1e-7, chunk=0):
    """标准 Lloyd 迭代：分配 -> 更新中心，直到中心最大位移 <= tol 或达到 max_iter。

    返回 (labels, centers, n_iter)。n_iter 为实际执行的迭代轮数。
    循环结束后再对最终中心做一次分配，保证返回的 labels 与 centers 一致
    （这次分配的代价计入迭代时间，三种方法完全相同）。
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
