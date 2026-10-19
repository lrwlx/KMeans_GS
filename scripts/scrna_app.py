#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单细胞应用实验：不同初始化在真实 scRNA-seq 数据上的质量与代价。

支持两种输入：
  paul15  ：Paul et al. 2015 造血数据（2730 细胞 × 8716 基因，19 个真实细胞类型），
            h5 文件含 data(基因×细胞) 与 cluster.id。
  pbmc10k ：10x Genomics pbmc_10k_v3 过滤矩阵（~10k 细胞，无真实标签），
            用轮廓系数 + 标记基因一致性评估。

流程（与常规 scRNA 分析一致，全部用 numpy/scipy/sklearn，不依赖 scanpy）：
  计数 -> 过滤低表达基因 -> 每细胞归一化到 1e4 -> log1p -> 高变基因(2000) -> PCA(50)
  -> K-means（random / kmeans++ / KMeans-GS / n_init=10）× repeats

用法：
  python scripts/scrna_app.py --dataset paul15 --path /home/hpclr/scdata/paul15.h5 \
      --repeats 10 --out results/scrna_paul15.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,  # noqa: E402
                             normalized_mutual_info_score, silhouette_score)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import seeding as S  # noqa: E402
from src.kmeans import lloyd  # noqa: E402


def load_paul15(path):
    import h5py
    f = h5py.File(path, "r")
    counts = np.asarray(f["data"])          # 基因 × 细胞
    genes = [g.decode() if isinstance(g, bytes) else str(g) for g in f["data_rownames"][:]]
    labels = np.asarray(f["cluster.id"]).ravel().astype(int)
    f.close()
    return sp.csr_matrix(counts.T.astype(np.float64)), genes, labels


def load_10x(path):
    """path 是解压后的 filtered_feature_bc_matrix 目录。"""
    mtx = os.path.join(path, "matrix.mtx.gz")
    if os.path.exists(mtx):
        with gzip.open(mtx, "rb") as fh:
            M = scipy.io.mmread(fh).tocsr()
    else:
        M = scipy.io.mmread(os.path.join(path, "matrix.mtx")).tocsr()
    X = sp.csr_matrix(M.T)                  # 细胞 × 基因
    feat = os.path.join(path, "features.tsv.gz")
    if not os.path.exists(feat):
        feat = os.path.join(path, "features.tsv")
    genes = [line.split(b"\t")[1].decode() for line in gzip.open(feat, "rb")] \
        if feat.endswith(".gz") else [line.split("\t")[1].strip() for line in open(feat)]
    return X, genes, None


def preprocess(counts, genes=None, n_hvg=2000, n_pcs=50):
    X = counts.astype(np.float32).tocsr()
    n_cells = X.shape[0]
    # 过滤：至少在 3 个细胞里表达的基因
    nnz = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = np.nonzero(nnz >= 3)[0]
    X = X[:, keep]
    kept_genes = [genes[j] for j in keep] if genes is not None else None
    # 每细胞归一化到 1e4 计数，再 log1p
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    X = sp.diags(1e4 / tot) @ X
    X = X.tocsr()
    X.data = np.log1p(X.data)
    # 高变基因（按方差，在稀疏矩阵上算 E[x^2]-E[x]^2）
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    var = sq - mean ** 2
    idx = np.argsort(var)[-min(n_hvg, var.size):]
    Xh = X[:, np.sort(idx)]
    dense = np.asarray(Xh.todense(), dtype=np.float32)
    pcs = PCA(n_components=int(n_pcs), svd_solver="randomized", random_state=0).fit_transform(dense)
    return np.ascontiguousarray(pcs, dtype=np.float64), X, np.sort(idx), kept_genes


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="paul15", choices=["paul15", "pbmc10k"])
    p.add_argument("--path", required=True)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--K", type=int, default=0, help="0 表示 paul15 用真实类别数、pbmc10k 用 10")
    p.add_argument("--out", default="results/scrna_app.csv")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    if args.dataset == "paul15":
        counts, genes, labels = load_paul15(args.path)
    else:
        counts, genes, labels = load_10x(args.path)
    print("载入 %s：%d 细胞 × %d 基因（%.1f s）"
          % (args.dataset, counts.shape[0], counts.shape[1], time.perf_counter() - t0), flush=True)

    t0 = time.perf_counter()
    Z, Xn, hvg_idx, kept_genes = preprocess(counts, genes, args.n_hvg, args.n_pcs)
    print("预处理 + PCA：%s（%.1f s）" % (Z.shape, time.perf_counter() - t0), flush=True)

    if args.K:
        K = args.K
    elif labels is not None:
        K = int(np.unique(labels).size)
    else:
        K = 10
    print("K=%d，重复 %d，标签=%s" % (K, args.repeats,
                                  "有" if labels is not None else "无"), flush=True)

    rows = []
    for r in range(args.repeats):
        seed = 42 + r
        for m in ("kmeans", "kmeanspp", "kmeans_gs"):
            rng = np.random.default_rng(seed)
            t1 = time.perf_counter()
            res = S.SEEDERS[m](Z, K, rng)
            t_seed = time.perf_counter() - t1
            t1 = time.perf_counter()
            lab, C, it = lloyd(Z, res.centers, max_iter=300, tol=1e-7)
            t_iter = time.perf_counter() - t1
            rows.append({"dataset": args.dataset, "repeat": r, "method": m,
                         "time_seed": t_seed, "time_total": t_seed + t_iter, "n_iter": it,
                         "ari": adjusted_rand_score(labels, lab) if labels is not None else np.nan,
                         "ami": adjusted_mutual_info_score(labels, lab) if labels is not None else np.nan,
                         "nmi": normalized_mutual_info_score(labels, lab) if labels is not None else np.nan,
                         "silhouette": float(silhouette_score(Z, lab, sample_size=min(len(Z), 5000),
                                                              random_state=seed)),
                         "trials": float(np.mean(res.trials)) if res.trials is not None else np.nan,
                         "fallbacks": int(res.fallbacks)})
        from sklearn.cluster import KMeans
        t1 = time.perf_counter()
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Z)
        rows.append({"dataset": args.dataset, "repeat": r, "method": "n_init10",
                     "time_seed": np.nan, "time_total": time.perf_counter() - t1,
                     "n_iter": km.n_iter_,
                     "ari": adjusted_rand_score(labels, km.labels_) if labels is not None else np.nan,
                     "ami": adjusted_mutual_info_score(labels, km.labels_) if labels is not None else np.nan,
                     "nmi": normalized_mutual_info_score(labels, km.labels_) if labels is not None else np.nan,
                     "silhouette": float(silhouette_score(Z, km.labels_,
                                                          sample_size=min(len(Z), 5000),
                                                          random_state=seed)),
                     "trials": np.nan, "fallbacks": 0})
        print("  repeat %d 完成" % r, flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    g = df.groupby("method").agg(ari=("ari", "mean"), ari_std=("ari", "std"),
                                 silhouette=("silhouette", "mean"),
                                 time_seed=("time_seed", "mean"),
                                 time_total=("time_total", "mean"),
                                 n_iter=("n_iter", "mean"),
                                 trials=("trials", "mean"),
                                 fallbacks=("fallbacks", "sum"))
    print("\n=== %s 结果（K=%d，%d 次重复）===" % (args.dataset, K, args.repeats))
    print(g.to_string(float_format=lambda v: "%.4f" % v))
    print("\n写入 %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
