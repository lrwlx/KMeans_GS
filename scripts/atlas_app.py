#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大规模单细胞图谱应用：146 万细胞上的端到端对比。

输入是 cellxgene 的 h5ad（CSR 稀疏矩阵）。流程：
  读 X -> 过滤低表达基因 -> 每细胞归一化 1e4 + log1p -> 2000 HVG -> PCA 50
  -> K-means（random / kmeans++ / KMeans-GS / sklearn n_init=restarts）× restarts 次取最优 SSE

报告：播种时间、总时间、Lloyd 迭代次数、最终 MSE、ARI/NMI（若有 cell_type 标签）。

用法：
  python scripts/atlas_app.py --path ~/scdata/covid_atlas.h5ad --max-cells 0 \
      --restarts 3 --out results/atlas_covid.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,  # noqa: E402
                             normalized_mutual_info_score, silhouette_score)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import seeding as S  # noqa: E402
from src.kmeans import lloyd  # noqa: E402


def _read_csr(node):
    """从 h5ad 的 X 组读 CSR（兼容 encoding-type=csr_matrix 的写法）。"""
    if "shape" in node:
        shape = tuple(int(x) for x in node["shape"][:])
    else:
        shape = tuple(int(x) for x in node.attrs["shape"])
    data = np.asarray(node["data"])
    indices = np.asarray(node["indices"])
    indptr = np.asarray(node["indptr"])
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def _read_cat(obs, key):
    if key not in obs:
        return None
    node = obs[key]
    if hasattr(node, "keys") and "categories" in node.keys():
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in node["categories"][:]]
        codes = np.asarray(node["codes"])
        out = np.full(codes.shape[0], "Unknown", dtype=object)
        out[codes >= 0] = [cats[c] for c in codes[codes >= 0]]
        return out
    vals = node[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals], dtype=object)


def load_h5ad(path, max_cells=0, seed=0, label_key="cell_type", pca_key="X_pca"):
    import h5py
    f = h5py.File(path, "r")
    pcs = None
    if pca_key and "obsm" in f and pca_key in f["obsm"]:
        pcs = np.asarray(f["obsm"][pca_key][:], dtype=np.float64)
        print("使用已有 PCA：obsm/%s %s" % (pca_key, pcs.shape), flush=True)
    X = None if pcs is not None else _read_csr(f["X"])
    obs = f["obs"]
    label = None
    for key in (label_key, "cell_type", "celltype", "cell_type_fine", "annotation"):
        if key is None:
            continue
        label = _read_cat(obs, key)
        if label is not None:
            print("标签列：obs/%s（%d 类）" % (key, len(set(label.tolist()))), flush=True)
            break
    if label is None:
        print("没有找到 cell_type 标签列；obs 现有列：%s" % list(obs.keys())[:20], flush=True)
    f.close()
    n = pcs.shape[0] if pcs is not None else X.shape[0]
    if max_cells and n > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_cells, replace=False))
        if pcs is not None:
            pcs = pcs[idx]
        else:
            X = X[idx]
        label = label[idx] if label is not None else None
        print("下采样到 %d 个细胞" % max_cells, flush=True)
    return X, pcs, label


def preprocess(X, n_hvg=2000, n_pcs=50, chunk_genes=4000):
    X = X.astype(np.float32).tocsr()
    nnz = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = np.nonzero(nnz >= 3)[0]
    X = X[:, keep]
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    X = sp.diags(1e4 / tot) @ X
    X = X.tocsr()
    X.data = np.log1p(X.data)
    # 逐块算方差，避免一次性 densify 全部基因
    n_genes = X.shape[1]
    var = np.empty(n_genes, dtype=np.float64)
    for s in range(0, n_genes, chunk_genes):
        e = min(s + chunk_genes, n_genes)
        blk = X[:, s:e]
        m = np.asarray(blk.mean(axis=0)).ravel()
        sq = np.asarray(blk.multiply(blk).mean(axis=0)).ravel()
        var[s:e] = sq - m ** 2
    idx = np.sort(np.argsort(var)[-min(n_hvg, n_genes):])
    dense = np.asarray(X[:, idx].todense(), dtype=np.float32)
    del X
    pcs = PCA(n_components=int(n_pcs), svd_solver="randomized",
              random_state=0).fit_transform(dense)
    return np.ascontiguousarray(pcs, dtype=np.float64)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--max-cells", type=int, default=0, help="0 表示用全部细胞")
    p.add_argument("--K", type=int, default=0, help="0 表示用标签类别数")
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--sk-init", type=int, default=3, help="sklearn KMeans 的 n_init")
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--out", default="results/atlas_covid.csv")
    p.add_argument("--label-key", default="cell_type")
    p.add_argument("--pca-key", default="X_pca", help="空字符串表示不用现成 PCA、自己预处理")
    p.add_argument("--methods", default="kmeans,kmeanspp,kmeans_gs,kmeans_gs_g")
    p.add_argument("--include-sklearn", action="store_true",
                   help="add an sklearn KMeans reference row (not in the paper table)")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    X, pcs, labels = load_h5ad(args.path, args.max_cells, label_key=args.label_key,
                               pca_key=args.pca_key)
    if pcs is not None:
        Z = pcs
        print("读入 PCA %s（%.1f s）" % (Z.shape, time.perf_counter() - t0), flush=True)
    else:
        print("读入 %d 细胞 × %d 基因（%.1f s）"
              % (X.shape[0], X.shape[1], time.perf_counter() - t0), flush=True)
        t0 = time.perf_counter()
        Z = preprocess(X, args.n_hvg, args.n_pcs)
        del X
        print("预处理 + PCA -> %s（%.1f s）" % (Z.shape, time.perf_counter() - t0), flush=True)

    K = args.K or (
        len(set(x for x in labels.tolist() if x is not None))
        if labels is not None
        else 20
    )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in S.SEEDERS]
    if unknown:
        raise ValueError("Unknown seeding methods: %s" % ", ".join(unknown))
    rows = []
    for m in methods:
        candidates = []
        for r in range(args.restarts):
            rng = np.random.default_rng(42 + r)
            t0 = time.perf_counter()
            res = S.SEEDERS[m](Z, K, rng)
            t_seed = time.perf_counter() - t0
            t0 = time.perf_counter()
            lab, C, it = lloyd(Z, res.centers, max_iter=300, tol=1e-7)
            t_iter = time.perf_counter() - t0
            mse = float(np.mean(np.sum((Z - C[lab]) ** 2, axis=1)))
            candidates.append({
                "dataset": os.path.basename(args.path).split(".")[0],
                "method": m,
                "restart": r,
                "selected": False,
                "N": Z.shape[0],
                "K": K,
                "restarts": args.restarts,
                "seed_s": t_seed,
                "iter_s": t_iter,
                "total_s": t_seed + t_iter,
                "n_iter": it,
                "mse": mse,
                "ari": adjusted_rand_score(labels, lab) if labels is not None else np.nan,
                "nmi": normalized_mutual_info_score(labels, lab) if labels is not None else np.nan,
                "ami": adjusted_mutual_info_score(labels, lab) if labels is not None else np.nan,
                "silhouette": float(silhouette_score(
                    Z, lab, sample_size=min(len(Z), 5000), random_state=0
                )),
                "trials": float(np.mean(res.trials)) if res.trials is not None else np.nan,
                "fallbacks": int(res.fallbacks),
            })
        best_i = int(np.argmin([row["mse"] for row in candidates]))
        candidates[best_i]["selected"] = True
        rows.extend(candidates)
        best = candidates[best_i]
        print("  %-10s best restart=%d seed %.2f s + iters %.1f s = %.1f s, "
              "n_iter=%d, MSE=%.2f, ARI=%.4f"
              % (m, best["restart"], best["seed_s"], best["iter_s"], best["total_s"],
                 best["n_iter"], best["mse"], best["ari"]), flush=True)

    if args.include_sklearn:
        from sklearn.cluster import KMeans

        t0 = time.perf_counter()
        km = KMeans(n_clusters=K, n_init=args.sk_init, random_state=42).fit(Z)
        dt = time.perf_counter() - t0
        lab = km.labels_
        rows.append({
            "dataset": os.path.basename(args.path).split(".")[0],
            "method": "sklearn",
            "restart": -1,
            "selected": True,
            "N": Z.shape[0],
            "K": K,
            "restarts": args.sk_init,
            "seed_s": np.nan,
            "iter_s": np.nan,
            "total_s": dt,
            "n_iter": km.n_iter_,
            "mse": float(km.inertia_ / Z.shape[0]),
            "ari": adjusted_rand_score(labels, lab) if labels is not None else np.nan,
            "nmi": normalized_mutual_info_score(labels, lab) if labels is not None else np.nan,
            "ami": adjusted_mutual_info_score(labels, lab) if labels is not None else np.nan,
            "silhouette": float(silhouette_score(
                Z, lab, sample_size=min(len(Z), 5000), random_state=0
            )),
            "trials": np.nan,
            "fallbacks": 0,
        })
        print("  sklearn    total %.1f s, n_iter=%d, MSE=%.2f, ARI=%.4f"
              % (dt, km.n_iter_, rows[-1]["mse"], rows[-1]["ari"]), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\n写入 %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
