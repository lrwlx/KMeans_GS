#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实测 KMeans-GS 的"一簇一种子"性质，以及定理 3、命题 1、命题 2 的几何量。

对每个数据集 × 每个重复：
  * 用正文口径跑 seed_kmeans_gs，拿到 K 个种子在 X 中的行号；
  * 用真实标签检查：K 个种子是否落在 K 个不同簇（双射）、碰撞数；
  * 计算种子划分（每个点分到最近种子）的 ARI —— 这是"初始化质量"的直接度量；
  * 计算团的几何量：半径 rho=max_k max_{x in C_k}||x-mu_k||、团心最小距 Delta、
    kappa=d0/rho、分离比 Delta/rho；
  * 计算命题 1 的 spread 系数 beta = min_k (在 C_k 内、距该簇种子 > tau_2 的点占比)。

输出逐次记录 + 汇总。

用法：
    python scripts/seed_assignment.py --out results/seed_assignment.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import datasets as ds  # noqa: E402
from src import seeding as S  # noqa: E402


def nearest_labels(X, centers, chunk=65536):
    out = np.empty(X.shape[0], dtype=np.int64)
    cn = np.einsum("ij,ij->i", centers, centers)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        Xb = X[s:e]
        d2 = np.einsum("ij,ij->i", Xb, Xb)[:, None] - 2.0 * (Xb @ centers.T) + cn[None, :]
        out[s:e] = np.argmin(d2, axis=1)
    return out


def theorem_bound(taus, K):
    if taus is None or K < 2:
        return float("nan")
    i = np.arange(2, K + 1, dtype=np.float64)
    return float(2.0 / (K * (K - 1)) * np.sum((i - 1.0) * taus))


def seed_mean_distance(centers):
    K = centers.shape[0]
    if K < 2:
        return float("nan")
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    return float(d[np.triu_indices(K, k=1)].mean())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/seed_assignment.csv")
    p.add_argument("--data-home", default=None)
    p.add_argument("--gs-first", default="random")
    p.add_argument("--methods", default="kmeans_gs,kmeans_gs_g")
    args = p.parse_args(argv)

    keys = ([k.strip() for k in args.datasets.split(",") if k.strip()]
            if args.datasets else ds.keys("all"))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("At least one method is required")
    rows = []
    for key in keys:
        X, y, meta = ds.load(key, data_home=args.data_home)
        X = np.ascontiguousarray(StandardScaler().fit_transform(X), dtype=np.float64)
        y = np.asarray(y)
        K = int(meta["K"])
        labs = np.unique(y)
        if labs.size != K:
            # 真实标签数可能多于 meta["K"]，以实际标签数为准并记录
            K_eff = labs.size
        else:
            K_eff = K

        # 团的几何量（用真实标签）
        mu = np.array([X[y == l].mean(0) for l in labs])
        rho = max(float(np.max(np.linalg.norm(X[y == l] - mu[i], axis=1)))
                  for i, l in enumerate(labs))
        D = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        Delta = float(D.min())

        for method in methods:
            if method not in S.SEEDERS:
                raise ValueError("Unknown method: %s" % method)
            for r in range(args.repeats):
                rng = np.random.default_rng(args.seed_base + r)
                kwargs = {"first": args.gs_first} if method == "kmeans_gs" else {}
                res = S.SEEDERS[method](X, K, rng, **kwargs)
                idx = res.indices
                seed_lab = y[idx]
                n_unique = len(set(seed_lab.tolist()))
                bij = n_unique == K
                coll = K - n_unique
                ari_seed = adjusted_rand_score(y, nearest_labels(X, res.centers))
                tau2 = res.d0 / np.sqrt(2.0)
                betas = []
                for j, l in enumerate(seed_lab):
                    pts = X[y == l]
                    if pts.shape[0] < 2:
                        continue
                    d = np.linalg.norm(pts - res.centers[j], axis=1)
                    betas.append(float(np.mean(d > tau2)))
                bound = theorem_bound(res.taus, K)
                mean_dist = seed_mean_distance(res.centers)
                rows.append({
                    "dataset": key, "method": method, "N": meta["N"], "d": meta["d"],
                    "K": K, "K_labels": K_eff, "repeat": r,
                    "bijection": bool(bij), "collisions": int(coll),
                    "ari_seed_partition": float(ari_seed),
                    "rho": rho, "Delta": Delta, "Delta_over_rho": Delta / rho,
                    "d0": float(res.d0), "kappa": float(res.d0) / rho,
                    "cond_d0_gt_2rho_sqrtK": bool(res.d0 > 2.0 * rho * np.sqrt(K)),
                    "cond_Delta_gt_4rho": bool(Delta > 4.0 * rho),
                    "beta": float(np.min(betas)) if betas else np.nan,
                    "trials_mean": float(res.trials.mean()) if res.trials is not None
                    and res.trials.size else np.nan,
                    "fallbacks": int(res.fallbacks),
                    "seed_mean_dist": mean_dist,
                    "theorem_bound": bound,
                    "bound_ratio": mean_dist / bound if np.isfinite(bound) and bound > 0
                    else np.nan,
                })
            sub = [r_ for r_ in rows if r_["dataset"] == key and r_["method"] == method]
            print("[%s/%s] N=%d d=%d K=%d 双射率=%.2f 种子划分ARI=%.3f "
                  "kappa=%.2f Delta/rho=%.2f beta=%.2f bound_ratio=%.2f"
                  % (key, method, meta["N"], meta["d"], K,
                     np.mean([r_["bijection"] for r_ in sub]),
                     np.mean([r_["ari_seed_partition"] for r_ in sub]),
                     np.mean([r_["kappa"] for r_ in sub]),
                     np.mean([r_["Delta_over_rho"] for r_ in sub]),
                     np.nanmean([r_["beta"] for r_ in sub]),
                     np.nanmean([r_["bound_ratio"] for r_ in sub])), flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    ratio = (
        out.groupby(["dataset", "method"], as_index=False)[
            ["bound_ratio", "seed_mean_dist", "theorem_bound", "trials_mean", "fallbacks"]
        ]
        .mean()
    )
    ratio_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                              "separation_ratio.csv")
    ratio.to_csv(ratio_path, index=False)
    print("\n写入 %s（%d 行）" % (os.path.abspath(args.out), len(out)))
    print("写入 %s" % ratio_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
