#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为每个数据集选一次 DBSCAN 的 eps（无监督网格搜索，不使用真实标签）。

做法：在不超过 sample 个点的子样本上，取第 min_samples 近邻距离曲线的若干分位点作为
eps 候选，逐一跑 DBSCAN，用轮廓系数（噪声点按最近中心指派）挑选最好的 eps；要求至少
找到 min_clusters 个簇、噪声比例不超过 max_noise。选出的 eps 在 30 次重复中固定不变，
写入 CSV 后由 run_experiment.py --dbscan-eps-file 注入。

用法：
    python scripts/choose_dbscan_eps.py --out results/dbscan_eps.csv
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
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import datasets as ds  # noqa: E402
from src.methods import choose_dbscan_eps  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="all", choices=["all", "synth", "uci"])
    p.add_argument("--datasets", default="")
    p.add_argument("--out", default="results/dbscan_eps.csv")
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--sample", type=int, default=3000)
    p.add_argument("--n-grid", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-points", type=int, default=20000,
                   help="超过该样本数则跳过（与 run_experiment 的 DBSCAN 上限一致）")
    p.add_argument("--verbose", action="store_true", help="打印每个候选 eps 的评估结果")
    p.add_argument("--data-home", default=None)
    args = p.parse_args(argv)

    keys = ([k.strip() for k in args.datasets.split(",") if k.strip()]
            if args.datasets else ds.keys(args.group))
    rows = []
    for key in keys:
        spec = ds.DATASETS[key]
        if spec["n"] > args.max_points:
            print("[skip] %-20s N=%d 超过上限 %d" % (key, spec["n"], args.max_points), flush=True)
            continue
        X, y, meta = ds.load(key, data_home=args.data_home)
        X = np.ascontiguousarray(StandardScaler().fit_transform(X), dtype=np.float64)
        t0 = time.perf_counter()
        res = choose_dbscan_eps(X, min_samples=args.min_samples, sample=args.sample,
                                n_grid=args.n_grid, seed=args.seed, verbose=args.verbose,
                                target_clusters=meta["K"])
        dt = time.perf_counter() - t0
        rows.append({"dataset": key, "N": meta["N"], "d": meta["d"], "K": meta["K"],
                     "eps": res["eps"], "silhouette": res["silhouette"],
                     "n_clusters": res["n_clusters"], "noise_frac": res["noise_frac"],
                     "eps_k": res["eps_k"], "n_clusters_k": res["n_clusters_k"],
                     "noise_frac_k": res["noise_frac_k"], "silhouette_k": res["silhouette_k"],
                     "n_candidates": res["n_candidates"], "sample": res["sample"],
                     "min_samples": args.min_samples, "seconds": dt})
        print("[ok] %-20s N=%-6d d=%-4d K=%-3d eps=%.4g sil=%.3f 簇数=%-4d 噪声=%.2f"
              " | eps_k=%.4g 簇数=%-4d 噪声=%.2f (%.1fs)"
              % (key, meta["N"], meta["d"], meta["K"], res["eps"], res["silhouette"],
                 res["n_clusters"], res["noise_frac"], res["eps_k"], res["n_clusters_k"],
                 res["noise_frac_k"], dt), flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print("\n写入 %s（%d 个数据集）" % (os.path.abspath(args.out), len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
