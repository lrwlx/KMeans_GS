# -*- coding: utf-8 -*-
"""IQR 子样本规模对初始阈值 d0 的影响（稳健性检查）。

KMeans-GS 的 d0 = 0.5*sqrt(sum_k IQR_k^2) 需要估计各维四分位距。若用全量数据
估计，代价 O(N*d)，"播种代价与 N 无关"这一结论就不成立；因此代码默认从数据中
有放回抽取固定规模子样本（--gs-iqr-sample，默认 2000）来估计。本脚本用重抽样
回答"2000 点是否够"这个问题：给出不同子样本规模下 d0 的均值、标准差、相对误差
以及单次估计耗时，并与全量数据的 d0 对比。

用法：
    python scripts/iqr_sensitivity.py --datasets musk,har,isolet,mnist_784
    python scripts/iqr_sensitivity.py --datasets har --m 500,1000,2000,4000,10000 --b 200
"""

from __future__ import annotations

# BLAS 线程锁定为 1（与 run_experiment.py 一致，避免计时抖动）
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import datasets as ds  # noqa: E402


def d0_from(X, seed=0):
    """按正文公式计算 d0 = 0.5*sqrt(sum_k IQR_k^2)。"""
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    return float(0.5 * np.sqrt(np.sum((q3 - q1) ** 2)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="iris,ecoli,musk,har,isolet,mnist_784")
    p.add_argument("--m", default="500,1000,2000,10000", help="子样本规模列表")
    p.add_argument("--b", type=int, default=100, help="每个规模重复的次数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/iqr_sensitivity.csv")
    p.add_argument("--no-standardize", dest="standardize", action="store_false", default=True)
    args = p.parse_args(argv)

    sizes = [int(s) for s in args.m.split(",") if s.strip()]
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    rng = np.random.default_rng(args.seed)
    rows = []

    for key in keys:
        X, y, meta = ds.load(key)
        if args.standardize:
            X = StandardScaler().fit_transform(X)
        X = np.ascontiguousarray(X, dtype=np.float64)
        n, d = X.shape
        t0 = time.perf_counter()
        full = d0_from(X)
        t_full = time.perf_counter() - t0
        print("== %s  N=%d d=%d  d0(全量)=%.4f  全量估计耗时=%.4fs" % (key, n, d, full, t_full))
        for m in sizes:
            mm = min(m, n)
            vals, times = [], []
            for _ in range(args.b):
                sub = X[rng.integers(0, n, size=mm)]
                t0 = time.perf_counter()
                vals.append(d0_from(sub))
                times.append(time.perf_counter() - t0)
            vals = np.asarray(vals)
            rel = vals.std(ddof=1) / vals.mean()
            print("   m=%-6d d0=%.4f +- %.4f（相对标准差 %.2f%%，最大偏差 %.2f%%）"
                  "  估计耗时=%.4fs（全量的 %.3f 倍）"
                  % (mm, vals.mean(), vals.std(ddof=1), 100 * rel,
                     100 * np.abs(vals - full).max() / full,
                     np.mean(times), np.mean(times) / t_full))
            rows.append({
                "dataset": key, "N": n, "d": d, "m": int(mm),
                "d0_mean": float(vals.mean()),
                "d0_sd": float(vals.std(ddof=1)),
                "relative_sd": float(rel),
                "max_relative_deviation_from_full": float(
                    np.abs(vals - full).max() / full
                ),
                "estimate_time_mean": float(np.mean(times)),
                "full_estimate_time": float(t_full),
                "time_ratio_vs_full": float(np.mean(times) / t_full),
                "d0_full": float(full),
            })
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("\n写入 %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
