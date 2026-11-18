#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate the coordinate-subsampled scale estimator against the full estimator.

For every tier-A dataset and 20 seeds:
  * d0_coord  : coordinate subsample (s = min(d, 128)), m = 2000 points
  * d0_full   : all coordinates, m = 2000 points (v1 estimator)

Outputs the relative deviation of d0_coord from the full-data IQR estimate,
the run-to-run relative standard deviation of both, and the wall-clock cost of
the scale step.  This backs the paper's claim that the fast estimator changes
the scale by only a few percent.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import datasets as ds  # noqa: E402
from src.seeding import robust_scale  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="tierA")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--n-coords", type=int, default=128)
    p.add_argument("--n-points", type=int, default=2000)
    p.add_argument("--out", default="results/scale_validation.csv")
    args = p.parse_args()

    keys = ds.TIER_A if args.datasets == "tierA" else args.datasets.split(",")
    rows = []
    for key in keys:
        X, y, meta = ds.load(key)
        X = np.ascontiguousarray(StandardScaler().fit_transform(X), dtype=np.float64)
        d_full, d_coord = [], []
        t_full = t_coord = 0.0
        for r in range(args.repeats):
            rng = np.random.default_rng(42 + r)
            t0 = time.perf_counter()
            v, _, _ = robust_scale(X, rng, mode="full", n_points=args.n_points)
            t_full += time.perf_counter() - t0
            d_full.append(v)
            rng = np.random.default_rng(42 + r)
            t0 = time.perf_counter()
            v, _, s_used = robust_scale(X, rng, mode="coord", n_points=args.n_points,
                                        n_coords=args.n_coords)
            t_coord += time.perf_counter() - t0
            d_coord.append(v)
        d_full = np.asarray(d_full)
        d_coord = np.asarray(d_coord)
        rel = np.abs(d_coord - d_full) / np.maximum(d_full, 1e-12)
        rows.append({
            "dataset": key, "N": meta["N"], "d": meta["d"],
            "d0_full_mean": float(d_full.mean()),
            "d0_coord_mean": float(d_coord.mean()),
            "rel_err_mean": float(rel.mean()),
            "rel_err_max": float(rel.max()),
            "rsd_full": float(d_full.std(ddof=1) / d_full.mean()),
            "rsd_coord": float(d_coord.std(ddof=1) / d_coord.mean()),
            "ms_full": 1e3 * t_full / args.repeats,
            "ms_coord": 1e3 * t_coord / args.repeats,
            "n_coords_used": int(s_used),
        })
        r = rows[-1]
        print("%-18s d=%-4d  rel_err mean %.3f max %.3f | rsd %.3f/%.3f | "
              "scale %.2f -> %.2f ms"
              % (key, meta["d"], r["rel_err_mean"], r["rel_err_max"],
                 r["rsd_full"], r["rsd_coord"], r["ms_full"], r["ms_coord"]),
              flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nmean rel err %.3f  max %.3f  scale speedup %.1fx"
          % (df.rel_err_mean.mean(), df.rel_err_max.max(),
             df.ms_full.sum() / max(df.ms_coord.sum(), 1e-9)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
