#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Choose DBSCAN epsilon for every tier-A dataset by a label-free silhouette
grid search, and write ``dbscan_eps.json`` for the main run.

Candidates come from the k-distance curve of a 3,000-point subsample; each
candidate is then run on the full data and scored by silhouette (noise points
assigned to the nearest centre).  Labels are never used.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import datasets as ds  # noqa: E402
from src.methods import choose_dbscan_eps  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="tierA")
    p.add_argument("--out", default="results/dbscan_eps.json")
    args = p.parse_args()

    keys = ds.TIER_A if args.datasets == "tierA" else args.datasets.split(",")
    eps = {"dbscan": {}}
    rows = []
    t_all = time.time()
    for key in keys:
        X, y, meta = ds.load(key)
        X = np.ascontiguousarray(StandardScaler().fit_transform(X), dtype=np.float64)
        t0 = time.time()
        best = choose_dbscan_eps(X, min_samples=5, sample=3000, n_grid=8, seed=42, max_noise=0.6, target_clusters=meta["K"])
        dt = time.time() - t0
        eps["dbscan"][key] = float(best["eps"])
        rows.append({"dataset": key, "N": meta["N"], "d": meta["d"],
                     "eps": best["eps"], "n_clusters": best["n_clusters"],
                     "noise_frac": best["noise_frac"],
                     "silhouette": best["silhouette"],
                     "n_candidates": len(best["grid"]), "seconds": dt})
        print("[eps] %-18s N=%-7d eps=%.4g clusters=%-3d noise=%.2f sil=%s (%.1fs)"
              % (key, meta["N"], best["eps"], best["n_clusters"],
                 best["noise_frac"],
                 ("%.3f" % best["silhouette"]) if np.isfinite(best["silhouette"]) else "nan",
                 dt), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(eps, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.splitext(args.out)[0] + ".csv", index=False)
    print("wrote %s (%.0f s total)" % (args.out, time.time() - t_all), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
