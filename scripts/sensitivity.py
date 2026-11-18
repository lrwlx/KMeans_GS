#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sensitivity of KMeans-GS to its design constants.

Each setting changes exactly one component; every setting is run on the full
tier-A suite with 30 seeds.  The baseline is the configuration used in the
paper.  Results are written as one row per (setting, dataset, seed).
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import datasets as ds  # noqa: E402
from src.runner import RunConfig, run_single  # noqa: E402

BASE = dict(scale_mode="coord", n_points=512, n_coords=128, alpha=0.5,
            decay="inv_sqrt", first="random", max_trials=100, batch=1)

SETTINGS = [
    ("baseline", "KMeans-GS", {}, 1e-4),
    ("m=256", "n_points", {"n_points": 256}, 1e-4),
    ("m=1000", "n_points", {"n_points": 1000}, 1e-4),
    ("m=2000", "n_points", {"n_points": 2000}, 1e-4),
    ("s=64", "n_coords", {"n_coords": 64}, 1e-4),
    ("s=256", "n_coords", {"n_coords": 256}, 1e-4),
    ("scale=full", "scale_mode", {"scale_mode": "full"}, 1e-4),
    ("alpha=0.3", "alpha", {"alpha": 0.3}, 1e-4),
    ("alpha=0.7", "alpha", {"alpha": 0.7}, 1e-4),
    ("decay=0.3", "decay", {"decay": "power", "decay_p": 0.3}, 1e-4),
    ("decay=0.7", "decay", {"decay": "power", "decay_p": 0.7}, 1e-4),
    ("first=median", "first", {"first": "median_sub"}, 1e-4),
    ("trials=20", "max_trials", {"max_trials": 20}, 1e-4),
    ("trials=200", "max_trials", {"max_trials": 200}, 1e-4),
    ("batch=8 (first)", "batch", {"batch": 8}, 1e-4),
    ("tol=1e-6", "tol", {}, 1e-6),
    ("GS-G b=8", "farthest", {"batch": 8, "select": "farthest"}, 1e-4),
    ("GS-G b=32", "farthest", {"batch": 32, "select": "farthest"}, 1e-4),
]


def run_dataset(key, seeds, out_path):
    X, y, meta = ds.load(key)
    X = np.ascontiguousarray(StandardScaler().fit_transform(X), dtype=np.float64)
    K = int(meta["K"])
    rows = []
    for name, comp, override, tol in SETTINGS:
        kw = dict(BASE)
        kw.update(override)
        cfg = RunConfig(repeats=1, methods=["kmeans_gs"], tol=tol,
                        kmeans_gs_kwargs=kw, max_iter=300,
                        max_seconds=1800.0)
        for seed in seeds:
            row = run_single(X, y, K, "kmeans_gs", cfg, seed, key)
            row.update({"setting": name, "component": comp, "tol": tol,
                        "dataset": key, "N": meta["N"], "d": meta["d"],
                        "K": K, "seed": seed})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return key, len(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="tierA")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--pin-base", type=int, default=0)
    p.add_argument("--out", default="results/sensitivity")
    args = p.parse_args()

    keys = ds.TIER_A if args.datasets == "tierA" else args.datasets.split(",")
    seeds = list(range(42, 42 + args.seeds))
    os.makedirs(os.path.join(args.out, "_tasks"), exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    def _init(base, counter):
        try:
            with counter.get_lock():
                idx = int(counter.value)
                counter.value = idx + 1
            os.sched_setaffinity(0, {base + idx})
        except Exception:
            pass

    counter = mp.Value("i", 0)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(keys)),
                             initializer=_init,
                             initargs=(args.pin_base, counter)) as pool:
        futs = {pool.submit(run_dataset, k, seeds,
                            os.path.join(args.out, "_tasks", k + ".csv")): k
                for k in keys}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                key, n = fut.result()
                print("[done] %-18s %d rows" % (key, n), flush=True)
            except Exception as exc:
                print("[fail] %s: %r" % (k, exc), flush=True)

    frags = [os.path.join(args.out, "_tasks", f)
             for f in os.listdir(os.path.join(args.out, "_tasks"))
             if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(f) for f in sorted(frags)], ignore_index=True)
    df.to_csv(os.path.join(args.out, "sensitivity_raw.csv"), index=False)
    print("wrote %s (%d rows, %.0f s)"
          % (os.path.join(args.out, "sensitivity_raw.csv"), len(df),
             time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
