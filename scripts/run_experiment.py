#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the KMeans-GS v2 benchmark.

Examples
--------
    python scripts/run_experiment.py --datasets tierA --methods all \
        --repeats 20 --jobs 24 --pin-base 0 --out results/main_tierA

The script pins one CPU core per worker and one BLAS thread per process so
that the timing columns are meaningful.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import datasets as ds  # noqa: E402
from src.methods import FULL_METHODS  # noqa: E402
from src.runner import (MAIN_METHODS, REF_METHODS, RunConfig, run_all)  # noqa: E402
from src.seeding import SEEDERS  # noqa: E402


def _parse_list(value, allowed):
    if value in ("all",):
        return list(allowed)
    if value == "main":
        return list(MAIN_METHODS)
    if value == "ref":
        return list(REF_METHODS)
    out = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in out if v not in allowed]
    if bad:
        raise SystemExit("unknown methods: %s (allowed: %s)" % (bad, sorted(allowed)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="tierA")
    p.add_argument("--methods", default="main")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--pin-base", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-seconds", type=float, default=3600.0)
    p.add_argument("--dbscan-eps-file", default="")
    p.add_argument("--scale-mode", default="coord", choices=["coord", "full"])
    p.add_argument("--n-coords", type=int, default=128)
    p.add_argument("--n-points", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--decay", default="inv_sqrt")
    p.add_argument("--g-batch", type=int, default=8)
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    all_methods = list(SEEDERS) + list(FULL_METHODS)
    if args.datasets in ("tierA", "A"):
        keys = ds.TIER_A
    elif args.datasets in ("tierB", "B"):
        keys = ds.TIER_B
    elif args.datasets == "all":
        keys = ds.TIER_A + ds.TIER_B
    else:
        keys = [v.strip() for v in args.datasets.split(",") if v.strip()]
    methods = _parse_list(args.methods, all_methods)

    dbscan_eps = {"dbscan": {}}
    if args.dbscan_eps_file and os.path.exists(args.dbscan_eps_file):
        with open(args.dbscan_eps_file, "r", encoding="utf-8") as f:
            dbscan_eps = json.load(f)

    cfg = RunConfig(
        repeats=args.repeats, methods=methods, max_iter=args.max_iter,
        tol=args.tol, seed_base=42,
        kmeans_gs_kwargs=dict(scale_mode=args.scale_mode, n_coords=args.n_coords,
                              n_points=args.n_points, alpha=args.alpha,
                              decay=args.decay, batch=args.g_batch),
        dbscan_eps=dbscan_eps, max_seconds=args.max_seconds,
    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "_config.json"), "w", encoding="utf-8") as f:
        json.dump({"datasets": keys, "methods": methods, "repeats": args.repeats,
                   "tol": args.tol, "max_iter": args.max_iter,
                   "scale_mode": args.scale_mode, "n_coords": args.n_coords,
                   "n_points": args.n_points, "alpha": args.alpha,
                   "decay": args.decay, "seed_base": 42}, f, indent=2)
    run_all(keys, cfg, args.out, jobs=args.jobs, job_mode="repeat",
            cpu_pin_base=args.pin_base, merge_only=args.merge_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
