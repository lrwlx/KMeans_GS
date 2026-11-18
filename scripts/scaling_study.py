#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Controlled seeding-cost study over N, K and d on synthetic data.

Only the seeding call is timed (no Lloyd), so the measurement isolates the
initialisation stage.  Each worker generates its own data, runs every method
once on it, and writes one fragment; the analysis fits log-log exponents per
mechanism.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import seeding as S  # noqa: E402

ALL = ["kmeans", "kmeanspp", "kmeanspp_greedy", "kmeanspp_sklearn",
       "kmeans_par", "afkmc2", "kmeans_gs", "kmeans_gs_g"]
CORNER = ["kmeans", "kmeanspp", "kmeanspp_sklearn", "afkmc2",
          "kmeans_gs", "kmeans_gs_g"]


def make_data(n, d, K, mechanism, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 2.0, size=(K, d))
    if mechanism == "separable":
        sd = np.full(K, 0.35)
    elif mechanism == "overlapping":
        sd = np.full(K, 1.0)
    elif mechanism == "varying":
        sd = 0.35 + rng.random(K) * 1.6
    else:
        raise ValueError(mechanism)
    sizes = rng.multinomial(n - K, np.ones(K) / K) + 1
    X = np.empty((n, d), dtype=np.float64)
    pos = 0
    for k in range(K):
        c = int(sizes[k])
        X[pos:pos + c] = centers[k] + rng.normal(0.0, 1.0, size=(c, d)) * sd[k]
        pos += c
    return np.ascontiguousarray(X)


def run_one(mechanism, n, K, d, rep, methods, out_dir):
    X = make_data(n, d, K, mechanism, seed=2026 + 1000 * rep + (n*7 + K*131 + d*17) % 997)
    rows = []
    for m in methods:
        rng = np.random.default_rng(42 + rep)
        try:
            t0 = time.perf_counter()
            res = S.SEEDERS[m](X, K, rng)
            dt = time.perf_counter() - t0
            rows.append({"mechanism": mechanism, "N": n, "K": K, "d": d,
                         "rep": rep, "method": m, "seed_s": dt,
                         "trials": float(np.mean(res.trials))
                         if res.trials is not None else np.nan,
                         "fallbacks": int(res.fallbacks), "status": "ok"})
        except Exception as exc:
            rows.append({"mechanism": mechanism, "N": n, "K": K, "d": d,
                         "rep": rep, "method": m, "seed_s": np.nan,
                         "trials": np.nan, "fallbacks": 0,
                         "status": "%s: %s" % (type(exc).__name__, exc)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(out_dir, "_tasks"), exist_ok=True)
    tag = "%s_n%d_k%d_d%d_r%d" % (mechanism, n, K, d, rep)
    df.to_csv(os.path.join(out_dir, "_tasks", tag + ".csv"), index=False)
    return tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/scaling")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--corner-reps", type=int, default=3)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--pin-base", type=int, default=0)
    p.add_argument("--only", default="", help="comma list of sweeps: N,K,d,corner")
    args = p.parse_args()

    sweeps = {
        "N": [dict(n=n, K=100, d=20) for n in (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7)],
        "K": [dict(n=10 ** 6, K=k, d=20) for k in (10, 50, 100, 500)],
        "d": [dict(n=10 ** 6, K=100, d=d) for d in (10, 100)],
        "corner": [dict(n=10 ** 7, K=500, d=100)],
    }
    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or list(sweeps)
    mechs = ("separable", "overlapping", "varying")

    tasks = []
    for name in wanted:
        for spec in sweeps[name]:
            reps = args.corner_reps if name == "corner" else args.reps
            methods = CORNER if name == "corner" else ALL
            for mech in mechs:
                for r in range(reps):
                    tasks.append((mech, spec["n"], spec["K"], spec["d"], r, methods))
    print("scaling tasks: %d" % len(tasks), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tasks.json"), "w") as f:
        json.dump([[t[0], t[1], t[2], t[3], t[4], t[5]] for t in tasks], f)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _init(base, counter):
        if base is not None and hasattr(os, "sched_setaffinity"):
            try:
                with counter.get_lock():
                    idx = int(counter.value)
                    counter.value = idx + 1
                os.sched_setaffinity(0, {base + idx})
            except Exception:
                pass

    import multiprocessing as mp
    counter = mp.Value("i", 0) if args.pin_base is not None else None
    weights = [min(t[1] * t[3], 2_000_000_000) for t in tasks]  # n*d as weight
    order = np.argsort(weights)[::-1]
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(args.pin_base, counter)) as pool:
        futs = {pool.submit(run_one, *tasks[i], out_dir=args.out): i for i in order}
        for fut in as_completed(futs):
            done += 1
            try:
                fut.result()
                print("[done] %d/%d" % (done, len(tasks)), flush=True)
            except Exception as exc:
                print("[fail] %r" % (exc,), flush=True)

    frags = [f for f in os.listdir(os.path.join(args.out, "_tasks"))
             if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(os.path.join(args.out, "_tasks", f))
                    for f in sorted(frags)], ignore_index=True)
    df.to_csv(os.path.join(args.out, "scaling_raw.csv"), index=False)
    ok = df[df.status == "ok"]
    print(ok.groupby(["mechanism", "N", "K", "d", "method"]).seed_s.mean()
          .to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
