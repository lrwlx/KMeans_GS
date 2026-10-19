#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""受控伸缩实验：只计时"播种"阶段，扫 N / K / d 三个方向。

三个生成机制（都先做逐列标准化，与真实流程一致）：
  separable  ：团心间距 6σ、团内各向同性，可分性好
  overlapping：团心间距 2σ，团之间明显重叠
  varying    ：团心间距 5σ，但各团样本量与方差按对数正态变化（变密度/不均衡）

三个扫描方向（每次只变一个因子，避免 N/d/K 混淆——这正是 20 个真实数据集的缺陷）：
  N : N∈{1e4,1e5,1e6,1e7}, K=100, d=20
  K : K∈{10,50,100,500},   N=1e6, d=20
  d : d∈{10,100},           N=1e6, K=100
另有 --corner：N=1e7,K=500,d=100 的单次极限点（只跑 kmeans_gs/afkmc2/kmeanspp）。

每个 (机制, 配置, 方法) 跑 repeats 次；任务级断点续跑（写 part csv）。

用法（H100，28 核并行、绑核 40 起）：
    python scripts/scaling_study.py --repeats 5 --jobs 28 --pin-base 40 \
        --out results/scaling_study
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import seeding as S  # noqa: E402


def make_data(n, k, d, mechanism, seed=0):
    """返回 (X, y)。三种机制的团心间距与团内方差不同。"""
    rng = np.random.default_rng(seed)
    sep = {"separable": 6.0, "overlapping": 2.0, "varying": 5.0}[mechanism]
    centers = rng.normal(scale=sep, size=(k, d))
    if mechanism == "varying":
        w = rng.lognormal(0.0, 0.8, size=k)
        w = w / w.sum()
        counts = np.maximum(2, np.floor(w * n).astype(np.int64))
        counts[-1] = n - counts[:-1].sum()
        scale = rng.lognormal(0.0, 0.6, size=k)
        y = np.repeat(np.arange(k), counts)
        X = centers[y] + rng.normal(size=(n, d)) * scale[y][:, None]
    else:
        y = rng.integers(0, k, size=n)
        X = centers[y] + rng.normal(size=(n, d))
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    return np.ascontiguousarray(X, dtype=np.float64), y


def build_tasks(ns, ks, ds, mechanisms):
    tasks = []
    for mech in mechanisms:
        for n in ns:
            tasks.append({"mechanism": mech, "sweep": "N", "N": n, "K": 100, "d": 20})
        for k in ks:
            tasks.append({"mechanism": mech, "sweep": "K", "N": 10 ** 6, "K": k, "d": 20})
        for d in ds:
            tasks.append({"mechanism": mech, "sweep": "d", "N": 10 ** 6, "K": 100, "d": d})
    return tasks


def tag_of(task):
    return "%s_%s_N%d_K%d_d%d" % (task["mechanism"], task["sweep"], task["N"], task["K"], task["d"])


_PIN = {"base": None, "counter": None}


def _init_worker(base, counter):
    _PIN["base"], _PIN["counter"] = base, counter
    if base is None or not hasattr(os, "sched_setaffinity"):
        return
    idx = 0
    try:
        with counter.get_lock():
            idx = int(counter.value)
            counter.value = idx + 1
    except Exception:
        idx = 0
    try:
        os.sched_setaffinity(0, {int(base) + idx})
    except Exception:
        pass


def run_task(task, methods, repeats, out_dir, seed_base=42):
    tag = tag_of(task)
    path = os.path.join(out_dir, "part_%s.csv" % tag)
    if os.path.exists(path):
        return tag, 0, "skip"
    X, y = make_data(task["N"], task["K"], task["d"], task["mechanism"], seed=seed_base)
    rows = []
    for m in methods:
        times, trials, fbs = [], [], 0
        for r in range(repeats):
            rng = np.random.default_rng(seed_base + 7 * r)
            t0 = time.perf_counter()
            try:
                res = S.SEEDERS[m](X, task["K"], rng)
            except Exception as exc:
                rows.append(dict(task, method=m, repeat=r, status="error",
                                 error="%s: %s" % (type(exc).__name__, exc),
                                 seed_time=np.nan, trials=np.nan, fallbacks=np.nan))
                continue
            dt = time.perf_counter() - t0
            times.append(dt)
            trials.append(float(np.mean(res.trials)) if res.trials is not None and res.trials.size else np.nan)
            fbs += int(res.fallbacks)
            rows.append(dict(task, method=m, repeat=r, status="ok", error="",
                             seed_time=dt, trials=trials[-1], fallbacks=int(res.fallbacks),
                             d0=float(res.d0) if np.isfinite(res.d0) else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return tag, len(df), "ok"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/scaling_study")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--pin-base", type=int, default=None)
    p.add_argument("--ns", default="10000,100000,1000000,10000000")
    p.add_argument("--ks", default="10,50,100,500")
    p.add_argument("--ds", default="10,100")
    p.add_argument("--mechanisms", default="separable,overlapping,varying")
    p.add_argument("--methods", default="kmeans_gs,kmeanspp,kmeans_par,afkmc2")
    p.add_argument("--corner", action="store_true",
                   help="额外跑 N=1e7,K=500,d=100 的极限点（kmeans_gs/afkmc2/kmeanspp，1 次）")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    ns = [int(x) for x in args.ns.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    ds = [int(x) for x in args.ds.split(",")]
    mechs = [x.strip() for x in args.mechanisms.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    def merge():
        files = sorted(glob.glob(os.path.join(args.out, "part_*.csv")))
        if not files:
            print("没有分片")
            return None
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(os.path.join(args.out, "scaling_all.csv"), index=False)
        ok = df[df.status == "ok"]
        agg = (ok.groupby(["mechanism", "sweep", "N", "K", "d", "method"])
               .agg(seed_mean=("seed_time", "mean"), seed_std=("seed_time", "std"),
                    trials=("trials", "mean"), fallbacks=("fallbacks", "sum"),
                    n=("seed_time", "size")).reset_index())
        agg.to_csv(os.path.join(args.out, "scaling_summary.csv"), index=False)
        print("合并 %d 个分片 -> %d 条原始记录、%d 行汇总"
              % (len(files), len(df), len(agg)))
        return agg

    if args.merge_only:
        merge()
        return 0

    tasks = build_tasks(ns, ks, ds, mechs)
    if args.corner:
        for mech in mechs:
            tasks.append({"mechanism": mech, "sweep": "corner", "N": 10 ** 7, "K": 500, "d": 100})

    print("任务数 %d（%d 机制 × 扫描 + 极限点），方法 %s，重复 %d，并行 %d"
          % (len(tasks), len(mechs), ",".join(methods), args.repeats, args.jobs), flush=True)

    if args.jobs <= 1:
        for t in tasks:
            try:
                tag, nrow, st = run_task(t, methods, args.repeats, args.out)
                print("[%s] %s (%d 行)" % (st, tag, nrow), flush=True)
            except Exception:
                print("[fail] %s\n%s" % (tag_of(t), traceback.format_exc()), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp

        counter = mp.Value("i", 0) if args.pin_base is not None else None
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                                 initargs=(args.pin_base, counter)) as pool:
            futs = {pool.submit(run_task, t, methods, args.repeats, args.out): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                t = futs[fut]
                try:
                    tag, nrow, st = fut.result()
                    print("[%d/%d][%s] %s (%d 行)" % (i, len(tasks), st, tag, nrow), flush=True)
                except Exception:
                    print("[fail] %s\n%s" % (tag_of(t), traceback.format_exc()), flush=True)

    agg = merge()
    print(json.dumps({"tasks": len(tasks), "methods": methods}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
