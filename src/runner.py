# -*- coding: utf-8 -*-
"""Experiment driver for the KMeans-GS v2 study.

One row is written per (dataset, method, repetition) and contains timing,
quality and seeding diagnostics.  The driver supports repeat-level
parallelism with one CPU core per worker, which keeps timing reproducible.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import datasets as ds
from .kmeans import lloyd
from .methods import FULL_LABELS, FULL_METHODS, FullResult
from .metrics import (mse_to_centers, safe_davies_bouldin, safe_silhouette,
                      seed_separation, supervised_scores, theorem_bound)
from .seeding import METHOD_LABELS, SEEDERS, SeedResult

METHOD_SPECS: Dict[str, Dict] = {}
for _k in SEEDERS:
    METHOD_SPECS[_k] = {"kind": "seed", "label": METHOD_LABELS[_k]}
for _k, _v in FULL_METHODS.items():
    METHOD_SPECS[_k] = {"kind": "full", "label": _v["label"]}

# Methods whose cost is a single seeding call followed by the shared Lloyd.
MAIN_METHODS = ["kmeans", "kmeanspp", "kmeanspp_greedy", "kmeanspp_sklearn",
                "kmeans_par", "afkmc2", "kmeans_gs", "kmeans_gs_g",
                "mbkmeans", "birch", "gmm", "dbscan"]
# End-to-end references with their own Lloyd implementation.
REF_METHODS = ["sklearn1", "sklearn10"]


@dataclass
class RunConfig:
    repeats: int = 20
    methods: Sequence[str] = tuple(MAIN_METHODS)
    max_iter: int = 300
    tol: float = 1e-4
    standardize: bool = True
    sil_sample: int = 10000
    seed_base: int = 42
    kmeans_gs_kwargs: Dict = field(default_factory=dict)
    extra_kwargs: Dict = field(default_factory=dict)
    full_kwargs: Dict = field(default_factory=dict)
    dbscan_eps: Dict = field(default_factory=dict)
    max_seconds: Optional[float] = 3600.0

    def method_kwargs(self, method: str) -> Dict:
        if method in ("kmeans_gs", "kmeans_gs_g"):
            return dict(self.kmeans_gs_kwargs)
        if method in ("kmeans_par", "afkmc2"):
            return dict(self.extra_kwargs.get(method, {}))
        return {}

    def full_method_kwargs(self, method: str) -> Dict:
        kw = dict(FULL_METHODS[method].get("defaults", {}))
        kw.update(self.full_kwargs.get(method, {}))
        return kw


@contextlib.contextmanager
def _time_limit(seconds):
    if (seconds is None or not hasattr(signal, "SIGALRM")
            or threading.current_thread() is not threading.main_thread()):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError("run exceeded %.0f s" % float(seconds))

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _warmup(X, K, cfg: RunConfig):
    n = X.shape[0]
    m = min(n, 2000)
    Xs = np.ascontiguousarray(X[:m])
    Kw = int(min(K, max(2, len(np.unique(Xs[:, 0])))))
    for method in cfg.methods:
        spec = METHOD_SPECS.get(method)
        if spec is None:
            continue
        try:
            rng = np.random.default_rng(0)
            if spec["kind"] == "seed":
                res = SEEDERS[method](Xs, Kw, rng, **cfg.method_kwargs(method))
                lloyd(Xs, res.centers, max_iter=2, tol=cfg.tol)
            else:
                FULL_METHODS[method]["fn"](np.ascontiguousarray(Xs[:500]), Kw,
                                           rng, **cfg.full_method_kwargs(method))
        except Exception:
            pass


def run_single(X, y, K, method, cfg: RunConfig, seed: int,
               dataset_key: Optional[str] = None) -> Dict:
    spec = METHOD_SPECS[method]
    row = {"method": method, "method_label": spec["label"], "kind": spec["kind"],
           "repeat_seed": int(seed), "status": "ok", "error": "", "note": ""}
    rng = np.random.default_rng(seed)
    t_seed = t_iter = t_total = float("nan")
    try:
        with _time_limit(cfg.max_seconds):
            if spec["kind"] == "seed":
                t0 = time.perf_counter()
                res: SeedResult = SEEDERS[method](X, K, rng,
                                                  **cfg.method_kwargs(method))
                t_seed = time.perf_counter() - t0
                t0 = time.perf_counter()
                labels, centers, n_iter = lloyd(X, res.centers,
                                                max_iter=cfg.max_iter, tol=cfg.tol)
                t_iter = time.perf_counter() - t0
                t_total = t_seed + t_iter
                labels_eval = labels
                seed_diag = res
            else:
                kw = cfg.full_method_kwargs(method)
                eps_map = (cfg.dbscan_eps or {}).get(method, {})
                if dataset_key and dataset_key in eps_map:
                    kw["eps"] = float(eps_map[dataset_key])
                    kw["eps_source"] = "silhouette grid search"
                t0 = time.perf_counter()
                fres: FullResult = FULL_METHODS[method]["fn"](X, K, rng, **kw)
                t_total = time.perf_counter() - t0
                labels, centers, n_iter = fres.labels, fres.centers, fres.n_iter
                labels_eval = fres.labels_eval
                row["note"] = fres.note
                for k, v in (fres.extra or {}).items():
                    row[k] = v
                seed_diag = None
    except TimeoutError as exc:
        row.update(status="timeout", error=str(exc), time_seed=np.nan,
                   time_iter=np.nan, time_total=np.nan, n_iter=np.nan)
        return row
    except ValueError as exc:
        msg = str(exc)
        if "infeasible" in msg:
            row.update(status="infeasible", error=msg, time_seed=np.nan,
                       time_iter=np.nan, time_total=np.nan, n_iter=np.nan)
            return row
        row.update(status="error", error="%s: %s" % (type(exc).__name__, msg),
                   time_seed=np.nan, time_iter=np.nan, time_total=np.nan,
                   n_iter=np.nan)
        return row
    except Exception as exc:
        row.update(status="error", error="%s: %s" % (type(exc).__name__, exc),
                   time_seed=np.nan, time_iter=np.nan, time_total=np.nan,
                   n_iter=np.nan)
        return row

    row.update({"time_seed": t_seed, "time_iter": t_iter, "time_total": t_total,
                "n_iter": int(n_iter) if n_iter is not None else float("nan")})
    row.update(supervised_scores(y, labels))
    row["mse"] = mse_to_centers(X, labels_eval, centers)
    row["silhouette"] = safe_silhouette(X, labels_eval, sample=cfg.sil_sample,
                                        seed=seed)
    row["davies_bouldin"] = safe_davies_bouldin(X, labels_eval)
    if seed_diag is not None:
        row.update(seed_separation(seed_diag.centers))
        row["scale_d0"] = float(seed_diag.d0)
        row["scale_mode"] = seed_diag.scale_mode
        row["scale_n_points"] = int(seed_diag.scale_n_points)
        row["scale_n_coords"] = int(seed_diag.scale_n_coords)
        if seed_diag.trials is not None:
            tr = seed_diag.trials
            row["fallbacks"] = int(seed_diag.fallbacks)
            row["trials_mean"] = float(tr.mean()) if tr.size else float("nan")
            row["trials_max"] = int(tr.max()) if tr.size else 0
        if seed_diag.taus is not None:
            b = theorem_bound(seed_diag.taus, K)
            row["bound"] = b
            row["bound_ratio"] = (row["seed_mean_dist"] / b
                                  if np.isfinite(b) and b > 0 else float("nan"))
    return row


def _load(key, cfg):
    X, y, meta = ds.load(key)
    if cfg.standardize:
        X = StandardScaler().fit_transform(X)
    return np.ascontiguousarray(X, dtype=np.float64), np.asarray(y), meta


def _row_meta(row, key, meta, K):
    row.update({"dataset": key, "tier": meta["tier"], "group": meta["group"],
                "N": meta["N"], "K": int(K), "d": meta["d"]})
    return row


def run_dataset(key: str, cfg: RunConfig) -> pd.DataFrame:
    X, y, meta = _load(key, cfg)
    K = int(meta["K"])
    _warmup(X, K, cfg)
    rows = []
    for r in range(cfg.repeats):
        seed = cfg.seed_base + r
        for method in cfg.methods:
            rows.append(_row_meta(run_single(X, y, K, method, cfg, seed, key),
                                  key, meta, K))
    return pd.DataFrame(rows)


def run_one_repeat(key: str, r: int, cfg: RunConfig) -> pd.DataFrame:
    X, y, meta = _load(key, cfg)
    K = int(meta["K"])
    seed = cfg.seed_base + r
    rows = [_row_meta(run_single(X, y, K, m, cfg, seed, key), key, meta, K)
            for m in cfg.methods]
    return pd.DataFrame(rows)


def merge_fragments(out_dir: str, keys=None):
    task_dir = os.path.join(out_dir, "_tasks")
    if not os.path.isdir(task_dir):
        return
    frags = [f for f in os.listdir(task_dir) if f.endswith(".csv")]
    if not frags:
        return
    by_key: Dict[str, List[str]] = {}
    for f in frags:
        key = f.split("__r")[0]
        if keys is not None and key not in keys:
            continue
        by_key.setdefault(key, []).append(os.path.join(task_dir, f))
    for key, paths in sorted(by_key.items()):
        df = pd.concat([pd.read_csv(p) for p in sorted(paths)], ignore_index=True)
        df = df.sort_values(["repeat_seed", "method"]).reset_index(drop=True)
        df.to_csv(os.path.join(out_dir, key + ".csv"), index=False)
        print("[merge] %-18s <- %d fragments (%d rows)"
              % (key, len(paths), len(df)), flush=True)


def _init_worker(cpu_pin_base, counter=None):
    """Pin the worker to a core and trigger one-off library initialisation so
    that the first timed run does not pay it."""
    if cpu_pin_base is not None and hasattr(os, "sched_setaffinity"):
        idx = 0
        if counter is not None:
            try:
                with counter.get_lock():
                    idx = int(counter.value)
                    counter.value = idx + 1
            except Exception:
                idx = 0
        try:
            os.sched_setaffinity(0, {int(cpu_pin_base) + idx})
        except Exception:
            pass
    try:
        Xw = np.random.default_rng(0).normal(size=(128, 6))
        for m in FULL_METHODS:
            try:
                FULL_METHODS[m]["fn"](Xw, 2, np.random.default_rng(0),
                                      **FULL_METHODS[m].get("defaults", {}))
            except Exception:
                pass
        for m in ("kmeanspp_sklearn", "kmeanspp_greedy", "kmeans_gs"):
            try:
                SEEDERS[m](Xw, 2, np.random.default_rng(0))
            except Exception:
                pass
    except Exception:
        pass


def _task(fn, key, r, cfg, out_dir):
    if fn == "repeat":
        df = run_one_repeat(key, r, cfg)
        os.makedirs(os.path.join(out_dir, "_tasks"), exist_ok=True)
        df.to_csv(os.path.join(out_dir, "_tasks", "%s__r%d.csv" % (key, r)),
                  index=False)
        return key, r, len(df)
    df = run_dataset(key, cfg)
    df.to_csv(os.path.join(out_dir, key + ".csv"), index=False)
    return key, None, len(df)


def run_all(keys: Sequence[str], cfg: RunConfig, out_dir: str, jobs: int = 1,
            resume: bool = True, job_mode: str = "repeat",
            cpu_pin_base: Optional[int] = None, merge_only: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    meta_rows = []
    for key in keys:
        spec = ds.DATASETS[key]
        X, y, meta = ds.load(key)
        meta_rows.append({"dataset": key, "tier": meta["tier"],
                          "group": meta["group"], "N": meta["N"],
                          "d": meta["d"], "K": meta["K"],
                          "source": meta["source"]})
    pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "_datasets.csv"),
                                   index=False)
    if merge_only:
        merge_fragments(out_dir, keys)
        return

    todo = []
    if job_mode == "repeat":
        for key in keys:
            for r in range(cfg.repeats):
                frag = os.path.join(out_dir, "_tasks", "%s__r%d.csv" % (key, r))
                if resume and os.path.exists(frag):
                    continue
                todo.append((key, r))
    else:
        for key in keys:
            if resume and os.path.exists(os.path.join(out_dir, key + ".csv")):
                continue
            todo.append((key, None))
    # heavy datasets first so that the pool stays balanced
    todo.sort(key=lambda kr: -ds.DATASETS[kr[0]].get("K", 0))

    if not todo:
        print("nothing to run", flush=True)
        merge_fragments(out_dir, keys)
        return
    print("tasks: %d (mode=%s, jobs=%d, pin=%s)"
          % (len(todo), job_mode, jobs, cpu_pin_base), flush=True)

    if jobs and jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        counter = mp.Value("i", 0) if cpu_pin_base is not None else None
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(cpu_pin_base, counter)) as pool:
            futs = {pool.submit(_task, job_mode, k, r, cfg, out_dir): (k, r)
                    for k, r in todo}
            done = 0
            for fut in as_completed(futs):
                key, r = futs[fut]
                done += 1
                try:
                    fut.result()
                    if done % 10 == 0 or done == len(todo):
                        print("[done] %s r=%s (%d/%d)"
                              % (key, r, done, len(todo)), flush=True)
                except Exception:
                    print("[fail] %s r=%s\n%s" % (key, r, traceback.format_exc()),
                          flush=True)
    else:
        for k, r in todo:
            try:
                _task(job_mode, k, r, cfg, out_dir)
                print("[done] %s r=%s" % (k, r), flush=True)
            except Exception:
                print("[fail] %s r=%s\n%s" % (k, r, traceback.format_exc()),
                      flush=True)
    merge_fragments(out_dir, keys)
