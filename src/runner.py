# -*- coding: utf-8 -*-
"""实验调度：对每个数据集跑各方法 × 多次重复，记录时间与质量指标。

方法分两类
----------
* kind="seed"：播种 + 共享 Lloyd 迭代（kmeans / kmeanspp / kmeans_gs / kmeans_par / afkmc2），
  时间拆成 time_seed 与 time_iter；
* kind="full"：自带完整流程的算法（n_init=10、DBSCAN、BIRCH、GMM），只记 time_total，
  time_seed / time_iter 留空，具体超参数写进 note 列。

输出为"每次运行一行"的原始记录，汇总交给 scripts/summarize_results.py。
原始记录保留下来才能做配对 t 检验。
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
                      seed_separation, supervised_scores)
from .seeding import METHOD_LABELS, SEEDERS, SeedResult

# 方法登记表：name -> {kind, label}
METHOD_SPECS: Dict[str, Dict] = {}
for _k in SEEDERS:
    METHOD_SPECS[_k] = {"kind": "seed", "label": METHOD_LABELS[_k]}
for _k, _v in FULL_METHODS.items():
    METHOD_SPECS[_k] = {"kind": "full", "label": _v["label"]}


@dataclass
class RunConfig:
    repeats: int = 30
    methods: Sequence[str] = ("kmeans", "kmeanspp", "kmeans_gs")
    max_iter: int = 300
    tol: float = 1e-7
    standardize: bool = True
    sil_sample: int = 10000
    seed_base: int = 42
    kmeanspp_impl: str = "optimized"
    kmeans_gs_kwargs: Dict = field(default_factory=dict)
    # 其它播种方法（kmeans_par / afkmc2）的参数，按方法名分组
    extra_kwargs: Dict = field(default_factory=dict)
    # 完整算法基线（mbkmeans / dbscan / birch / gmm）的参数，按方法名分组
    full_kwargs: Dict = field(default_factory=dict)
    # 方法 -> (数据集 -> eps)：DBSCAN 的 eps，由 scripts/choose_dbscan_eps.py 预先搜索得到
    dbscan_eps: Dict = field(default_factory=dict)
    max_seconds: Optional[float] = None                # 单个 (数据集, 方法, 重复) 的硬上限

    @staticmethod
    def _sub_dict(bag: Dict, method: str) -> Dict:
        """取按方法名分组的参数字典；若传入的是扁平字典则整体返回（兼容旧写法）。"""
        if not bag:
            return {}
        if any(isinstance(v, dict) for v in bag.values()):
            sub = bag.get(method)
            return dict(sub) if isinstance(sub, dict) else {}
        return dict(bag)

    def method_kwargs(self, method: str) -> Dict:
        if method == "kmeanspp":
            return {"impl": self.kmeanspp_impl}
        if method in ("kmeans_gs", "kmeans_gs_g", "kmeans_gs_w"):
            return dict(self.kmeans_gs_kwargs)
        if method in ("kmeans_par", "afkmc2"):
            return self._sub_dict(self.extra_kwargs, method)
        return {}

    def full_method_kwargs(self, method: str) -> Dict:
        kw = self._sub_dict(self.full_kwargs, method)
        return kw


@contextlib.contextmanager
def _time_limit(seconds):
    """给单次运行加硬超时（Linux 下用 SIGALRM；不支持时退化为不设限）。"""
    if (seconds is None or not hasattr(signal, "SIGALRM")
            or threading.current_thread() is not threading.main_thread()):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError("超过单次运行上限 %.0f 秒" % float(seconds))

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _kmeans_gs_theorem_bound(taus, K):
    """定理 2：seed_mean_dist 的下界 2/(K(K-1)) * sum_{i=2..K} (i-1) * tau_i。"""
    if taus is None or K < 2:
        return float("nan")
    idx = np.arange(2, K + 1, dtype=np.float64)
    return float(2.0 / (K * (K - 1)) * np.sum((idx - 1.0) * taus))


def _warmup(X, y, K, cfg):
    """正式计时前的预热：让 BLAS 线程池、内存分配器进入稳定状态（不写入结果）。"""
    n = X.shape[0]
    m = min(n, 2000)
    Xs = np.ascontiguousarray(X[:m])
    Kw = min(K, max(2, len(np.unique(y[:m]))))
    for method in cfg.methods:
        spec = METHOD_SPECS.get(method)
        if spec is None:
            continue
        try:
            rng = np.random.default_rng(0)
            if spec["kind"] == "seed":
                res = SEEDERS[method](Xs, Kw, rng, **cfg.method_kwargs(method))
                lloyd(Xs, res.centers, max_iter=3, tol=cfg.tol)
            else:
                FULL_METHODS[method]["fn"](np.ascontiguousarray(Xs[:500]), Kw, rng,
                                           **cfg.full_method_kwargs(method))
        except Exception:
            pass


def run_single(X, y, K, method, cfg: RunConfig, seed: int,
               dataset_key: Optional[str] = None) -> Dict:
    """跑一次 (方法, 种子)，返回一行结果。dataset_key 用于注入数据集级超参数。"""
    spec = METHOD_SPECS[method]
    row = {"method": method, "method_label": spec["label"], "kind": spec["kind"],
           "repeat_seed": int(seed), "status": "ok", "error": "", "note": ""}
    rng = np.random.default_rng(seed)
    try:
        with _time_limit(cfg.max_seconds):
            if spec["kind"] == "seed":
                t0 = time.perf_counter()
                res: SeedResult = SEEDERS[method](X, K, rng, **cfg.method_kwargs(method))
                t_seed = time.perf_counter() - t0

                t0 = time.perf_counter()
                labels, centers, n_iter = lloyd(X, res.centers, max_iter=cfg.max_iter,
                                                tol=cfg.tol)
                t_iter = time.perf_counter() - t0
                labels_eval = labels
                seed_diag = res
            else:
                kw = cfg.full_method_kwargs(method)
                eps_map = (cfg.dbscan_eps or {}).get(method, {})
                if dataset_key and dataset_key in eps_map:
                    kw["eps"] = float(eps_map[dataset_key])
                    kw["eps_source"] = ("轮廓系数网格搜索" if method == "dbscan"
                                        else "网格搜索取簇数最接近 K")
                t0 = time.perf_counter()
                fres: FullResult = FULL_METHODS[method]["fn"](X, K, rng, **kw)
                t_total = time.perf_counter() - t0
                t_seed = t_iter = float("nan")
                labels, centers, n_iter = fres.labels, fres.centers, fres.n_iter
                labels_eval = fres.labels_eval
                row["note"] = fres.note
                for k, v in (fres.extra or {}).items():
                    row[k] = v
                seed_diag = None
    except TimeoutError as exc:
        row["status"] = "timeout"
        row["error"] = str(exc)
        row["time_seed"] = row["time_iter"] = row["time_total"] = float("nan")
        return row
    except Exception as exc:  # 单个配置失败不应中断整个实验
        row["status"] = "error"
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        row["time_seed"] = row["time_iter"] = row["time_total"] = float("nan")
        return row

    row.update({
        "time_seed": t_seed,
        "time_iter": t_iter,
        "time_total": t_total if spec["kind"] == "full" else (t_seed + t_iter),
        "n_iter": int(n_iter) if n_iter is not None else float("nan"),
    })
    row.update(supervised_scores(y, labels))
    row["mse"] = mse_to_centers(X, labels_eval, centers)
    row["silhouette"] = safe_silhouette(X, labels_eval, sample=cfg.sil_sample, seed=seed)
    row["davies_bouldin"] = safe_davies_bouldin(X, labels_eval)
    if seed_diag is not None:
        row.update(seed_separation(seed_diag.centers))

    # 拒绝采样类方法（KMeans-GS / AFK-MC²）的播种诊断
    if seed_diag is not None and seed_diag.trials is not None:
        trials = seed_diag.trials
        fallbacks = int(seed_diag.fallbacks)
        row["kmeans_gs_d0"] = float(seed_diag.d0)
        row["kmeans_gs_iqr_n"] = int(seed_diag.iqr_n_used)
        row["kmeans_gs_fallbacks"] = fallbacks
        row["kmeans_gs_trials_mean"] = float(trials.mean()) if trials.size else float("nan")
        row["kmeans_gs_trials_max"] = int(trials.max()) if trials.size else 0
        if seed_diag.taus is not None:
            bound = _kmeans_gs_theorem_bound(seed_diag.taus, K)
            row["kmeans_gs_bound"] = bound
            row["kmeans_gs_bound_ratio"] = (row["seed_mean_dist"] / bound
                                     if np.isfinite(bound) and bound > 0 else float("nan"))
    return row


def _load(key, cfg, data_home):
    X, y, meta = ds.load(key, data_home=data_home)
    if cfg.standardize:
        X = StandardScaler().fit_transform(X)
    return np.ascontiguousarray(X, dtype=np.float64), np.asarray(y), meta


def _row_meta(row, key, meta, K, r):
    row.update({"dataset": key, "group": meta["group"], "N": meta["N"],
                "K": K, "d": meta["d"], "repeat": r})
    return row


def run_dataset(key: str, cfg: RunConfig, data_home: Optional[str] = None) -> pd.DataFrame:
    """跑完一个数据集的全部方法 × 重复（顺序执行）。"""
    X, y, meta = _load(key, cfg, data_home)
    K = int(meta["K"])
    _warmup(X, y, K, cfg)
    rows: List[Dict] = []
    for r in range(cfg.repeats):
        seed = cfg.seed_base + r            # 同一重复内各方法共用种子（配对比较）
        for method in cfg.methods:
            rows.append(_row_meta(run_single(X, y, K, method, cfg, seed, dataset_key=key),
                                  key, meta, K, r))
    return pd.DataFrame(rows)


def run_one_repeat(key: str, r: int, cfg: RunConfig,
                   data_home: Optional[str] = None) -> pd.DataFrame:
    """只跑一个 (数据集, 重复)：用于把重复级并行摊到多核上。"""
    X, y, meta = _load(key, cfg, data_home)
    K = int(meta["K"])
    seed = cfg.seed_base + r
    rows = [_row_meta(run_single(X, y, K, m, cfg, seed, dataset_key=key), key, meta, K, r)
            for m in cfg.methods]
    return pd.DataFrame(rows)


def merge_task_fragments(out_dir: str, keys: Optional[Sequence[str]] = None) -> None:
    """把重复级并行的分片（_tasks/<key>__r*.csv）合并成 <key>.csv。"""
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
        df = df.sort_values(["repeat", "method"]).reset_index(drop=True)
        df.to_csv(os.path.join(out_dir, key + ".csv"), index=False)
        print("[merge] %s <- %d 个分片（%d 行）" % (key, len(paths), len(df)), flush=True)


def _init_worker(cpu_pin_base, counter=None):
    """每个 worker 进程启动时把自己绑到一个独立 CPU 核（Linux）。

    counter 必须由父进程创建并通过 initargs 传入：如果在 worker 里创建
    multiprocessing.Value，每个 worker 拿到的是各自的副本，序号会全部是 0，
    所有进程就会挤在同一个核上（实测会让并行阶段慢 20 倍以上）。
    """
    if cpu_pin_base is None or not hasattr(os, "sched_setaffinity"):
        return
    idx = 0
    if counter is not None:
        try:
            with counter.get_lock():
                idx = int(counter.value)
                counter.value = idx + 1
        except Exception:
            idx = 0
    else:
        idx = int(os.getpid()) % 8      # 退化方案：至少不要全挤在一个核
    try:
        os.sched_setaffinity(0, {int(cpu_pin_base) + idx})
    except Exception:
        pass


def _task(fn_name, key, r, cfg, data_home, out_dir, cpu_pin_base, n_cores):
    if fn_name == "repeat":
        df = run_one_repeat(key, r, cfg, data_home)
        os.makedirs(os.path.join(out_dir, "_tasks"), exist_ok=True)
        df.to_csv(os.path.join(out_dir, "_tasks", "%s__r%d.csv" % (key, r)), index=False)
        return key, r, len(df)
    df = run_dataset(key, cfg, data_home)
    df.to_csv(os.path.join(out_dir, key + ".csv"), index=False)
    return key, None, len(df)


def run_all(keys: Sequence[str], cfg: RunConfig, out_dir: str, jobs: int = 1,
            resume: bool = True, data_home: Optional[str] = None,
            job_mode: str = "dataset", cpu_pin_base: Optional[int] = None,
            merge_only: bool = False) -> None:
    """批量运行。

    job_mode="dataset"：一个数据集一个任务（默认）。
    job_mode="repeat" ：一个 (数据集, 重复) 一个任务，适合把重活摊到多核上。
    cpu_pin_base       ：给并行 worker 依次绑定 CPU 核（Linux，用于稳定计时）。
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_rows = []
    for key in keys:
        spec = ds.DATASETS[key]
        meta_rows.append({"dataset": key, "group": spec["group"], "N": spec["n"],
                          "d": spec["d"], "K_paper": spec.get("K_paper", spec.get("K")),
                          "desc": spec.get("desc", spec.get("onml", "")),
                          "note": spec.get("note", "")})
    pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "_datasets.csv"), index=False)

    if merge_only:
        merge_task_fragments(out_dir, keys)
        return

    if job_mode == "repeat":
        todo = []
        for key in keys:
            for r in range(cfg.repeats):
                frag = os.path.join(out_dir, "_tasks", "%s__r%d.csv" % (key, r))
                if resume and os.path.exists(frag):
                    continue
                todo.append((key, r))
        todo.sort(key=lambda kr: -ds.DATASETS[kr[0]]["n"] * ds.DATASETS[kr[0]]["d"])
    else:
        todo = []
        for key in keys:
            if resume and os.path.exists(os.path.join(out_dir, key + ".csv")):
                print("[skip] %s 已存在（--overwrite 可强制重跑）" % key, flush=True)
                continue
            todo.append((key, None))
        todo.sort(key=lambda kr: -ds.DATASETS[kr[0]]["n"] * ds.DATASETS[kr[0]]["d"])

    if not todo:
        print("没有需要运行的任务", flush=True)
        merge_task_fragments(out_dir, keys)
        return
    print("任务数 %d（%s 模式，并行 %d，绑核起点 %s）"
          % (len(todo), job_mode, jobs, cpu_pin_base), flush=True)

    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        import multiprocessing as mp

        counter = mp.Value("i", 0) if cpu_pin_base is not None else None
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(cpu_pin_base, counter)) as pool:
            futs = {pool.submit(_task, job_mode, k, r, cfg, data_home, out_dir,
                                None, None): (k, r) for k, r in todo}
            done = 0
            for fut in as_completed(futs):
                key, r = futs[fut]
                done += 1
                try:
                    fut.result()
                    print("[done] %s%s（%d/%d）" % (key, "" if r is None else " r=%d" % r,
                                                    done, len(todo)), flush=True)
                except Exception:
                    print("[fail] %s r=%s\n%s" % (key, r, traceback.format_exc()), flush=True)
    else:
        for k, r in todo:
            try:
                _task(job_mode, k, r, cfg, data_home, out_dir, cpu_pin_base, None)
                print("[done] %s%s" % (k, "" if r is None else " r=%d" % r), flush=True)
            except Exception:
                print("[fail] %s r=%s\n%s" % (k, r, traceback.format_exc()), flush=True)

    merge_task_fragments(out_dir, keys)
