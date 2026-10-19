#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验入口：播种型方法 + 完整算法基线的时间与质量指标。

两类方法
--------
kind="seed"：播种 + 共享 Lloyd 迭代（本文的 kmeans / kmeanspp / kmeans_gs，
             以及对照 kmeans_par（K-means||）与 afkmc2（AFK-MC²）），
             时间拆成 采样时间（time_seed）与迭代时间（time_iter）。
kind="full"：自带完整流程的算法（mbkmeans / dbscan / birch / gmm），
             只记 总时间（time_total）。

用法示例
--------
python scripts/run_experiment.py --list                       # 列出数据集与方法
python scripts/run_experiment.py --prepare                    # 先把数据下载到 cache/
python scripts/run_experiment.py --datasets iris --repeats 2 --out results/smoke
python scripts/run_experiment.py --group uci --repeats 30 --jobs 1 --out results/raw

# 基线第一遍：与本文方法同构的播种 / K-means 族（时间表用，必须串行）
python scripts/run_experiment.py --group uci --repeats 30 --jobs 1 --cpu-pin-base 40 \
    --methods kmeans,kmeanspp,kmeans_gs,kmeans_gs_g,kmeans_gs_w,kmeans_par,afkmc2,mbkmeans \
    --out results/base_raw

# 基线第二遍：结构型算法（DBSCAN / BIRCH / GMM），按重复并行
python scripts/run_experiment.py --group uci --repeats 30 --jobs 32 --job-mode repeat \
    --cpu-pin-base 8 --methods dbscan,birch,gmm --out results/base_par_raw
"""

from __future__ import annotations

# BLAS 线程必须在导入 numpy 之前锁定为 1，否则计时会被线程抖动污染
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import datasets as ds  # noqa: E402
from src import runner  # noqa: E402

DEFAULT_METHODS = ("kmeans,kmeanspp,kmeans_gs,kmeans_gs_g,"
                   "kmeans_par,afkmc2,mbkmeans,dbscan,birch,gmm")


def build_parser():
    p = argparse.ArgumentParser(description="KMeans-GS 对比实验")
    p.add_argument("--datasets", default="", help="逗号分隔的数据集 key，如 iris,ecoli")
    p.add_argument("--group", default="uci", choices=["all", "uci", "synth"],
                   help="按分组选择数据集（--datasets 优先）")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--out", default="results/raw")
    p.add_argument("--jobs", type=int, default=1, help="并行进程数")
    p.add_argument("--job-mode", default="dataset", choices=["dataset", "repeat"],
                   help="dataset=一个数据集一个任务；repeat=一个(数据集,重复)一个任务")
    p.add_argument("--cpu-pin-base", type=int, default=None,
                   help="并行 worker 依次绑定的起始 CPU 核号（Linux，稳定计时用）")
    p.add_argument("--merge-only", action="store_true",
                   help="只把 _tasks/ 的分片合并成数据集级 CSV，不跑实验")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--overwrite", dest="resume", action="store_false")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument("--no-standardize", dest="standardize", action="store_false", default=True)
    p.add_argument("--sil-sample", type=int, default=10000)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--data-home", default=None)
    p.add_argument("--max-seconds", type=float, default=3600.0,
                   help="单个(数据集,方法,重复)的硬超时（秒）；0 表示不设限")
    # 对比实现的选择
    p.add_argument("--kmeanspp-impl", default="optimized", choices=["optimized", "classic"],
                   help="optimized=O(KNd) 增量实现；classic=O(K^2Nd) 逐轮全量重算")
    # KMeans-GS 算法的可切换选项
    p.add_argument("--gs-d0-mode", default="paper", choices=["paper", "kdep"])
    p.add_argument("--gs-decay", default="inv_sqrt", choices=["inv_sqrt", "log", "power"],
                   help="power 时用 tau_i = d0 / i^--gs-decay-p")
    p.add_argument("--gs-decay-p", type=float, default=0.5)
    p.add_argument("--gs-d0-coef", type=float, default=0.5,
                   help="d0 = coef * ||IQR||_2（正文取 0.5）")
    p.add_argument("--gs-first", default="random",
                   choices=["random", "median", "median_sub", "lds"],
                   help="median 用全量数据（O(Nd)）；median_sub/lds 只在子样本里找（与 N 无关）")
    p.add_argument("--gs-first-sample", type=int, default=2000)
    p.add_argument("--gs-max-trials", type=int, default=100)
    p.add_argument("--gs-iqr-sample", type=int, default=2000,
                   help="估计四分位距所用的样本数上限；0 表示使用全量数据")
    p.add_argument("--gs-edge", dest="gs_edge", action="store_true", default=False)
    p.add_argument("--gs-edge-rel", type=float, default=0.5)
    # K-means|| 与 AFK-MC²
    p.add_argument("--kmeans-par-oversample", type=int, default=0,
                   help="k-means|| 的过采样因子 l；0 表示按文献默认 2K")
    p.add_argument("--kmeans-par-rounds", type=int, default=5,
                   help="k-means|| 的采样轮数（Spark 默认 5）")
    p.add_argument("--afkmc2-chain", type=int, default=200,
                   help="AFK-MC² 估计正常数的 MCMC 链长 m")
    p.add_argument("--afkmc2-max-trials", type=int, default=100)
    p.add_argument("--afkmc2-slack", type=float, default=2.0,
                   help="AFK-MC² 接受概率分母的松弛系数（原文取 2）")
    # 完整算法基线的超参数
    p.add_argument("--dbscan-min-samples", type=int, default=5)
    p.add_argument("--dbscan-max-points", type=int, default=20000,
                   help="超过该样本数的数据集跳过 DBSCAN（蛮力近邻不可行），记为 error")
    p.add_argument("--dbscan-eps-file", default="results/dbscan_eps.csv",
                   help="由 scripts/choose_dbscan_eps.py 生成的 eps 表；不存在则退化为 k-距离拐点")
    p.add_argument("--birch-threshold", type=float, default=None)
    p.add_argument("--birch-branching-factor", type=int, default=50)
    p.add_argument("--gmm-covariance", default=None,
                   choices=[None, "full", "tied", "diag", "spherical"])
    p.add_argument("--gmm-max-iter", type=int, default=100)
    p.add_argument("--gmm-tol", type=float, default=1e-3)
    p.add_argument("--mbk-restarts", type=int, default=3)
    p.add_argument("--mbk-batch-size", type=int, default=0, help="0 表示 max(1024, 20K)")
    p.add_argument("--mbk-max-iter", type=int, default=100)
    p.add_argument("--list", action="store_true", help="列出全部数据集与方法后退出")
    p.add_argument("--prepare", action="store_true", help="下载并缓存全部数据集后退出")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list:
        print("== 数据集 ==")
        for k in ds.keys("all"):
            s = ds.DATASETS[k]
            print("%-20s %-6s N=%-6d d=%-4d K=%-3s %s" % (
                k, s["group"], s["n"], s["d"], s.get("K"), s.get("note", "")))
        print("\n== 方法（%d 个）==" % len(runner.METHOD_SPECS))
        for k, v in sorted(runner.METHOD_SPECS.items()):
            print("%-18s kind=%-5s %s" % (k, v["kind"], v["label"]))
        return 0

    if args.prepare:
        for k in ds.keys("all"):
            X, y, meta = ds.load(k, data_home=args.data_home)
            print("[ok] %-20s N=%d d=%d K=%d" % (k, meta["N"], meta["d"], meta["K"]), flush=True)
        return 0

    keys = ([k.strip() for k in args.datasets.split(",") if k.strip()]
            if args.datasets else ds.keys(args.group))
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = [m for m in methods if m not in runner.METHOD_SPECS]
    if unknown:
        print("未知方法：%s\n可用方法：%s"
              % (", ".join(unknown), ", ".join(sorted(runner.METHOD_SPECS))), file=sys.stderr)
        return 2

    dbscan_eps = {}
    if "dbscan" in methods:
        if args.dbscan_eps_file and os.path.exists(args.dbscan_eps_file):
            eps_df = pd.read_csv(args.dbscan_eps_file)
            dcol = "dataset" if "dataset" in eps_df.columns else eps_df.columns[0]
            if "eps" in eps_df.columns:
                dbscan_eps["dbscan"] = {str(k): float(v)
                                        for k, v in zip(eps_df[dcol], eps_df["eps"])}
            print("DBSCAN eps 取自 %s：%s"
                  % (args.dbscan_eps_file,
                     "、".join("%s(%d 个数据集)" % (m, len(v)) for m, v in dbscan_eps.items())))
        else:
            print("[warn] 未找到 %s，DBSCAN 退化为 k-距离拐点启发式（建议先跑 "
                  "scripts/choose_dbscan_eps.py）" % args.dbscan_eps_file)

    cfg = runner.RunConfig(
        repeats=args.repeats,
        methods=methods,
        max_iter=args.max_iter,
        tol=args.tol,
        standardize=args.standardize,
        sil_sample=args.sil_sample,
        seed_base=args.seed_base,
        kmeanspp_impl=args.kmeanspp_impl,
        max_seconds=(args.max_seconds if args.max_seconds and args.max_seconds > 0 else None),
        kmeans_gs_kwargs=dict(d0_mode=args.gs_d0_mode, decay=args.gs_decay, first=args.gs_first,
                              max_trials=args.gs_max_trials, edge_check=args.gs_edge,
                              edge_rel=args.gs_edge_rel, iqr_sample=args.gs_iqr_sample,
                              d0_coef=args.gs_d0_coef, decay_p=args.gs_decay_p,
                              first_sample=args.gs_first_sample),
        extra_kwargs={
            "kmeans_par": dict(oversample=(args.kmeans_par_oversample or None),
                               rounds=args.kmeans_par_rounds),
            "afkmc2": dict(chain=args.afkmc2_chain, max_trials=args.afkmc2_max_trials,
                           accept_slack=args.afkmc2_slack),
        },
        full_kwargs={
            "dbscan": dict(min_samples=args.dbscan_min_samples,
                           max_points=args.dbscan_max_points),
            "birch": dict(threshold=args.birch_threshold,
                          branching_factor=args.birch_branching_factor),
            "gmm": dict(covariance=args.gmm_covariance, max_iter=args.gmm_max_iter,
                        tol=args.gmm_tol),
            "mbkmeans": dict(restarts=args.mbk_restarts,
                             batch_size=(args.mbk_batch_size or None),
                             max_iter=args.mbk_max_iter),
        },
        dbscan_eps=dbscan_eps,
    )
    print("数据集: %s" % ", ".join(keys))
    print("方法  : %s" % ", ".join("%s(%s)" % (m, runner.METHOD_SPECS[m]["kind"]) for m in methods))
    print("重复  : %d   并行: %d   模式: %s   输出: %s"
          % (args.repeats, args.jobs, args.job_mode, args.out))
    print("KMeans-GS 配置: d0=%s decay=%s first=%s max_trials=%d edge=%s iqr_sample=%d"
          % (args.gs_d0_mode, args.gs_decay, args.gs_first, args.gs_max_trials,
             args.gs_edge, args.gs_iqr_sample))
    print("K-means|| 配置: oversample=%s rounds=%d" %
          (args.kmeans_par_oversample or "2K（自动）", args.kmeans_par_rounds))
    print("AFK-MC² 配置: chain=%d max_trials=%d slack=%g"
          % (args.afkmc2_chain, args.afkmc2_max_trials, args.afkmc2_slack))
    print("基线配置: dbscan(min_samples=%d, max_points=%d) "
          "birch(threshold=%s, bf=%d) gmm(cov=%s, max_iter=%d)  单次超时=%s 秒"
          % (args.dbscan_min_samples, args.dbscan_max_points,
             args.birch_threshold if args.birch_threshold else "自动", args.birch_branching_factor,
             args.gmm_covariance or "自动(d>50 用 diag)", args.gmm_max_iter,
             args.max_seconds if args.max_seconds else "不限"))
    runner.run_all(keys, cfg, out_dir=args.out, jobs=args.jobs, resume=args.resume,
                   data_home=args.data_home, job_mode=args.job_mode,
                   cpu_pin_base=args.cpu_pin_base, merge_only=args.merge_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
