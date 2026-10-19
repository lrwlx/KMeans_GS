# -*- coding: utf-8 -*-
"""把 results 下的逐次运行记录整理成论文用的表格（xlsx）。

可以同时读多个目录（--raw a,b,c），按 (数据集, 方法, 重复) 去重：前面的目录优先，
因此可以把"主实验 results/raw"与"基线 results/base_raw,results/base_par_raw"合起来出表。

产出（默认写到 results/paper_tables/paper_tables.xlsx）：
  表1_数据集          数据集清单（N / K / d）
  表2_有标签质量       ARI / AMI / FMI，mean ± std，** 表示该行最优
  表3_无标签质量       MSE / SS / DB
  表4_时间            采样时间 / 迭代时间 / 总时间 / 迭代次数，含播种加速比
  表5_显著性检验      KMeans-GS 与其它方法的配对 t 检验（同一重复共用随机种子）
  表6_KMeans-GS诊断    d0、实测试探次数、兜底次数、定理 2 下界比值
  表7_ARI对比          全方法 ARI 宽表（含跨数据集平均与胜出次数）
  表8_时间对比         全方法 总时间 宽表（含几何平均）
  表9_汇总排名         每个方法的平均 ARI / 平均排名 / 几何平均时间 / 相对 KMeans-GS 倍数
  表10_失败记录        未完成或抛错的 (数据集, 方法) 及原因
  long_format          每个 (数据集, 方法, 重复) 一行

用法：
    python scripts/make_paper_tables.py --raw results/raw,results/base_raw,results/base_par_raw \
        --out results/paper_tables
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import runner as R  # noqa: E402

METHOD_ORDER = ["kmeans", "kmeanspp", "kmeans_gs", "kmeans_gs_g",
                "kmeans_par", "afkmc2", "mbkmeans", "dbscan", "birch", "gmm"]
SHORT = {"kmeans": "K-means", "kmeanspp": "K-means++", "kmeans_gs": "KMeans-GS",
         "kmeans_par": "K-means||", "afkmc2": "AFK-MC²",
         "kmeans_gs_g": "KMeans-GS-G",
         "mbkmeans": "MiniBatchKMeans",
         "dbscan": "DBSCAN",
         "birch": "BIRCH", "gmm": "GMM"}

LABELED = ["dermatology", "wine_quality", "wdbc", "synthetic_control", "letter",
           "musk", "isolet", "segment", "yeast", "vehicle", "pendigits",
           "satimage", "synth_imbalanced"]
UNLABELED = LABELED

QUAL_SUP = ["ari", "ami", "fmi"]
QUAL_UNSUP = ["mse", "silhouette", "davies_bouldin"]
TIME_METRICS = ["time_seed", "time_iter", "time_total", "n_iter"]
TIME_NAMES = {"time_seed": "采样时间", "time_iter": "迭代时间",
              "time_total": "总时间", "n_iter": "迭代次数"}
LOWER_BETTER = {"mse", "davies_bouldin"}


def kind_of(method):
    spec = R.METHOD_SPECS.get(method)
    return spec["kind"] if spec else "seed"


def fmt(mean, std, nd=3):
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        return "%.*f" % (nd, mean)
    return "%.*f ± %.*f" % (nd, mean, nd, std)


def geo(vals):
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(vals)))) if vals else float("nan")


def is_result_file(path):
    """只接受逐次运行记录（必须同时有 method 与 status 两列），跳过 eps 表之类的辅助 CSV。"""
    try:
        cols = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return ("method" in cols) and ("status" in cols)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="results/raw",
                   help="逗号分隔的多个目录；靠前的目录在去重时优先")
    p.add_argument("--out", default="results/paper_tables")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args(argv)

    raw_dirs = [x.strip() for x in args.raw.split(",") if x.strip()]
    files = []
    for rd in raw_dirs:
        if not os.path.isdir(rd):
            print("[warn] 目录不存在：%s" % rd)
            continue
        files += [f for f in sorted(glob.glob(os.path.join(rd, "*.csv")))
                  if not os.path.basename(f).startswith("_") and is_result_file(f)]
    if not files:
        print("没有找到任何原始结果")
        return 1

    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_before = len(raw)
    raw = raw.drop_duplicates(subset=["dataset", "method", "repeat"], keep="first")
    if len(raw) != n_before:
        print("去重：%d -> %d 行（同一 (数据集, 方法, 重复) 保留靠前目录的记录）"
              % (n_before, len(raw)))

    failed = raw[raw["status"] != "ok"][["dataset", "method", "status", "error"]].copy()
    failed = failed.drop_duplicates(subset=["dataset", "method"])
    df = raw[raw["status"] == "ok"].copy()
    os.makedirs(args.out, exist_ok=True)

    present = set(df["method"].unique())
    METHODS = [m for m in METHOD_ORDER if m in present] + \
              [m for m in sorted(present) if m not in METHOD_ORDER]
    if "kmeans_gs" in METHODS:                       # KMeans-GS 放前面便于阅读
        METHODS = ["kmeans_gs"] + [m for m in METHODS if m != "kmeans_gs"]
    LABELS = {m: SHORT.get(m, R.METHOD_SPECS.get(m, {}).get("label", m)) for m in METHODS}

    n_rep = df.groupby("dataset")["repeat"].nunique()
    print("数据：%d 条有效记录，%d 个数据集，每数据集 %d~%d 次重复，%d 个方法"
          % (len(df), df["dataset"].nunique(), n_rep.min(), n_rep.max(), len(METHODS)))
    for m in METHODS:
        k = kind_of(m)
        n_ds = df[df.method == m]["dataset"].nunique()
        print("  %-16s kind=%-5s 覆盖 %2d 个数据集" % (m, k, n_ds))
    if len(failed):
        print("失败/跳过 %d 个 (数据集, 方法)：\n%s" % (len(failed), failed.to_string(index=False)))

    metrics = [m for m in TIME_METRICS + QUAL_SUP + QUAL_UNSUP if m in df.columns]

    # (数据集, 方法) -> 指标 mean/std；重复级序列用于配对检验
    g = df.groupby(["dataset", "method"])[metrics].agg(["mean", "std"])
    series_cache = {}

    def ms(ds, method, met, what):
        try:
            return float(g.loc[(ds, method), (met, what)])
        except KeyError:
            return float("nan")

    def series(ds, method, met):
        key = (ds, method, met)
        if key not in series_cache:
            sub = df[(df.dataset == ds) & (df.method == method)].sort_values("repeat")
            series_cache[key] = sub[met].to_numpy(float)
        return series_cache[key]

    def paired_p(ds, met, a, b):
        xa, xb = series(ds, a, met), series(ds, b, met)
        if len(xa) != len(xb) or len(xa) < 3:
            return float("nan")
        if np.allclose(xa, xb):
            return 1.0
        try:
            return float(stats.ttest_rel(xa, xb).pvalue)
        except Exception:
            return float("nan")

    datasets = sorted(df["dataset"].unique(),
                      key=lambda k: -float(df[df.dataset == k]["N"].iloc[0]))
    Ks = df.groupby("dataset")["K"].first().to_dict()
    Ns = df.groupby("dataset")["N"].first().to_dict()
    dims = df.groupby("dataset")["d"].first().to_dict()

    # ---------------- 表 1 ----------------
    t1 = pd.DataFrame([{"数据集": k, "样本数 N": int(Ns[k]), "类别数 K": int(Ks[k]),
                        "特征维数 d": int(dims[k])} for k in datasets])

    # ---------------- 表 2 / 表 3 ----------------
    def quality_table(keys, metas):
        rows = []
        for dset in [k for k in keys if k in datasets]:
            row = {"数据集": dset, "K": int(Ks[dset])}
            for m in METHODS:
                cells = []
                for met in metas:
                    mu, sd = ms(dset, m, met, "mean"), ms(dset, m, met, "std")
                    vals = [v for v in (ms(dset, x, met, "mean") for x in METHODS) if np.isfinite(v)]
                    best = (min(vals) if met in LOWER_BETTER else max(vals)) if vals else np.nan
                    isbest = np.isfinite(mu) and np.isfinite(best) and abs(mu - best) < 1e-12
                    txt = fmt(mu, sd)
                    cells.append(("**%s**" % txt) if (txt and isbest) else txt)
                row[LABELS[m]] = " / ".join(cells)
            rows.append(row)
        avg = {"数据集": "**跨数据集平均**", "K": ""}
        for m in METHODS:
            parts = []
            for met in metas:
                vals = [ms(k, m, met, "mean") for k in keys if k in datasets]
                vals = [v for v in vals if np.isfinite(v)]
                parts.append("%.3f" % float(np.mean(vals)) if vals else "")
            avg[LABELS[m]] = " / ".join(parts)
        return pd.concat([pd.DataFrame(rows), pd.DataFrame([avg])], ignore_index=True)

    t2 = quality_table(LABELED, QUAL_SUP)
    t3 = quality_table(UNLABELED, QUAL_UNSUP)

    # ---------------- 表 4 ----------------
    rows = []
    for dset in datasets:
        first = True
        for m in METHODS:
            r = {"数据集": dset if first else "", "方法": LABELS[m], "类型": kind_of(m)}
            first = False
            for met in TIME_METRICS:
                nd = 6 if met.startswith("time") else 2
                blank = (kind_of(m) == "full" and met in ("time_seed", "time_iter"))
                r[TIME_NAMES[met]] = "" if blank else fmt(ms(dset, m, met, "mean"),
                                                          ms(dset, m, met, "std"), nd)
            for a, b, name in (("kmeanspp", "kmeans_gs", "播种加速比 K-means++/KMeans-GS"),
                               ("kmeanspp", "kmeans_par", "播种加速比 K-means++/K-means||"),
                               ("kmeans_par", "kmeans_gs", "播种加速比 K-means||/KMeans-GS")):
                if a in METHODS and b in METHODS:
                    ta, tb = ms(dset, a, "time_seed", "mean"), ms(dset, b, "time_seed", "mean")
                    r[name] = "%.2f×" % (ta / tb) if tb > 0 else ""
            rows.append(r)
    geo_row = {"数据集": "**几何平均**", "方法": "", "类型": ""}
    for met in TIME_METRICS:
        cells = []
        for m in METHODS:
            vals = [ms(k, m, met, "mean") for k in datasets]
            v = geo(vals)
            cells.append("%.4g" % v if np.isfinite(v) else "")
        geo_row[TIME_NAMES[met]] = " | ".join("%s=%s" % (LABELS[m], c)
                                              for m, c in zip(METHODS, cells) if c)
    if "kmeans_gs" in METHODS and "kmeanspp" in METHODS:
        geo_row["播种加速比 K-means++/KMeans-GS"] = "%.2f×" % geo(
            [ms(k, "kmeanspp", "time_seed", "mean") / ms(k, "kmeans_gs", "time_seed", "mean")
             for k in datasets])
    if "kmeans_gs" in METHODS and "kmeans_par" in METHODS:
        geo_row["播种加速比 K-means||/KMeans-GS"] = "%.2f×" % geo(
            [ms(k, "kmeans_par", "time_seed", "mean") / ms(k, "kmeans_gs", "time_seed", "mean")
             for k in datasets])
    t4 = pd.concat([pd.DataFrame(rows), pd.DataFrame([geo_row])], ignore_index=True)

    # ---------------- 表 5：KMeans-GS 与其它方法的配对 t 检验 ----------------
    sig = []
    for keys, metas in ((LABELED, QUAL_SUP), (UNLABELED, QUAL_UNSUP)):
        for dset in [k for k in keys if k in datasets]:
            for met in metas:
                groups = [series(dset, m, met) for m in METHODS if len(series(dset, m, met)) >= 3]
                if len(groups) < 2:
                    continue
                try:
                    F, p_anova = stats.f_oneway(*groups)
                except Exception:
                    F, p_anova = np.nan, np.nan
                row = {"数据集": dset, "指标": met, "F (ANOVA)": float(F), "p (ANOVA)": float(p_anova)}
                d = -1.0 if met in LOWER_BETTER else 1.0
                for m in METHODS:
                    if m == "kmeans_gs":
                        continue
                    mu_kmeans_gs = ms(dset, "kmeans_gs", met, "mean")
                    mu_m = ms(dset, m, met, "mean")
                    if not (np.isfinite(mu_kmeans_gs) and np.isfinite(mu_m)):
                        continue
                    if mu_kmeans_gs == mu_m:
                        row["方向 %s" % LABELS[m]] = "相同"
                    else:
                        row["方向 %s" % LABELS[m]] = "KMeans-GS 更好" if d * (mu_kmeans_gs - mu_m) > 0 else "KMeans-GS 更差"
                    row["p (KMeans-GS vs %s)" % LABELS[m]] = paired_p(dset, met, "kmeans_gs", m)
                sig.append(row)
    t5 = pd.DataFrame(sig)
    if t5.empty:
        t5 = pd.DataFrame(columns=["数据集", "指标"])
    else:
        t5 = t5.sort_values(["数据集", "指标"]).reset_index(drop=True)

    # ---------------- 表 6：KMeans-GS 诊断 ----------------
    diag = [c for c in ["kmeans_gs_d0", "kmeans_gs_iqr_n", "kmeans_gs_trials_mean",
                        "kmeans_gs_trials_max", "kmeans_gs_fallbacks", "seed_min_dist",
                        "seed_mean_dist", "kmeans_gs_bound", "kmeans_gs_bound_ratio"]
            if c in df.columns]
    t6 = (df[df.method == "kmeans_gs"].groupby("dataset")[diag].mean()
          .reindex(datasets).reset_index().rename(columns={"dataset": "数据集"}))

    # ---------------- 表 7：ARI 宽表 + 胜出次数 ----------------
    def wide(met, nd=4, best=False, lower=False, agg="mean", avg_label="跨数据集平均"):
        rows = []
        wins = {m: 0 for m in METHODS}
        for dset in datasets:
            row = {"数据集": dset, "K": int(Ks[dset])}
            vals = {m: ms(dset, m, met, "mean") for m in METHODS}
            fin = [v for v in vals.values() if np.isfinite(v)]
            bestv = (min(fin) if lower else max(fin)) if fin else np.nan
            for m in METHODS:
                mu, sd = vals[m], ms(dset, m, met, "std")
                txt = fmt(mu, sd, nd)
                isbest = best and np.isfinite(mu) and np.isfinite(bestv) and abs(mu - bestv) < 1e-12
                if isbest:
                    wins[m] += 1
                    txt = "**%s**" % txt
                row[LABELS[m]] = txt
            rows.append(row)
        avg = {"数据集": "**%s**" % avg_label, "K": ""}
        means = {}
        for m in METHODS:
            vals = [ms(k, m, met, "mean") for k in datasets]
            vals = [v for v in vals if np.isfinite(v)]
            means[m] = (geo(vals) if agg == "geo" else float(np.mean(vals))) if vals else np.nan
        fin = [v for v in means.values() if np.isfinite(v)]
        target = (min(fin) if lower else max(fin)) if fin else np.nan
        for m in METHODS:
            v = means[m]
            if not np.isfinite(v):
                avg[LABELS[m]] = ""
                continue
            txt = "%.*f" % (nd, v)
            avg[LABELS[m]] = ("**%s**" % txt) if (best and np.isfinite(target)
                                                  and abs(v - target) < 1e-12) else txt
        return pd.concat([pd.DataFrame(rows), pd.DataFrame([avg])], ignore_index=True), wins

    t7, wins = wide("ari", 4, best=True, avg_label="跨数据集平均（胜出数据集数见 表9）")
    t8, _ = wide("time_total", 3, best=True, lower=True, agg="geo", avg_label="几何平均")

    # ---------------- 表 9：汇总排名 ----------------
    rows = []
    ranks = {}
    for dset in datasets:
        vals = [(m, ms(dset, m, "ari", "mean")) for m in METHODS]
        vals = [(m, v) for m, v in vals if np.isfinite(v)]
        for i, (m, _) in enumerate(sorted(vals, key=lambda t: -t[1]), start=1):
            ranks.setdefault(m, []).append(i)
    kmeans_gs_ari = np.mean([ms(k, "kmeans_gs", "ari", "mean") for k in datasets
                             if np.isfinite(ms(k, "kmeans_gs", "ari", "mean"))]) if "kmeans_gs" in METHODS else np.nan
    kmeans_gs_tt = geo([ms(k, "kmeans_gs", "time_total", "mean") for k in datasets]) if "kmeans_gs" in METHODS else np.nan
    kmeans_gs_seed = geo([ms(k, "kmeans_gs", "time_seed", "mean") for k in datasets]) if "kmeans_gs" in METHODS else np.nan
    for m in METHODS:
        aris = [ms(k, m, "ari", "mean") for k in datasets]
        aris = [v for v in aris if np.isfinite(v)]
        tt = geo([ms(k, m, "time_total", "mean") for k in datasets])
        sd = geo([ms(k, m, "time_seed", "mean") for k in datasets])
        rows.append({
            "方法": LABELS[m], "类型": kind_of(m), "覆盖数据集数": len(aris),
            "平均 ARI": float(np.mean(aris)) if aris else np.nan,
            "ARI 排名均值": float(np.mean(ranks.get(m, [np.nan]))) if ranks.get(m) else np.nan,
            "ARI 胜出数据集数": wins.get(m, 0),
            "几何平均总时间(s)": tt,
            "相对 KMeans-GS 总时间": (tt / kmeans_gs_tt) if (np.isfinite(tt) and np.isfinite(kmeans_gs_tt)) else np.nan,
            "几何平均采样时间(s)": sd,
            "相对 KMeans-GS 采样时间": (sd / kmeans_gs_seed) if (np.isfinite(sd) and np.isfinite(kmeans_gs_seed)) else np.nan,
        })
    t9 = pd.DataFrame(rows).sort_values("平均 ARI", ascending=False).reset_index(drop=True)

    long = df.groupby(["dataset", "method", "repeat"])[metrics].mean().reset_index()

    path = os.path.join(args.out, "paper_tables.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        t1.to_excel(xw, sheet_name="表1_数据集", index=False)
        t2.to_excel(xw, sheet_name="表2_有标签质量", index=False)
        t3.to_excel(xw, sheet_name="表3_无标签质量", index=False)
        t4.to_excel(xw, sheet_name="表4_时间", index=False)
        t5.to_excel(xw, sheet_name="表5_显著性检验", index=False)
        t6.to_excel(xw, sheet_name="表6_KMeans-GS诊断", index=False)
        t7.to_excel(xw, sheet_name="表7_ARI对比", index=False)
        t8.to_excel(xw, sheet_name="表8_时间对比", index=False)
        t9.to_excel(xw, sheet_name="表9_汇总排名", index=False)
        failed.to_excel(xw, sheet_name="表10_失败记录", index=False)
        long.to_excel(xw, sheet_name="long_format", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n== 表 7：ARI（mean ± std，** = 该数据集最优）==")
    print(t7.to_string(index=False))
    print("\n== 表 9：汇总排名 ==")
    print(t9.to_string(index=False, float_format=lambda v: "%.4g" % v))
    print("\n== 表 8：总时间（秒）==")
    print(t8.to_string(index=False))
    if len(t6):
        print("\n== 表 6：KMeans-GS 诊断 ==")
        print(t6.to_string(index=False, float_format=lambda v: "%.4g" % v))
    print("\n输出：%s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
