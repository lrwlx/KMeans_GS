#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate the v2 benchmark.

The independent unit is the *dataset*: the 20 repetitions within a dataset are
not independent, so every inferential statement uses dataset-level means.  The
script writes the tables used by the manuscript:

    per_dataset.csv   dataset x method means / SDs
    summary.csv       per-method descriptive summary and mean rank
    pairwise.csv      KMeans-GS vs every other method, dataset-level tests
    friedman.csv      Friedman test over the main methods
    nemenyi.csv       mean ranks and the Nemenyi critical difference
    loo.csv           leave-one-dataset-out sensitivity
    subgroups.csv     convex vs non-convex, K bands, N bands
    best_of.csv       best-of-R ARI per method
    times.csv         per-dataset geometric-mean seeding and total times
    report.json       machine-readable summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, rankdata, wilcoxon

MAIN = ["kmeans", "kmeanspp", "kmeanspp_greedy", "kmeanspp_sklearn",
        "kmeans_par", "afkmc2", "kmeans_gs", "kmeans_gs_g",
        "mbkmeans", "birch", "gmm", "dbscan"]
REF = ["sklearn1", "sklearn10"]
LABEL = {
    "kmeans": "K-means (random)",
    "kmeanspp": "k-means++",
    "kmeanspp_greedy": "k-means++ (greedy)",
    "kmeanspp_sklearn": "k-means++ (compiled)",
    "kmeans_par": "k-means∥",
    "afkmc2": "AFK-MC2",
    "kmeans_gs": "KMeans-GS",
    "kmeans_gs_g": "KMeans-GS-G",
    "mbkmeans": "MiniBatchKMeans",
    "birch": "BIRCH",
    "gmm": "Gaussian mixture",
    "dbscan": "DBSCAN",
    "sklearn1": "scikit-learn KMeans (1 restart)",
    "sklearn10": "scikit-learn KMeans (10 restarts)",
}
METRICS = ["ari", "ami", "fmi", "mse", "silhouette", "davies_bouldin",
           "n_iter", "time_seed", "time_total"]

# Demšar's q_{0.05} for the Nemenyi test, k = 2..20
Q05 = [None, None, 1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102,
       3.164, 3.219, 3.268, 3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517,
       3.544]


def read_runs(path):
    frames = []
    for f in sorted(glob.glob(os.path.join(path, "*.csv"))):
        if os.path.basename(f).startswith("_"):
            continue
        df = pd.read_csv(f)
        if "method" in df and "status" in df:
            frames.append(df)
    if not frames:
        raise RuntimeError("no run files in %s" % path)
    df = pd.concat(frames, ignore_index=True)
    for c in METRICS + ["N", "K", "d", "fallbacks", "trials_mean", "seed_min_dist",
                        "seed_mean_dist", "bound", "bound_ratio", "scale_d0"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def geomean(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.log(v).mean())) if v.size else float("nan")


def boot_ci(v, rng, reps=20000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, v.size, size=(reps, v.size))
    m = v[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def safe_wilcoxon(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size < 5 or np.allclose(d, 0):
        return float("nan"), 1.0
    try:
        return float(wilcoxon(d, zero_method="wilcox").statistic), \
            float(wilcoxon(d, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan"), 1.0


def rank_biserial(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    if not d.size:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / (r[d > 0].sum() + r[d < 0].sum()))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def nemenyi_cd(k, n, alpha=0.05):
    if k >= len(Q05):
        raise ValueError("k too large for the critical-value table")
    return Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", default=",".join(MAIN))
    p.add_argument("--reference", default=",".join(REF))
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(20261113)

    raw = read_runs(args.raw)
    ok = raw[raw.status == "ok"].copy()
    methods = [m for m in args.methods.split(",") if m in set(ok.method)]
    datasets = sorted(ok.dataset.unique())
    ok = ok[ok.method.isin(methods)]

    # completeness report
    counts = ok.groupby(["dataset", "method"]).size().unstack()
    coverage = counts.notna()

    per = {}
    for metric in METRICS:
        g = ok.groupby(["dataset", "method"])[metric]
        per[metric] = g.mean().unstack().reindex(index=datasets, columns=methods)

    ari = per["ari"]
    ari_ranks = ari.rank(axis=1, ascending=False, method="average")

    # wide matrices used by the figures
    ari.to_csv(os.path.join(args.out, "matrix_ari_mean.csv"))
    for metric, fname in [("time_seed", "matrix_time_seed_geomean.csv"),
                          ("time_iter", "matrix_time_iter_geomean.csv"),
                          ("time_total", "matrix_time_total_geomean.csv"),
                          ("n_iter", "matrix_n_iter_mean.csv")]:
        mat = ok.pivot_table(index="dataset", columns="method", values=metric,
                             aggfunc=lambda x: np.exp(np.log(x[x > 0]).mean())
                             if metric.startswith("time") else np.mean(x))
        mat.to_csv(os.path.join(args.out, fname))

    # per-dataset table
    rows = []
    for d in datasets:
        for m in methods:
            sub = ok[(ok.dataset == d) & (ok.method == m)]
            if not len(sub):
                continue
            r = {"dataset": d, "method": m, "label": LABEL[m],
                 "N": int(sub.N.iloc[0]), "K": int(sub.K.iloc[0]),
                 "d": int(sub.d.iloc[0]), "runs": len(sub),
                 "ari_rank": float(ari_ranks.loc[d, m])}
            for metric in METRICS:
                v = sub[metric].to_numpy(float)
                v = v[np.isfinite(v)]
                r[metric] = float(v.mean()) if v.size else np.nan
                r[metric + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else np.nan
            rows.append(r)
    per_ds = pd.DataFrame(rows)
    per_ds.to_csv(os.path.join(args.out, "per_dataset.csv"), index=False)

    # summary
    ds_n = ok.groupby("dataset").N.first()
    total_n = float(ds_n.sum())
    summ = []
    for m in methods:
        a = ari[m].to_numpy(float)
        lo, hi = boot_ci(a, rng)
        summ.append({
            "method": m, "label": LABEL[m],
            "mean_ari": float(np.nanmean(a)),
            "mean_ari_lo": lo, "mean_ari_hi": hi,
            "median_ari": float(np.nanmedian(a)),
            "trimmed_ari": float(np.nanmean(np.sort(a)[1:-1])),
            "mean_rank": float(ari_ranks[m].mean()),
            "rank_sd": float(ari_ranks[m].std(ddof=1)),
            "top1": int((ari_ranks[m] == 1).sum()),
            "top3": int((ari_ranks[m] <= 3).sum()),
            "n_weighted_ari": float(np.nansum(ds_n[ari.index].values * a) / total_n),
            "time_seed_geo": geomean(per["time_seed"][m]) if m != "kmeans" else geomean(per["time_seed"][m]),
            "time_total_geo": geomean(per["time_total"][m]),
            "n_iter_mean": float(np.nanmean(per["n_iter"][m])),
            "runs": int(coverage[m].sum()),
        })
    summary = pd.DataFrame(summ)
    base_seed = summary.loc[summary.method == "kmeans_gs", "time_seed_geo"].iloc[0]
    summary["seed_speedup_vs"] = base_seed / summary["time_seed_geo"]
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    # pairwise: KMeans-GS vs each other method, dataset-level
    ref = "kmeans_gs"
    pw = []
    pvals, names = [], []
    for m in methods:
        if m == ref:
            continue
        diff = (ari[ref] - ari[m]).dropna()
        lo, hi = boot_ci(diff.to_numpy(), rng)
        w, p_w = safe_wilcoxon(ari[ref], ari[m])
        wins = int((diff > 1e-9).sum())
        losses = int((diff < -1e-9).sum())
        ties = int(len(diff) - wins - losses)
        p_sign = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
        pw.append({"comparison": "KMeans-GS minus %s" % LABEL[m],
                   "other": m, "mean_diff": float(diff.mean()),
                   "ci_lo": lo, "ci_hi": hi,
                   "wins": wins, "ties": ties, "losses": losses,
                   "sign_p": float(p_sign), "wilcoxon_p": float(p_w),
                   "rank_biserial": rank_biserial(ari[ref], ari[m])})
        pvals.append(p_w)
        names.append(m)
    pairwise = pd.DataFrame(pw)
    pairwise["holm_p"] = holm(pvals)
    pairwise.to_csv(os.path.join(args.out, "pairwise.csv"), index=False)

    # Friedman + Nemenyi
    mat = ari[methods].to_numpy(float)
    fr = friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])
    cd = nemenyi_cd(len(methods), len(datasets))
    fried = pd.DataFrame([{"n_datasets": len(datasets), "n_methods": len(methods),
                           "statistic": float(fr.statistic), "p_value": float(fr.pvalue),
                           "kendall_w": float(fr.statistic / (len(datasets) * (len(methods) - 1))),
                           "nemenyi_cd": float(cd)}])
    fried.to_csv(os.path.join(args.out, "friedman.csv"), index=False)
    nem = summary[["method", "label", "mean_rank"]].copy()
    nem["rank"] = nem["mean_rank"].rank(method="average")
    nem["ci_low"] = nem["mean_rank"] - cd / 2
    nem["ci_high"] = nem["mean_rank"] + cd / 2
    nem.sort_values("mean_rank").to_csv(os.path.join(args.out, "nemenyi.csv"), index=False)

    # leave-one-dataset-out
    loo = []
    for held in datasets:
        keep = [d for d in datasets if d != held]
        avg = ari.loc[keep].mean()
        for m in methods:
            loo.append({"held_out": held, "method": m, "label": LABEL[m],
                        "loo_mean_ari": float(avg[m]),
                        "change": float(avg[m] - ari[m].mean())})
    pd.DataFrame(loo).to_csv(os.path.join(args.out, "loo.csv"), index=False)

    # subgroups
    meta = ok.groupby("dataset")[["N", "K", "d"]].first()
    groups = {
        "all": datasets,
        "convex": [d for d in datasets if d not in ("circles", "moons")],
        "non-convex": [d for d in datasets if d in ("circles", "moons")],
        "K<=6": [d for d in datasets if meta.loc[d, "K"] <= 6],
        "7<=K<=19": [d for d in datasets if 7 <= meta.loc[d, "K"] <= 19],
        "K>=20": [d for d in datasets if meta.loc[d, "K"] >= 20],
        "N<10^4": [d for d in datasets if meta.loc[d, "N"] < 1e4],
        "N>=10^4": [d for d in datasets if meta.loc[d, "N"] >= 1e4],
    }
    sub_rows = []
    for name, ds in groups.items():
        if not ds:
            continue
        for m in methods:
            sub_rows.append({"subgroup": name, "n_datasets": len(ds),
                             "datasets": ",".join(ds), "method": m,
                             "label": LABEL[m],
                             "mean_ari": float(ari.loc[ds, m].mean()),
                             "mean_rank": float(ari_ranks.loc[ds, m].mean())})
    pd.DataFrame(sub_rows).to_csv(os.path.join(args.out, "subgroups.csv"), index=False)

    # best of R
    best = ok.groupby(["dataset", "method"]).ari.max().unstack()
    bo = [{"method": m, "label": LABEL[m],
           "mean_best": float(best[m].mean()),
           "mean_best_rank": float(best.rank(axis=1, ascending=False)[m].mean()),
           "mean_worst": float(ok.groupby(["dataset", "method"]).ari.min().unstack()[m].mean())}
          for m in methods]
    pd.DataFrame(bo).to_csv(os.path.join(args.out, "best_of.csv"), index=False)

    # times
    tm = []
    for d in datasets:
        row = {"dataset": d, "N": int(meta.loc[d, "N"]), "K": int(meta.loc[d, "K"]),
               "d": int(meta.loc[d, "d"])}
        for m in methods:
            row[m + "_seed"] = geomean(ok[(ok.dataset == d) & (ok.method == m)].time_seed)
            row[m + "_total"] = geomean(ok[(ok.dataset == d) & (ok.method == m)].time_total)
        tm.append(row)
    pd.DataFrame(tm).to_csv(os.path.join(args.out, "times.csv"), index=False)

    # seed-time ratio per dataset (k-means++ / KMeans-GS)
    t = pd.DataFrame(tm).set_index("dataset")
    ratio = (t["kmeanspp_seed"] / t["kmeans_gs_seed"]).sort_values()
    ratio.to_csv(os.path.join(args.out, "seed_ratio.csv"),
                 header=["kmeanspp_over_gs"])

    report = {
        "datasets": datasets, "methods": methods, "runs": int(len(ok)),
        "friedman": fried.iloc[0].to_dict(),
        "mean_ari": {m: float(summary.loc[summary.method == m, "mean_ari"].iloc[0])
                     for m in methods},
        "mean_rank": {m: float(summary.loc[summary.method == m, "mean_rank"].iloc[0])
                      for m in methods},
        "seed_geomean": {m: float(summary.loc[summary.method == m, "time_seed_geo"].iloc[0])
                         for m in methods},
        "total_geomean": {m: float(summary.loc[summary.method == m, "time_total_geo"].iloc[0])
                          for m in methods},
        "seed_ratio_median": float(ratio.median()),
        "seed_ratio_below_one": int((ratio < 1).sum()),
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(summary.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\npairwise\n", pairwise.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nseed ratio (k-means++/KMeans-GS): median %.2f, below 1 on %d/%d datasets"
          % (ratio.median(), int((ratio < 1).sum()), len(ratio)))
    print("Friedman p=%.4g  Nemenyi CD=%.2f" % (fr.pvalue, cd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
