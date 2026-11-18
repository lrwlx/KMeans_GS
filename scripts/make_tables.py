#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the manuscript tables from the aggregated results.

Every number is derived from the CSV files produced by the analysis scripts.
Outputs markdown tables (for pandoc) plus a JSON bundle.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

MAIN_ORDER = ["kmeans_gs", "kmeanspp", "kmeanspp_greedy", "kmeanspp_sklearn",
              "kmeans", "afkmc2", "kmeans_par", "kmeans_gs_g",
              "mbkmeans", "birch", "gmm", "dbscan"]
SHORT = {"kmeans_gs": "KMeans-GS", "kmeans_gs_g": "KMeans-GS-G",
         "kmeanspp": "k-means++", "kmeanspp_greedy": "k-means++ (greedy)",
         "kmeanspp_sklearn": "k-means++ (compiled)", "kmeans": "K-means (random)",
         "afkmc2": "AFK-MC2", "kmeans_par": "k-means∥", "mbkmeans": "MiniBatchKMeans",
         "birch": "BIRCH", "gmm": "Gaussian mixture", "dbscan": "DBSCAN",
         "sklearn1": "scikit-learn KMeans (1 restart)",
         "sklearn10": "scikit-learn KMeans (10 restarts)"}


def md(df, fmts=None):
    fmts = fmts or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            try:
                missing = bool(pd.isna(v))
            except (TypeError, ValueError):
                missing = False
            if missing:
                vals.append("-")
                continue
            f = fmts.get(c)
            if f is not None:
                try:
                    vals.append(f % v)
                    continue
                except (TypeError, ValueError):
                    pass
            vals.append(str(v).replace("||", "\u2225").replace("|", "\u2225"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_md(df, path, fmts=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(md(df, fmts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tierA", default="results/summary_tierA")
    p.add_argument("--tierB", default="results/summary_tierB")
    p.add_argument("--apps", default="results/summary_applications")
    p.add_argument("--scaling", default="results/scaling")
    p.add_argument("--sensitivity", default="results/sensitivity")
    p.add_argument("--out", default="paper/tables")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bundle = {}

    # ---- Table 1: datasets ------------------------------------------------
    per = pd.read_csv(os.path.join(args.tierA, "per_dataset.csv"))
    t1 = per[["dataset", "N", "K", "d"]].drop_duplicates().sort_values("N")
    t1 = t1.rename(columns={"dataset": "Dataset", "N": "N", "K": "K", "d": "d"})
    t1["Tier"] = "A"
    if os.path.exists(os.path.join(args.tierB, "per_dataset.csv")):
        b = pd.read_csv(os.path.join(args.tierB, "per_dataset.csv"))
        b1 = b[["dataset", "N", "K", "d"]].drop_duplicates().sort_values("N")
        b1 = b1.rename(columns={"dataset": "Dataset"})
        b1["Tier"] = "B"
        t1 = pd.concat([b1, t1], ignore_index=True)
    t1 = t1.sort_values("N").reset_index(drop=True)
    t1["N"] = t1["N"].map(lambda x: "{:,}".format(int(x)))
    t1.to_csv(os.path.join(args.out, "table1_datasets.csv"), index=False)
    save_md(t1, os.path.join(args.out, "table1_datasets.md"))
    bundle["table1"] = t1.to_dict(orient="records")

    # ---- Table 2: main ranking -------------------------------------------
    s = pd.read_csv(os.path.join(args.tierA, "summary.csv"))
    s["ord"] = s["method"].map({m: i for i, m in enumerate(MAIN_ORDER)})
    s = s.sort_values("ord")
    gs_seed = float(s.loc[s.method == "kmeans_gs", "time_seed_geo"].iloc[0])
    gs_tot = float(s.loc[s.method == "kmeans_gs", "time_total_geo"].iloc[0])
    rows = []
    for _, r in s.iterrows():
        rows.append({
            "Method": SHORT.get(r.method, r.label),
            "Mean ARI": "%.4f" % r.mean_ari,
            "Median": "%.4f" % r.median_ari,
            "Mean rank": "%.2f" % r.mean_rank,
            "Best": "%d" % r.top1,
            "Seed (ms)": "%.2f" % (1e3 * r.time_seed_geo) if np.isfinite(r.time_seed_geo) else "-",
            "Total (ms)": "%.1f" % (1e3 * r.time_total_geo),
            "Iters": ("%.1f" % r.n_iter_mean) if (np.isfinite(r.n_iter_mean) and r.n_iter_mean > 0) else "-",
            "Seed ratio vs GS": "%.2f" % (r.time_seed_geo / gs_seed) if np.isfinite(r.time_seed_geo) else "-",
            "Total ratio vs GS": "%.2f" % (r.time_total_geo / gs_tot),
        })
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(args.out, "table2_main.csv"), index=False)
    save_md(t2, os.path.join(args.out, "table2_main.md"))
    bundle["table2"] = rows

    # ---- Table 3: pairwise ------------------------------------------------
    pw = pd.read_csv(os.path.join(args.tierA, "pairwise.csv"))
    rows = []
    for _, r in pw.iterrows():
        rows.append({
            "Comparison": r.comparison.replace("KMeans-GS minus ", "vs "),
            "\u0394ARI": "%+.4f" % r.mean_diff,
            "95% CI": "[%+.3f, %+.3f]" % (r.ci_lo, r.ci_hi),
            "W/T/L": "%d/%d/%d" % (r.wins, r.ties, r.losses),
            "Wilcoxon p": "%.3f" % r.wilcoxon_p,
            "Holm p": "%.3f" % r.holm_p,
            "r": "%+.2f" % r.rank_biserial,
        })
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(args.out, "table3_pairwise.csv"), index=False)
    save_md(t3, os.path.join(args.out, "table3_pairwise.md"))
    bundle["table3"] = rows

    # ---- Table 4: large tabular ------------------------------------------
    pathB = os.path.join(args.tierB, "per_dataset.csv")
    if os.path.exists(pathB):
        b = pd.read_csv(pathB)
        order = ["kmeans_gs", "kmeans_gs_g", "kmeanspp", "kmeanspp_greedy",
                 "kmeanspp_sklearn", "kmeans", "afkmc2", "kmeans_par",
                 "mbkmeans", "birch", "gmm", "sklearn1", "sklearn10", "dbscan"]
        rows = []
        for m in order:
            sub = b[b.method == m]
            if not len(sub):
                continue
            row = {"Method": SHORT.get(m, m)}
            for d in ["mnist", "covertype"]:
                x = sub[sub.dataset == d]
                row["%s ARI" % d] = "%.4f" % x.ari.iloc[0] if len(x) else "-"
                row["%s seed (s)" % d] = ("%.3f" % x.time_seed.iloc[0]) if len(x) and np.isfinite(x.time_seed.iloc[0]) else "-"
                row["%s total (s)" % d] = ("%.2f" % x.time_total.iloc[0]) if len(x) else "-"
                row["%s iters" % d] = ("%d" % x.n_iter.iloc[0]) if len(x) else "-"
            rows.append(row)
        t4 = pd.DataFrame(rows)
        t4.to_csv(os.path.join(args.out, "table5_large.csv"), index=False)
        save_md(t4, os.path.join(args.out, "table5_large.md"))
        bundle["table4"] = rows

    # ---- Tables 5-7: applications ----------------------------------------
    ap = os.path.join(args.apps, "application_summary.csv")
    if os.path.exists(ap):
        a = pd.read_csv(ap)
        for key, name in [("covid_k18", "table6_covid"), ("hlca_k25", "table7_hlca"),
                          ("dbpedia14", "table8_text")]:
            sub = a[a.dataset == key]
            if not len(sub):
                continue
            rows = []
            for _, r in sub.iterrows():
                rows.append({
                    "Method": SHORT.get(r.method, r.label),
                    "ARI mean": "%.4f" % r.ari_mean,
                    "SD": "%.4f" % r.ari_sd,
                    "ARI selected": "%.4f" % r.ari_selected,
                    "Total cost (s)": "%.1f" % r.cost_total,
                    "Iters": "%.1f" % r.iter_mean,
                    "Seed (ms)": ("%.2f" % (1e3 * r.seed_geo)) if np.isfinite(r.seed_geo) else "-",
                })
            t = pd.DataFrame(rows)
            t.to_csv(os.path.join(args.out, name + ".csv"), index=False)
            save_md(t, os.path.join(args.out, name + ".md"))
            bundle[name] = rows

    # ---- Table 8: scaling exponents --------------------------------------
    ex = os.path.join(args.scaling, "scaling_exponents.csv")
    if os.path.exists(ex):
        per_mech = pd.read_csv(ex)
        methods = ["kmeans_gs", "kmeans_gs_g", "kmeanspp", "kmeanspp_greedy",
                   "kmeanspp_sklearn", "kmeans_par", "afkmc2", "kmeans"]
        rows = []
        for m in methods:
            sub = per_mech[per_mech.method == m]
            if not len(sub):
                continue
            def rng(col):
                lo, hi = sub[col].min(), sub[col].max()
                return "%.2f" % lo if abs(hi - lo) < 5e-3 else "[%.2f, %.2f]" % (lo, hi)
            rows.append({"Method": SHORT.get(m, m),
                         "N exponent": rng("N_exp"), "K exponent": rng("K_exp"),
                         "d exponent": rng("d_exp"),
                         "R2 (min)": "%.3f" % sub["mv_r2"].min()})
        t8 = pd.DataFrame(rows)
        t8.to_csv(os.path.join(args.out, "table4_scaling.csv"), index=False)
        save_md(t8, os.path.join(args.out, "table4_scaling.md"))
        bundle["table8"] = rows

    # ---- Table 9: sensitivity --------------------------------------------
    sp = os.path.join(args.sensitivity, "sensitivity_raw.csv")
    if os.path.exists(sp):
        sv = pd.read_csv(sp)
        sv = sv[sv.status == "ok"]
        g = sv.groupby(["setting", "component"], sort=False).agg(
            n=("ari", "count"), ari=("ari", "mean"),
            seed_ms=("time_seed", lambda x: 1e3 * x.median()),
            fallbacks=("fallbacks", "mean")).reset_index()
        base = float(g.loc[g.setting == "baseline", "ari"].iloc[0])
        rows = []
        for _, r in g.iterrows():
            rows.append({"Setting": r.setting, "Component": r.component,
                         "Mean ARI": "%.4f" % r.ari,
                         "dARI": "%+.4f" % (r.ari - base),
                         "Median seed (ms)": "%.2f" % r.seed_ms,
                         "Fallbacks": "%.2f" % r.fallbacks})
        t9 = pd.DataFrame(rows)
        t9.to_csv(os.path.join(args.out, "table9_sensitivity.csv"), index=False)
        save_md(t9, os.path.join(args.out, "table9_sensitivity.md"))
        bundle["table9"] = rows

    # ---- Table B1: best of R ---------------------------------------------
    bo = pd.read_csv(os.path.join(args.tierA, "best_of.csv"))
    bo["ord"] = bo["method"].map({m: i for i, m in enumerate(MAIN_ORDER)})
    bo = bo.sort_values("ord")
    rows = [{"Method": SHORT.get(r.method, r.label),
             "Best-of-20 ARI": "%.4f" % r.mean_best,
             "Best rank": "%.2f" % r.mean_best_rank,
             "Worst-of-20 ARI": "%.4f" % r.mean_worst} for _, r in bo.iterrows()]
    tb = pd.DataFrame(rows)
    tb.to_csv(os.path.join(args.out, "table10_best.csv"), index=False)
    save_md(tb, os.path.join(args.out, "table10_best.md"))
    bundle["table10"] = rows

    with open(os.path.join(args.out, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print("wrote tables to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
