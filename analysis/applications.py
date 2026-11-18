#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Application-level summaries with the restart protocol as the unit.

For every atlas and for the text corpus we report, per method:
  * restart-averaged ARI / NMI and their standard deviation,
  * the SSE-selected restart (the protocol a user would apply),
  * the summed cost of all restarts (the cost a user actually pays),
  * mean Lloyd iterations over restarts.
Paired tests compare KMeans-GS variants with k-means++ across the same seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LABEL = {
    "kmeans": "K-means (random)", "kmeanspp": "k-means++",
    "kmeanspp_greedy": "k-means++ (greedy)",
    "kmeanspp_sklearn": "k-means++ (compiled)",
    "kmeans_gs": "KMeans-GS", "kmeans_gs_g": "KMeans-GS-G",
    "mbkmeans": "MiniBatchKMeans", "sklearn1": "scikit-learn KMeans (1 restart)",
    "sklearn10": "scikit-learn KMeans (10 restarts)",
}


def summarise(df, key):
    idx = "restart" if "restart" in df.columns else "repeat"
    rows = []
    for m, g in df.groupby("method"):
        sel = g[g["selected"] == True] if "selected" in g else g.iloc[[0]]  # noqa: E712
        rows.append({
            "dataset": key, "method": m, "label": LABEL.get(m, m),
            "restarts": int(g[idx].nunique()),
            "ari_mean": float(g.ari.mean()), "ari_sd": float(g.ari.std(ddof=1)),
            "ari_min": float(g.ari.min()), "ari_max": float(g.ari.max()),
            "nmi_mean": float(g.nmi.mean()),
            "ari_selected": float(sel.ari.iloc[0]),
            "iter_mean": float(g.n_iter.mean()),
            "iter_selected": int(sel.n_iter.iloc[0]),
            "mse_mean": float(g.mse.mean()),
            "mse_selected": float(sel.mse.iloc[0]),
            "seed_geo": float(np.exp(np.log(g.seed_s.dropna()).mean()))
            if g.seed_s.notna().any() else np.nan,
            "cost_total": float(g.total_s.sum()),
            "cost_mean": float(g.total_s.mean()),
            "sil_mean": float(g.silhouette.mean()),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/applications")
    p.add_argument("--out", default="results/summary_applications")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tables = []
    pairs = []
    for f in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        key = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        if "method" not in df:
            continue
        t = summarise(df, key)
        tables.append(t)
        idx = "restart" if "restart" in df.columns else "repeat"
        wide = df.pivot_table(index=idx, columns="method", values="ari")
        if "kmeans_gs" in wide and "kmeanspp" in wide:
            for other in ["kmeanspp", "kmeanspp_greedy", "kmeans_gs_g"]:
                if other not in wide:
                    continue
                d = (wide["kmeans_gs"] - wide[other]).dropna()
                p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
                pairs.append({"dataset": key, "a": "KMeans-GS", "b": LABEL.get(other, other),
                              "n": len(d), "mean_diff": float(d.mean()),
                              "p": float(p) if np.isfinite(p) else np.nan})
        if "kmeans_gs_g" in wide and "kmeanspp" in wide:
            d = (wide["kmeans_gs_g"] - wide["kmeanspp"]).dropna()
            p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
            pairs.append({"dataset": key, "a": "KMeans-GS-G", "b": "k-means++",
                          "n": len(d), "mean_diff": float(d.mean()),
                          "p": float(p) if np.isfinite(p) else np.nan})

    tab = pd.concat(tables, ignore_index=True)
    tab.to_csv(os.path.join(args.out, "application_summary.csv"), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(args.out, "application_tests.csv"),
                               index=False)
    print(tab.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\npaired tests (over restarts)\n",
          pd.DataFrame(pairs).to_string(index=False))
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"tables": tab.to_dict(orient="records"),
                   "tests": pairs}, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
