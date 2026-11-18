#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the manuscript figures from the aggregated CSVs.

Outputs 300 dpi PNG (for the Word manuscript) and vector PDF (for the
submission system).  Figure numbers follow the order in which the figures are
first cited in the text.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C = {"gs": "#1f4e79", "gsg": "#2e7d32", "kpp": "#6e6e6e", "greedy": "#b8860b",
     "compiled": "#8c564b", "par": "#17becf", "afk": "#e377c2", "rand": "#111111",
     "mbk": "#7f7f7f", "birch": "#bcbd22", "gmm": "#9467bd", "dbscan": "#d62728"}

ORDER = ["kmeans_gs", "kmeanspp", "kmeanspp_greedy", "kmeanspp_sklearn",
         "kmeans", "afkmc2", "kmeans_par", "kmeans_gs_g",
         "mbkmeans", "birch", "gmm", "dbscan"]
SHORT = {"kmeans_gs": "KMeans-GS", "kmeans_gs_g": "KMeans-GS-G",
         "kmeanspp": "k-means++", "kmeanspp_greedy": "k-means++ (greedy)",
         "kmeanspp_sklearn": "k-means++ (compiled)", "kmeans": "random",
         "afkmc2": "AFK-MC2", "kmeans_par": "k-means||",
         "mbkmeans": "MiniBatch", "birch": "BIRCH", "gmm": "GMM",
         "dbscan": "DBSCAN"}


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, "%s.%s" % (name, ext)),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_cd(tierA, outdir):
    nem = pd.read_csv(os.path.join(tierA, "nemenyi.csv"))
    fr = pd.read_csv(os.path.join(tierA, "friedman.csv")).iloc[0]
    cd = float(fr["nemenyi_cd"])
    nem = nem.sort_values("mean_rank").reset_index(drop=True)
    n = len(nem)
    fig, ax = plt.subplots(figsize=(5.6, 0.30 * n + 0.9))
    y = np.arange(n)
    best = float(nem.mean_rank.min())
    ax.axvspan(best, best + cd, color="0.90", zorder=0)
    ax.scatter(nem.mean_rank, y, s=22, color=C["gs"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(nem.label.tolist())
    ax.set_xlabel("Mean rank (lower is better)")
    ax.set_ylim(n - 0.4, -1.7)
    ax.set_xlim(min(nem.mean_rank) - 0.5, max(nem.mean_rank) + 0.6)
    ax.axvline(best, color="0.55", lw=0.7, ls=":")
    ax.axvline(best + cd, color="0.55", lw=0.7, ls=":")
    ytop = -1.0
    ax.plot([best, best + cd], [ytop, ytop], color="k", lw=1.2, clip_on=False)
    for xv in (best, best + cd):
        ax.plot([xv, xv], [ytop - 0.18, ytop + 0.18], color="k", lw=1.2,
                clip_on=False)
    ax.text(best + cd / 2, ytop - 0.35, "CD = %.2f" % cd, ha="center",
            va="bottom", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, outdir, "fig2_critical_difference")


def fig_heatmap(tierA, outdir):
    ari = pd.read_csv(os.path.join(tierA, "matrix_ari_mean.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(tierA, "per_dataset.csv"))
    n_by_ds = meta.groupby("dataset").N.first()
    ari = ari.loc[sorted(ari.index, key=lambda d: n_by_ds[d])]
    cols = [c for c in ari.columns if c in SHORT]
    M = ari[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.4, 0.26 * len(ari) + 1.0))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=np.nanmin(M),
                   vmax=np.nanmax(M))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([SHORT[c] for c in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ari)))
    ax.set_yticklabels(["%s (N=%s)" % (d, f"{int(n_by_ds[d]):,}") for d in ari.index])
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=9.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=9.6,
                    color="w" if v < np.nanmean(M) else "k")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("ARI", fontsize=9.5)
    save(fig, outdir, "fig3_heatmap")


def fig_seed_ratio(tierA, outdir):
    t = pd.read_csv(os.path.join(tierA, "times.csv")).set_index("dataset")
    t = t.sort_values("N")
    labels = t.index.tolist()
    y = np.arange(len(labels))
    r_kpp = (t["kmeanspp_seed"] / t["kmeans_gs_seed"]).values
    r_cmp = (t["kmeanspp_sklearn_seed"] / t["kmeans_gs_seed"]).values
    fig, ax = plt.subplots(figsize=(4.4, 0.26 * len(labels) + 0.8))
    ax.barh(y - 0.2, r_kpp, height=0.38, color=C["kpp"], label="k-means++ / KMeans-GS")
    ax.barh(y + 0.2, r_cmp, height=0.38, color=C["compiled"],
            label="compiled k-means++ / KMeans-GS")
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(["%s (N=%s)" % (d, f"{int(t.loc[d,'N']):,}") for d in labels])
    ax.set_xscale("log")
    ax.set_xlabel("Seeding-time ratio (higher means KMeans-GS is cheaper)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, outdir, "fig4_seed_ratio")


def fig_time_split(tierA, outdir):
    seed = pd.read_csv(os.path.join(tierA, "matrix_time_seed_geomean.csv"), index_col=0)
    it = pd.read_csv(os.path.join(tierA, "matrix_time_iter_geomean.csv"), index_col=0)
    methods = [m for m in ORDER if m in seed.columns and m in it.columns]
    s_geo = np.array([np.exp(np.log(seed[m].dropna()).mean()) for m in methods])
    i_geo = np.array([np.exp(np.log(it[m].dropna()).mean()) for m in methods])
    share = 100.0 * s_geo / (s_geo + i_geo)
    order = np.argsort(share)
    fig, ax = plt.subplots(figsize=(4.4, 0.30 * len(methods) + 1.0))
    yy = np.arange(len(methods))
    ax.barh(yy, share[order], color=[C["gs"] if methods[k] in ("kmeans_gs", "kmeans_gs_g") else "0.55"
                                     for k in order], height=0.62)
    for y, k in zip(yy, order):
        ax.text(share[k] + 0.3, y, "%.1f%%" % share[k], va="center", fontsize=7)
    ax.set_yticks(yy)
    ax.set_yticklabels([SHORT[methods[k]] for k in order])
    ax.set_xlabel("Seeding share of total run time (%, geometric mean)")
    ax.set_xlim(0, max(share) * 1.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, outdir, "fig5_time_split")


def fig_scaling(raw, outdir):
    df = pd.read_csv(raw)
    df = df[df.status == "ok"]
    methods = [("kmeans_gs", "KMeans-GS", C["gs"], "o", "-"),
               ("kmeanspp", "k-means++", C["kpp"], "s", "-"),
               ("kmeanspp_sklearn", "k-means++ (compiled)", C["compiled"], "^", "--"),
               ("afkmc2", "AFK-MC2", C["afk"], "D", "-")]
    with plt.rc_context({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                         "legend.fontsize": 11.5}):
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
        specs = [
            ("N", dict(K=100, d=20), "Number of points $N$"),
            ("K", dict(N=10 ** 6, d=20), "Number of clusters $K$"),
            ("d", dict(N=10 ** 6, K=100), "Dimensionality $d$"),
        ]
        for ax, (var, fixed, xlabel) in zip(axes, specs):
            sub = df.copy()
            for k, v in fixed.items():
                sub = sub[sub[k] == v]
            for m, lab, col, mk, ls in methods:
                gm = sub[sub.method == m].groupby(var).seed_s.agg(["mean", "std"])
                if not len(gm):
                    continue
                ax.plot(gm.index, gm["mean"], marker=mk, ls=ls, color=col, label=lab,
                        ms=3.2)
                if var == "N":
                    lo = gm["mean"] - gm["std"].fillna(0)
                    hi = gm["mean"] + gm["std"].fillna(0)
                    ax.fill_between(gm.index, np.maximum(lo, 1e-6), hi, color=col,
                                    alpha=0.12, lw=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if var == "N":
                ax.set_xticks([1e4, 1e5, 1e6, 1e7])
            elif var == "K":
                ax.set_xticks([10, 100, 500])
                ax.set_xticklabels(["10", "100", "500"])
            else:
                ax.set_xticks([10, 50, 100])
                ax.set_xticklabels(["10", "50", "100"])
            ax.set_xlabel(xlabel)
            if ax is axes[0]:
                ax.set_ylabel("Seeding time (s)")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.31,
                            wspace=0.32)
        save(fig, outdir, "fig6_scaling")


def fig_atlas(appdir, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5))
    order = ["kmeans", "kmeanspp", "kmeanspp_greedy", "kmeans_gs",
             "kmeans_gs_g", "mbkmeans", "sklearn1"]
    labels = ["random", "k-means++", "greedy", "GS", "GS-G", "MBK", "sklearn"]
    colors = [C["rand"], C["kpp"], C["greedy"], C["gs"], C["gsg"], C["mbk"], C["kpp"]]
    for ax, key, title in zip(axes, ["covid_k18", "hlca_k25"],
                              ["COVID-19 atlas (1.46M cells, K = 18)",
                               "Lung atlas (585k cells, K = 25)"]):
        df = pd.read_csv(os.path.join(appdir, key + ".csv"))
        data = [df[df.method == m].ari.values for m in order]
        pos = np.arange(len(order))
        bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker="o", ms=2.5, mfc="0.3", mec="none"),
                        medianprops=dict(color="k", lw=1))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.45)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("ARI over ten restarts")
        ax.set_title(title, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    save(fig, outdir, "fig7_atlases")


def fig_text(appdir, outdir):
    df = pd.read_csv(os.path.join(appdir, "dbpedia14.csv"))
    order = ["kmeans", "kmeanspp", "kmeanspp_greedy", "kmeanspp_sklearn",
             "kmeans_gs", "kmeans_gs_g", "mbkmeans", "sklearn1"]
    labels = ["random", "k-means++", "greedy", "compiled", "GS", "GS-G",
              "MBK", "sklearn"]
    colors = [C["rand"], C["kpp"], C["greedy"], C["compiled"], C["gs"],
              C["gsg"], C["mbk"], C["kpp"]]
    m = [df[df.method == o].ari.mean() for o in order]
    s = [df[df.method == o].ari.std(ddof=1) for o in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    pos = np.arange(len(order))
    axes[0].bar(pos, m, yerr=s, capsize=2, color=colors, alpha=0.65)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("ARI (560k documents)")
    seed_methods = ["kmeans", "kmeanspp", "kmeanspp_greedy", "kmeanspp_sklearn",
                    "kmeans_gs", "kmeans_gs_g"]
    seed_labels = ["random", "k-means++", "greedy", "compiled", "GS", "GS-G"]
    sv = [df[df.method == o].seed_s.median() for o in seed_methods]
    axes[1].bar(np.arange(len(seed_methods)), sv, color="0.55")
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(seed_methods)))
    axes[1].set_xticklabels(seed_labels, rotation=25, ha="right")
    axes[1].set_ylabel("Seeding time (s, median)")
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    save(fig, outdir, "fig8_text")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tierA", default="results/summary_tierA")
    p.add_argument("--apps", default="results/applications")
    p.add_argument("--scaling", default="results/scaling")
    p.add_argument("--sensitivity", default="results/sensitivity")
    p.add_argument("--out", default="paper/figures")
    args = p.parse_args()
    if os.path.exists(os.path.join(args.tierA, "nemenyi.csv")):
        fig_cd(args.tierA, args.out)
        fig_heatmap(args.tierA, args.out)
        fig_seed_ratio(args.tierA, args.out)
        fig_time_split(args.tierA, args.out)
    raw = os.path.join(args.scaling, "scaling_raw.csv")
    if os.path.exists(raw):
        fig_scaling(raw, args.out)
    if os.path.exists(os.path.join(args.apps, "covid_k18.csv")):
        fig_atlas(args.apps, args.out)
    if os.path.exists(os.path.join(args.apps, "dbpedia14.csv")):
        fig_text(args.apps, args.out)
    print("figures written to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
