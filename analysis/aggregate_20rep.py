from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, rankdata, wilcoxon


METHOD_ORDER = [
    "kmeans_gs",
    "kmeanspp",
    "kmeans",
    "kmeans_gs_g",
    "mbkmeans",
    "kmeans_par",
    "afkmc2",
    "birch",
    "gmm",
    "dbscan",
]
LABEL = {
    "kmeans_gs": "KMeans-GS",
    "kmeanspp": "k-means++",
    "kmeans": "K-means (random)",
    "kmeans_gs_g": "KMeans-GS-G",
    "mbkmeans": "MiniBatchKMeans",
    "kmeans_par": "k-means∥",
    "afkmc2": "AFK-MC²",
    "birch": "BIRCH",
    "gmm": "Gaussian mixture",
    "dbscan": "DBSCAN",
}
METRICS = {
    "ari": ("ARI", True),
    "ami": ("AMI", True),
    "fmi": ("FMI", True),
    "mse": ("MSE", False),
    "silhouette": ("Silhouette", True),
    "davies_bouldin": ("Davies-Bouldin", False),
    "time_total": ("Total time (s)", False),
    "time_seed": ("Seeding time (s)", False),
    "time_iter": ("Lloyd time (s)", False),
    "n_iter": ("Lloyd iterations", False),
}
EXTRA_METRICS = {
    "seed_min_dist": ("Seed minimum distance", True),
    "seed_mean_dist": ("Seed mean distance", True),
    "sse": ("SSE", False),
    "n_clusters_found": ("Clusters found", True),
    "noise_frac": ("Noise fraction", False),
    "kmeans_gs_fallbacks": ("GS fallbacks", False),
    "kmeans_gs_trials_mean": ("GS mean trials", False),
    "kmeans_gs_trials_max": ("GS max trials", False),
}


def read_dir(path: str) -> pd.DataFrame:
    frames = []
    for file in sorted(glob.glob(os.path.join(path, "*.csv"))):
        if os.path.basename(file).startswith("_"):
            continue
        frame = pd.read_csv(file)
        if "method" in frame.columns and "status" in frame.columns:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No run-level CSV files found")
    out = pd.concat(frames, ignore_index=True)
    for col in [
        "repeat_seed", "repeat", "ari", "ami", "fmi", "mse", "silhouette",
        "davies_bouldin", "time_seed", "time_iter", "time_total", "n_iter",
        "N", "K", "d", "kmeans_gs_trials_mean", "kmeans_gs_fallbacks",
        "seed_min_dist", "seed_mean_dist", "sse", "n_clusters_found",
        "noise_frac", "kmeans_gs_trials_max",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def trim_mean(values, fraction):
    values = np.sort(np.asarray(values, dtype=float))
    cut = int(math.floor(len(values) * fraction))
    if cut:
        values = values[cut:-cut]
    return float(values.mean())


def geomean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.log(values).mean())) if values.size else np.nan


def bootstrap_mean_ci(values, rng, reps=20000):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def safe_wilcoxon(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if not d.size or np.allclose(d, 0):
        return 1.0, 1.0
    res = wilcoxon(d, zero_method="wilcox", method="auto")
    return float(res.statistic), float(res.pvalue)


def rank_biserial(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    if not d.size:
        return 0.0
    r = rankdata(np.abs(d))
    pos = float(r[d > 0].sum())
    neg = float(r[d < 0].sum())
    return (pos - neg) / (pos + neg)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted_sorted[rank] = min(1.0, running)
    out = np.empty_like(p)
    out[order] = adjusted_sorted
    return out.tolist()


def rank_matrix(matrix, higher_is_better):
    rows = []
    for row in matrix.to_numpy(float):
        valid = np.isfinite(row)
        ranks = np.full(len(row), np.nan)
        if valid.any():
            ranks[valid] = rankdata(-row[valid] if higher_is_better else row[valid],
                                    method="average")
        rows.append(ranks)
    return pd.DataFrame(rows, index=matrix.index, columns=matrix.columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--old", default="")
    args = parser.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261019)

    raw = read_dir(args.raw)
    valid = raw[raw.status == "ok"].copy()
    expected = 13 * 10 * 20
    if len(valid) != expected:
        raise RuntimeError(f"Expected {expected} successful runs, got {len(valid)}")
    counts = valid.groupby(["dataset", "method"]).size()
    if len(counts) != 130 or not counts.eq(20).all():
        raise RuntimeError("Expected 13 datasets x 10 methods x 20 runs")
    methods = [m for m in METHOD_ORDER if m in set(valid.method)]
    datasets = sorted(valid.dataset.unique())
    if len(methods) != 10 or len(datasets) != 13:
        raise RuntimeError("Final method/dataset coverage is incomplete")

    raw["method_label"] = raw["method"].map(LABEL)
    valid["method_label"] = valid["method"].map(LABEL)

    means = {}
    sds = {}
    medians = {}
    mins = {}
    maxs = {}
    for metric in METRICS:
        group = valid.groupby(["dataset", "method"])[metric]
        means[metric] = group.mean().unstack().reindex(index=datasets, columns=methods)
        sds[metric] = group.std(ddof=1).unstack().reindex(index=datasets, columns=methods)
        medians[metric] = group.median().unstack().reindex(index=datasets, columns=methods)
        mins[metric] = group.min().unstack().reindex(index=datasets, columns=methods)
        maxs[metric] = group.max().unstack().reindex(index=datasets, columns=methods)

    ari = means["ari"]
    ari_ranks = rank_matrix(ari, True)
    metric_ranks = {
        metric: rank_matrix(means[metric], higher)
        for metric, (_, higher) in METRICS.items()
    }

    time_geo = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    seed_geo = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    iter_geo = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    for dataset in datasets:
        for method in methods:
            subset = valid[(valid.dataset == dataset) & (valid.method == method)]
            time_geo.loc[dataset, method] = geomean(subset.time_total)
            seed_geo.loc[dataset, method] = geomean(subset.time_seed)
            iter_geo.loc[dataset, method] = geomean(subset.time_iter)

    ds_n = valid.groupby("dataset").N.first()
    ds_k = valid.groupby("dataset").K.first()
    ds_d = valid.groupby("dataset").d.first()
    total_n = float(ds_n.sum())

    rows = []
    for method in methods:
        a = ari[method].to_numpy(float)
        ci_low, ci_high = bootstrap_mean_ci(a, rng)
        row = {
            "method": method,
            "method_label": LABEL[method],
            "mean_ari": a.mean(),
            "mean_ari_ci_low": ci_low,
            "mean_ari_ci_high": ci_high,
            "median_ari": np.median(a),
            "trimmed_ari_10pct": trim_mean(a, 0.10),
            "trimmed_ari_20pct": trim_mean(a, 0.20),
            "mean_rank": ari_ranks[method].mean(),
            "top1_datasets": int((ari_ranks[method] <= 1).sum()),
            "top3_datasets": int((ari_ranks[method] <= 3).sum()),
            "bottom3_datasets": int((ari_ranks[method] >= len(methods) - 2).sum()),
            "n_weighted_ari": float(sum(ds_n[d] * ari.loc[d, method] for d in datasets) / total_n),
            "dataset_sd": a.std(ddof=1),
            "dataset_min": a.min(),
            "dataset_max": a.max(),
            "dataset_range": a.max() - a.min(),
            "time_arith_mean": means["time_total"][method].mean(),
            "time_geomean": geomean(time_geo[method]),
            "seed_arith_mean": means["time_seed"][method].mean(),
            "seed_geomean": geomean(seed_geo[method]),
            "iter_arith_mean": means["time_iter"][method].mean(),
            "iter_geomean": geomean(iter_geo[method]),
            "iterations_mean": means["n_iter"][method].mean(),
        }
        for metric, (label, _) in METRICS.items():
            row[f"{label} mean"] = means[metric][method].mean()
            row[f"{label} mean rank"] = metric_ranks[metric][method].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    reference_time = float(summary.loc[summary.method == "kmeans_gs", "time_geomean"].iloc[0])
    summary["time_geomean_vs_kmeans_gs"] = summary["time_geomean"] / reference_time
    summary["ari_rank_position"] = summary["mean_ari"].rank(ascending=False, method="min").astype(int)
    summary["mean_rank_position"] = summary["mean_rank"].rank(ascending=True, method="min").astype(int)

    aggregation = summary[[
        "method", "method_label", "mean_ari", "mean_ari_ci_low", "mean_ari_ci_high",
        "median_ari", "trimmed_ari_10pct", "trimmed_ari_20pct", "mean_rank",
        "top1_datasets", "top3_datasets", "bottom3_datasets", "n_weighted_ari",
        "dataset_sd", "dataset_min", "dataset_max", "dataset_range",
    ]].copy()
    for col in ["mean_ari", "median_ari", "trimmed_ari_10pct", "trimmed_ari_20pct"]:
        aggregation[col + "_rank"] = aggregation[col].rank(ascending=False, method="min").astype(int)

    # Pairwise comparisons against KMeans-GS.
    comparisons = [m for m in methods if m != "kmeans_gs"]
    run_wide = valid.pivot_table(index=["dataset", "repeat_seed"], columns="method",
                                 values="ari", aggfunc="first")
    pair_rows = []
    pvals = []
    for ref in comparisons:
        run_diff = (run_wide["kmeans_gs"] - run_wide[ref]).dropna()
        ds_diff = ari["kmeans_gs"] - ari[ref]
        low, high = bootstrap_mean_ci(ds_diff.to_numpy(), rng)
        run_w, run_p = safe_wilcoxon(run_wide["kmeans_gs"], run_wide[ref])
        ds_w, ds_p = safe_wilcoxon(ari["kmeans_gs"], ari[ref])
        wins = int((ds_diff > 1e-12).sum())
        losses = int((ds_diff < -1e-12).sum())
        ties = len(ds_diff) - wins - losses
        sign_p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
        pair_rows.append({
            "comparison": f"KMeans-GS minus {LABEL[ref]}",
            "mean_dataset_difference": ds_diff.mean(),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "dataset_wins": wins,
            "dataset_ties": ties,
            "dataset_losses": losses,
            "dataset_sign_test_p": sign_p,
            "dataset_wilcoxon_statistic": ds_w,
            "dataset_wilcoxon_p": ds_p,
            "run_mean_difference": run_diff.mean(),
            "run_wilcoxon_statistic": run_w,
            "run_wilcoxon_p": run_p,
            "run_rank_biserial": rank_biserial(run_wide["kmeans_gs"], run_wide[ref]),
        })
        pvals.append(run_p)
    pairwise = pd.DataFrame(pair_rows)
    pairwise["run_wilcoxon_p_holm"] = holm(pvals)

    # Dataset-level evidence table: ranking and per-dataset deltas.
    dataset_rows = []
    for dataset in datasets:
        row = {
            "dataset": dataset,
            "N": int(ds_n[dataset]),
            "d": int(ds_d[dataset]),
            "K": int(ds_k[dataset]),
        }
        for method in methods:
            row[f"{LABEL[method]} ARI"] = ari.loc[dataset, method]
        for ref in ["kmeanspp", "kmeans", "kmeans_gs_g"]:
            row[f"GS minus {LABEL[ref]}"] = ari.loc[dataset, "kmeans_gs"] - ari.loc[dataset, ref]
        baseline_methods = [
            m for m in methods
            if m not in {"kmeans_gs", "kmeans_gs_g"}
        ]
        best_baseline = ari.loc[dataset, baseline_methods].idxmax()
        row["best non-GS baseline"] = LABEL[best_baseline]
        row["best non-GS ARI"] = ari.loc[dataset, best_baseline]
        row["GS minus best non-GS baseline"] = (
            ari.loc[dataset, "kmeans_gs"] - ari.loc[dataset, best_baseline]
        )
        row["GS ARI rank"] = ari_ranks.loc[dataset, "kmeans_gs"]
        dataset_rows.append(row)
    dataset_evidence = pd.DataFrame(dataset_rows)

    # Leave-one-dataset-out summaries.
    loo_rows = []
    for held in datasets:
        keep = [d for d in datasets if d != held]
        avg = ari.loc[keep].mean()
        rank_pos = avg.rank(ascending=False, method="min")
        avg_rank = ari_ranks.loc[keep].mean()
        for method in methods:
            loo_rows.append({
                "held_out_dataset": held,
                "method": method,
                "method_label": LABEL[method],
                "loo_mean_ari": avg[method],
                "loo_mean_ari_rank": rank_pos[method],
                "loo_average_within_dataset_rank": avg_rank[method],
                "change_from_full_mean": avg[method] - ari[method].mean(),
            })
    loo = pd.DataFrame(loo_rows)

    # Per-dataset/method detail with all metrics.
    detail_rows = []
    for dataset in datasets:
        for method in methods:
            g = valid[(valid.dataset == dataset) & (valid.method == method)]
            row = {
                "dataset": dataset, "N": int(ds_n[dataset]), "d": int(ds_d[dataset]),
                "K": int(ds_k[dataset]), "method": method, "method_label": LABEL[method],
                "runs": len(g),
            }
            for metric, (label, _) in METRICS.items():
                vals = g[metric].to_numpy(float)
                valid_vals = vals[np.isfinite(vals)]
                row[f"{label} mean"] = (
                    float(valid_vals.mean()) if valid_vals.size else None
                )
                row[f"{label} median"] = (
                    float(np.median(valid_vals)) if valid_vals.size else None
                )
                row[f"{label} SD"] = (
                    float(valid_vals.std(ddof=1)) if valid_vals.size > 1 else None
                )
                row[f"{label} min"] = (
                    float(valid_vals.min()) if valid_vals.size else None
                )
                row[f"{label} max"] = (
                    float(valid_vals.max()) if valid_vals.size else None
                )
                row[f"{label} range"] = (
                    float(valid_vals.max() - valid_vals.min())
                    if valid_vals.size else None
                )
            for metric, (label, _) in EXTRA_METRICS.items():
                vals = g[metric].to_numpy(float)
                valid_vals = vals[np.isfinite(vals)]
                row[f"{label} mean"] = (
                    float(valid_vals.mean()) if valid_vals.size else None
                )
                row[f"{label} median"] = (
                    float(np.median(valid_vals)) if valid_vals.size else None
                )
                row[f"{label} SD"] = (
                    float(valid_vals.std(ddof=1)) if valid_vals.size > 1 else None
                )
                row[f"{label} min"] = (
                    float(valid_vals.min()) if valid_vals.size else None
                )
                row[f"{label} max"] = (
                    float(valid_vals.max()) if valid_vals.size else None
                )
                row[f"{label} range"] = (
                    float(valid_vals.max() - valid_vals.min())
                    if valid_vals.size else None
                )
            row["ARI rank within dataset"] = float(ari_ranks.loc[dataset, method])
            row["ARI difference vs KMeans-GS"] = float(ari.loc[dataset, method] - ari.loc[dataset, "kmeans_gs"])
            detail_rows.append(row)
    detail = pd.DataFrame(detail_rows)

    # Method-level matrix ranks and time ratios.
    matrix_rank = ari_ranks.rename(columns=LABEL)
    matrix_ari = ari.rename(columns=LABEL)
    matrix_time = means["time_total"].rename(columns=LABEL)
    matrix_time_geo = time_geo.rename(columns=LABEL)

    # Friedman test on the 13 dataset-level ARI means.
    friedman = friedmanchisquare(*[ari[m].to_numpy() for m in methods])
    friedman_table = pd.DataFrame([{
        "test": "Friedman rank test",
        "n_datasets": len(datasets),
        "n_methods": len(methods),
        "statistic": float(friedman.statistic),
        "df": len(methods) - 1,
        "p_value": float(friedman.pvalue),
        "kendall_w": float(friedman.statistic / (len(datasets) * (len(methods) - 1))),
    }])

    # Evidence map: fixed, pre-specified descriptive checks.
    evidence = []
    for _, row in summary.iterrows():
        evidence.append({
            "method": row["method_label"],
            "mean ARI": row["mean_ari"],
            "mean rank": row["mean_rank"],
            "top-3 datasets": row["top3_datasets"],
            "median ARI": row["median_ari"],
            "10% trimmed ARI": row["trimmed_ari_10pct"],
            "N-weighted ARI": row["n_weighted_ari"],
            "geomean time": row["time_geomean"],
            "time vs KMeans-GS": row["time_geomean_vs_kmeans_gs"],
        })
    evidence = pd.DataFrame(evidence)

    # Exploratory subgroup views. These groups were not used to select datasets
    # or methods, and are reported as descriptive checks rather than main tests.
    subgroup_defs = []
    for low, high, label in [
        (0, 6, "K <= 6"),
        (7, 19, "7 <= K <= 19"),
        (20, 10**9, "K >= 20"),
    ]:
        subgroup_defs.append((
            label,
            [d for d in datasets if low <= int(ds_k[d]) <= high],
        ))
    for low, high, label in [(0, 50, "d <= 50"), (51, 10**9, "d > 50")]:
        subgroup_defs.append((
            label,
            [d for d in datasets if low <= int(ds_d[d]) <= high],
        ))
    for low, high, label in [
        (0, 999, "N < 1,000"),
        (1000, 10**9, "N >= 1,000"),
    ]:
        subgroup_defs.append((
            label,
            [d for d in datasets if low <= int(ds_n[d]) <= high],
        ))

    subgroup_rows = []
    for group_label, group_datasets in subgroup_defs:
        if not group_datasets:
            continue
        for method in methods:
            subgroup_rows.append({
                "subgroup": group_label,
                "datasets": ", ".join(group_datasets),
                "n_datasets": len(group_datasets),
                "method": method,
                "method_label": LABEL[method],
                "mean_ari": ari.loc[group_datasets, method].mean(),
                "mean_rank": ari_ranks.loc[group_datasets, method].mean(),
                "vs_kmeanspp": (
                    ari.loc[group_datasets, method]
                    - ari.loc[group_datasets, "kmeanspp"]
                ).mean(),
                "vs_random": (
                    ari.loc[group_datasets, method]
                    - ari.loc[group_datasets, "kmeans"]
                ).mean(),
                "vs_gs_g": (
                    ari.loc[group_datasets, method]
                    - ari.loc[group_datasets, "kmeans_gs_g"]
                ).mean(),
            })
    subgroups = pd.DataFrame(subgroup_rows)

    # Seed split-half check: the first ten and second ten paired seeds.
    half_ari = {}
    half_ranks = {}
    for half_label, mask in [
        ("42-51", valid.repeat_seed.between(42, 51)),
        ("52-61", valid.repeat_seed.between(52, 61)),
    ]:
        h = valid[mask]
        h_ari = h.groupby(["dataset", "method"]).ari.mean().unstack()
        half_ari[half_label] = h_ari
        half_ranks[half_label] = rank_matrix(h_ari, True)
    seed_half_rows = []
    for method in methods:
        seed_half_rows.append({
            "method": method,
            "method_label": LABEL[method],
            "ARI seeds 42-51": half_ari["42-51"][method].mean(),
            "ARI seeds 52-61": half_ari["52-61"][method].mean(),
            "ARI change second minus first": (
                half_ari["52-61"][method].mean()
                - half_ari["42-51"][method].mean()
            ),
            "mean rank seeds 42-51": half_ranks["42-51"][method].mean(),
            "mean rank seeds 52-61": half_ranks["52-61"][method].mean(),
        })
    seed_halves = pd.DataFrame(seed_half_rows)

    # Optional 10-vs-20 comparison.
    comparison = pd.DataFrame()
    if args.old and os.path.isdir(args.old):
        old = read_dir(args.old)
        old = old[old.status == "ok"]
        old_ari = old.groupby(["dataset", "method"]).ari.mean().unstack()
        old_time = old.groupby(["dataset", "method"]).time_total.mean().unstack()
        compare_rows = []
        for method in methods:
            if method not in old_ari:
                continue
            compare_rows.append({
                "method": LABEL[method],
                "10-run ARI": old_ari[method].mean(),
                "20-run ARI": ari[method].mean(),
                "ARI change": ari[method].mean() - old_ari[method].mean(),
                "10-run mean time": old_time[method].mean(),
                "20-run mean time": means["time_total"][method].mean(),
            })
        comparison = pd.DataFrame(compare_rows)

    # Presentation payloads for the workbook.
    payload = {
        "protocol": {
            "datasets": len(datasets),
            "methods": len(methods),
            "repetitions": 20,
            "runs": int(len(valid)),
            "seeds": "42-61",
        },
        "summary": summary,
        "aggregation": aggregation,
        "pairwise": pairwise,
        "friedman": friedman_table,
        "evidence": evidence,
        "dataset_evidence": dataset_evidence,
        "subgroups": subgroups,
        "seed_halves": seed_halves,
        "detail": detail,
        "loo": loo,
        "comparison_10_vs_20": comparison,
        "matrix_ari": matrix_ari,
        "matrix_ari_rank": matrix_rank,
        "matrix_time": matrix_time,
        "matrix_time_geo": matrix_time_geo,
        "all_runs": raw,
    }
    for name, frame in payload.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(
                outdir / f"{name}.csv",
                index=name.startswith("matrix_"),
                index_label="dataset" if name.startswith("matrix_") else None,
            )
    for metric, (label, _) in METRICS.items():
        means[metric].rename(columns=LABEL).to_csv(
            outdir / f"matrix_{metric}_mean.csv", index_label="dataset"
        )
        sds[metric].rename(columns=LABEL).to_csv(
            outdir / f"matrix_{metric}_sd.csv", index_label="dataset"
        )
        metric_ranks[metric].rename(columns=LABEL).to_csv(
            outdir / f"matrix_{metric}_rank.csv", index_label="dataset"
        )
    time_geo.rename(columns=LABEL).to_csv(
        outdir / "matrix_time_total_geomean.csv", index_label="dataset"
    )
    seed_geo.rename(columns=LABEL).to_csv(
        outdir / "matrix_time_seed_geomean.csv", index_label="dataset"
    )
    iter_geo.rename(columns=LABEL).to_csv(
        outdir / "matrix_time_iter_geomean.csv", index_label="dataset"
    )
    outdir.joinpath("report.json").write_text(
        json.dumps({
            "protocol": payload["protocol"],
            "friedman": friedman_table.iloc[0].to_dict(),
            "pairwise": pairwise.to_dict(orient="records"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Wrote", outdir)
    print(summary[[
        "method_label", "mean_ari", "mean_rank", "median_ari",
        "trimmed_ari_10pct", "trimmed_ari_20pct", "top3_datasets",
        "n_weighted_ari", "time_geomean", "time_geomean_vs_kmeans_gs",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPairwise\n", pairwise.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
