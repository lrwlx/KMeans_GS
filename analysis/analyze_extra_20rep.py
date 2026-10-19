from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


_HERE = Path(__file__).resolve()
_CANDIDATE = _HERE.parent.parent
ROOT = _CANDIDATE if (_CANDIDATE / "results").exists() else _HERE.parents[2]
STATS = ROOT / "results" / "summary"
OUT = STATS / "extra_evidence"

METHODS = [
    "kmeans_gs", "kmeanspp", "kmeans", "kmeans_gs_g", "mbkmeans",
    "kmeans_par", "afkmc2", "birch", "gmm", "dbscan",
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


def q(frame, column, quantile):
    return frame.groupby(["dataset", "method"])[column].quantile(quantile).unstack()


def main():
    global STATS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", default=str(STATS))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    STATS = Path(args.stats)
    OUT = Path(args.out) if args.out else STATS / "extra_evidence"
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(STATS / "all_runs.csv")
    summary = pd.read_csv(STATS / "summary.csv")
    means = runs.groupby(["dataset", "method"]).ari.mean().unstack()
    datasets = sorted(means.index)

    # Run-distribution diagnostics. Quantiles are computed inside each dataset
    # and then averaged with equal dataset weight.
    q10 = q(runs, "ari", 0.10)
    q25 = q(runs, "ari", 0.25)
    q50 = q(runs, "ari", 0.50)
    q75 = q(runs, "ari", 0.75)
    q90 = q(runs, "ari", 0.90)
    distribution_rows = []
    for method in METHODS:
        distribution_rows.append({
            "method": method,
            "method_label": LABEL[method],
            "mean_q10_ari": q10[method].mean(),
            "mean_q25_ari": q25[method].mean(),
            "mean_median_ari": q50[method].mean(),
            "mean_q75_ari": q75[method].mean(),
            "mean_q90_ari": q90[method].mean(),
            "mean_iqr_ari": (q75[method] - q25[method]).mean(),
            "datasets_with_negative_mean": int((means[method] < 0).sum()),
            "runs_with_negative_ari": int((runs.loc[runs.method == method, "ari"] < 0).sum()),
            "negative_run_rate": float((runs.loc[runs.method == method, "ari"] < 0).mean()),
        })
    distribution = pd.DataFrame(distribution_rows)

    # Best-of-20 and worst-of-20: upper and lower attainable quality from the
    # same 20 seeds. This is a diagnostic of basin quality, not a deployable
    # single-run result.
    best = runs.groupby(["dataset", "method"]).ari.max().unstack()
    worst = runs.groupby(["dataset", "method"]).ari.min().unstack()
    best_rank = best.rank(axis=1, ascending=False, method="average")
    worst_rank = worst.rank(axis=1, ascending=False, method="average")
    best_rows = []
    for method in METHODS:
        best_rows.append({
            "method": method,
            "method_label": LABEL[method],
            "mean_best_of_20_ari": best[method].mean(),
            "mean_best_of_20_rank": best_rank[method].mean(),
            "best_of_20_first_count": int((best_rank[method] <= 1).sum()),
            "best_of_20_top3_count": int((best_rank[method] <= 3).sum()),
            "mean_worst_of_20_ari": worst[method].mean(),
            "mean_worst_of_20_rank": worst_rank[method].mean(),
            "worst_of_20_top3_count": int((worst_rank[method] <= 3).sum()),
        })
    best_of_n = pd.DataFrame(best_rows)

    # Regret against the per-dataset best method.
    oracle = means.max(axis=1)
    regret = oracle.to_frame("oracle_ari").join(
        means.apply(lambda row: oracle.loc[row.name] - row, axis=1)
    )
    regret_rows = []
    for method in METHODS:
        r = regret[method]
        regret_rows.append({
            "method": method,
            "method_label": LABEL[method],
            "mean_regret_to_oracle": r.mean(),
            "median_regret_to_oracle": r.median(),
            "datasets_within_0_005_of_oracle": int((r <= 0.005).sum()),
            "datasets_within_0_01_of_oracle": int((r <= 0.01).sum()),
            "regret_rank": float(r.mean()),
        })
    regret_table = pd.DataFrame(regret_rows)
    regret_table["regret_rank"] = regret_table["mean_regret_to_oracle"].rank(
        ascending=True, method="min"
    ).astype(int)

    # Quality-time Pareto frontier.
    method_time = summary.set_index("method")["time_geomean"]
    method_ari = means.mean()
    pareto_rows = []
    for method in METHODS:
        dominated = False
        dominators = []
        for other in METHODS:
            if other == method:
                continue
            at_least_as_good = (
                method_ari[other] >= method_ari[method]
                and method_time[other] <= method_time[method]
            )
            strictly_better = (
                method_ari[other] > method_ari[method]
                or method_time[other] < method_time[method]
            )
            if at_least_as_good and strictly_better:
                dominated = True
                dominators.append(LABEL[other])
        pareto_rows.append({
            "method": method,
            "method_label": LABEL[method],
            "mean_ari": method_ari[method],
            "geomean_time": method_time[method],
            "on_pareto_front": not dominated,
            "dominated_by": ", ".join(dominators),
        })
    pareto = pd.DataFrame(pareto_rows)

    # Empirical scaling of seeding and total cost. The exponents come from
    # log(time) = a + b log(N) + c log(K) + e log(d), fitted across datasets.
    detail = pd.read_csv(STATS / "detail.csv")
    scaling_rows = []
    for method in ["kmeans_gs", "kmeanspp", "kmeans", "kmeans_gs_g", "kmeans_par", "afkmc2"]:
        g = detail[detail.method == method].set_index("dataset")
        x = np.column_stack([
            np.ones(len(g)),
            np.log(g.N),
            np.log(g.K),
            np.log(g.d),
        ])
        result = {"method": method, "method_label": LABEL[method]}
        for prefix, column in [
            ("seeding", "Seeding time (s) mean"),
            ("total", "Total time (s) mean"),
        ]:
            y = np.log(g[column].clip(lower=1e-12).to_numpy())
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            prediction = x @ beta
            r2 = 1 - ((y - prediction) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            result[f"{prefix}_N_exponent"] = beta[1]
            result[f"{prefix}_K_exponent"] = beta[2]
            result[f"{prefix}_d_exponent"] = beta[3]
            result[f"{prefix}_R2"] = r2
        scaling_rows.append(result)
    scaling = pd.DataFrame(scaling_rows)

    # Paired dataset bootstrap for the difference in seeding-cost exponents.
    gs_detail = detail[detail.method == "kmeans_gs"].set_index("dataset")
    pp_detail = detail[detail.method == "kmeanspp"].set_index("dataset")
    shared = sorted(set(gs_detail.index) & set(pp_detail.index))
    x = np.column_stack([
        np.ones(len(shared)),
        np.log(gs_detail.loc[shared, "N"]),
        np.log(gs_detail.loc[shared, "K"]),
        np.log(gs_detail.loc[shared, "d"]),
    ])
    rng = np.random.default_rng(20261019)
    exponent_rows = []
    for cost_label, column in [
        ("Seeding time", "Seeding time (s) mean"),
        ("Total time", "Total time (s) mean"),
    ]:
        y_gs = np.log(gs_detail.loc[shared, column].clip(lower=1e-12).to_numpy())
        y_pp = np.log(pp_detail.loc[shared, column].clip(lower=1e-12).to_numpy())
        exponent_boot = []
        for _ in range(10000):
            idx = rng.integers(0, len(shared), len(shared))
            b_gs = np.linalg.lstsq(x[idx], y_gs[idx], rcond=None)[0]
            b_pp = np.linalg.lstsq(x[idx], y_pp[idx], rcond=None)[0]
            exponent_boot.append(b_gs[1:3] - b_pp[1:3])
        exponent_boot = np.asarray(exponent_boot)
        exponent_rows.append({
            "cost": cost_label,
            "comparison": "KMeans-GS minus k-means++",
            "N_exponent_difference": float(exponent_boot[:, 0].mean()),
            "N_exponent_ci_low": float(np.quantile(exponent_boot[:, 0], 0.025)),
            "N_exponent_ci_high": float(np.quantile(exponent_boot[:, 0], 0.975)),
            "K_exponent_difference": float(exponent_boot[:, 1].mean()),
            "K_exponent_ci_low": float(np.quantile(exponent_boot[:, 1], 0.025)),
            "K_exponent_ci_high": float(np.quantile(exponent_boot[:, 1], 0.975)),
            "bootstrap_reps": 10000,
        })
    exponent_test = pd.DataFrame(exponent_rows)

    # Large-workload subsets: absolute seeding time, end-to-end time, and the
    # fraction of total time spent seeding.
    subset_rows = []
    for label, mask in [
        ("N >= 10,000", detail.N >= 10000),
        ("K >= 20", detail.K >= 20),
        ("N*K >= 100,000", detail.N * detail.K >= 100000),
    ]:
        subset = detail[mask]
        group_datasets = sorted(subset.dataset.unique())
        metrics = {}
        for method in ["kmeans_gs", "kmeanspp"]:
            g = subset[subset.method == method]
            metrics[method] = {
                "ari": g["ARI mean"].mean(),
                "seed_geo": float(np.exp(np.log(g["Seeding time (s) mean"].clip(lower=1e-12)).mean())),
                "total_geo": float(np.exp(np.log(g["Total time (s) mean"].clip(lower=1e-12)).mean())),
            }
        subset_rows.append({
            "subset": label,
            "datasets": ", ".join(group_datasets),
            "n_datasets": len(group_datasets),
            "GS mean ARI": metrics["kmeans_gs"]["ari"],
            "k-means++ mean ARI": metrics["kmeanspp"]["ari"],
            "ARI difference": metrics["kmeans_gs"]["ari"] - metrics["kmeanspp"]["ari"],
            "GS seed time geo (s)": metrics["kmeans_gs"]["seed_geo"],
            "k-means++ seed time geo (s)": metrics["kmeanspp"]["seed_geo"],
            "seed time speedup": metrics["kmeanspp"]["seed_geo"] / metrics["kmeans_gs"]["seed_geo"],
            "GS total time geo (s)": metrics["kmeans_gs"]["total_geo"],
            "k-means++ total time geo (s)": metrics["kmeanspp"]["total_geo"],
            "total time speedup": metrics["kmeanspp"]["total_geo"] / metrics["kmeans_gs"]["total_geo"],
            "GS seed share of total (%)": 100 * metrics["kmeans_gs"]["seed_geo"] / metrics["kmeans_gs"]["total_geo"],
            "k-means++ seed share of total (%)": 100 * metrics["kmeanspp"]["seed_geo"] / metrics["kmeanspp"]["total_geo"],
        })
    subsets = pd.DataFrame(subset_rows)
    seed_ratio = pd.DataFrame({
        "dataset": shared,
        "N": gs_detail.loc[shared, "N"].to_numpy(),
        "K": gs_detail.loc[shared, "K"].to_numpy(),
        "d": gs_detail.loc[shared, "d"].to_numpy(),
        "kmeanspp_over_gs_seed_time": (
            pp_detail.loc[shared, "Seeding time (s) mean"]
            / gs_detail.loc[shared, "Seeding time (s) mean"]
        ).to_numpy(),
    }).sort_values("kmeanspp_over_gs_seed_time", ascending=False)

    # Composite external-metric rank: equal weight over ARI, AMI and FMI.
    rank_matrices = {}
    for metric in ["ari", "ami", "fmi"]:
        matrix = runs.groupby(["dataset", "method"])[metric].mean().unstack()
        rank_matrices[metric] = matrix.rank(axis=1, ascending=False, method="average")
    composite_rows = []
    for method in METHODS:
        composite_rows.append({
            "method": method,
            "method_label": LABEL[method],
            "mean_ari_rank": rank_matrices["ari"][method].mean(),
            "mean_ami_rank": rank_matrices["ami"][method].mean(),
            "mean_fmi_rank": rank_matrices["fmi"][method].mean(),
            "mean_external_composite_rank": np.mean([
                rank_matrices["ari"][method].mean(),
                rank_matrices["ami"][method].mean(),
                rank_matrices["fmi"][method].mean(),
            ]),
            "datasets_top3_ari_and_ami_and_fmi": int((
                (rank_matrices["ari"][method] <= 3)
                & (rank_matrices["ami"][method] <= 3)
                & (rank_matrices["fmi"][method] <= 3)
            ).sum()),
        })
    composite = pd.DataFrame(composite_rows).sort_values(
        "mean_external_composite_rank"
    )

    # Seed-matched paired comparison against k-means++.
    wide = runs.pivot_table(
        index=["dataset", "repeat_seed"], columns="method", values="ari",
        aggfunc="first",
    )
    diff = wide["kmeans_gs"] - wide["kmeanspp"]
    paired_rows = []
    for dataset in datasets:
        d = diff.loc[dataset]
        mean_d = float(d.mean())
        sd_d = float(d.std(ddof=1))
        wins = int((d > 1e-12).sum())
        ties = int((d.abs() <= 1e-12).sum())
        losses = int((d < -1e-12).sum())
        paired_rows.append({
            "dataset": dataset,
            "mean_difference_vs_kmeanspp": mean_d,
            "median_difference_vs_kmeanspp": float(d.median()),
            "sd_of_paired_differences": sd_d,
            "cohen_dz": mean_d / sd_d if sd_d > 0 else np.nan,
            "run_wins": wins,
            "run_ties": ties,
            "run_losses": losses,
            "run_win_rate_excluding_ties": wins / (wins + losses) if wins + losses else np.nan,
        })
    paired = pd.DataFrame(paired_rows)
    paired["seed_matched_positive"] = paired.mean_difference_vs_kmeanspp > 0

    # A practically meaningful tolerance: count datasets where GS is within
    # 0.005 ARI of k-means++ or better. This is a non-inferiority view, not a
    # superiority test.
    tolerance = 0.005
    gap = means["kmeans_gs"] - means["kmeanspp"]
    tolerance_table = pd.DataFrame({
        "criterion": [
            "Better than k-means++ by > 0.005 ARI",
            "Within +/-0.005 ARI of k-means++",
            "Worse than k-means++ by > 0.005 ARI",
            "Better than k-means++ by > 0.010 ARI",
            "Worse than k-means++ by > 0.010 ARI",
        ],
        "datasets": [
            int((gap > tolerance).sum()),
            int((gap.abs() <= tolerance).sum()),
            int((gap < -tolerance).sum()),
            int((gap > 0.01).sum()),
            int((gap < -0.01).sum()),
        ],
    })
    tolerance_table["datasets_named"] = [
        ", ".join(gap[gap > tolerance].index),
        ", ".join(gap[gap.abs() <= tolerance].index),
        ", ".join(gap[gap < -tolerance].index),
        ", ".join(gap[gap > 0.01].index),
        ", ".join(gap[gap < -0.01].index),
    ]

    # Hierarchical bootstrap over datasets for the mean ARI difference.
    rng = np.random.default_rng(20261019)
    ds_diff = gap.to_numpy(float)
    boot = rng.choice(ds_diff, size=(50000, len(ds_diff)), replace=True).mean(axis=1)
    hierarchical = pd.DataFrame([{
        "mean_dataset_difference": float(ds_diff.mean()),
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "probability_positive": float((boot > 0).mean()),
        "probability_above_0_005": float((boot > 0.005).mean()),
        "probability_below_minus_0_005": float((boot < -0.005).mean()),
        "dataset_wilcoxon_p": float(wilcoxon(ds_diff).pvalue),
    }])

    tables = {
        "run_distribution": distribution,
        "best_of_20": best_of_n,
        "regret": regret_table,
        "pareto": pareto,
        "scaling": scaling,
        "exponent_test": exponent_test,
        "large_workloads": subsets,
        "seed_time_ratio": seed_ratio,
        "composite_external": composite,
        "paired_by_dataset": paired,
        "tolerance": tolerance_table,
        "hierarchical_bootstrap": hierarchical,
        "dataset_mean_ari": means.rename(columns=LABEL).reset_index(),
        "best_of_20_matrix": best.rename(columns=LABEL).reset_index(),
        "regret_matrix": regret[methods_ := [m for m in METHODS if m in regret.columns]].rename(
            columns=LABEL
        ).reset_index(),
    }
    for name, frame in tables.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)

    print("RUN DISTRIBUTION")
    print(distribution.sort_values("mean_q10_ari", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print("\nBEST OF 20")
    print(best_of_n.sort_values("mean_best_of_20_ari", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print("\nREGRET")
    print(regret_table.sort_values("mean_regret_to_oracle").to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print("\nPARETO")
    print(pareto.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSEEDING SCALING")
    print(scaling.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nEXPONENT DIFFERENCE")
    print(exponent_test.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nLARGE WORKLOADS")
    print(subsets.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCOMPOSITE EXTERNAL")
    print(composite.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPAIRED BY DATASET")
    print(paired.sort_values("mean_difference_vs_kmeanspp", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print("\nTOLERANCE")
    print(tolerance_table.to_string(index=False))
    print("\nHIERARCHICAL")
    print(hierarchical.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
