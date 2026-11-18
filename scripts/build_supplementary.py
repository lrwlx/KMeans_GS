#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the supplementary workbook from the aggregated results."""

from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tierA", default="results/summary_tierA")
    p.add_argument("--raw", default="results/main_tierA")
    p.add_argument("--tierB", default="results/summary_tierB")
    p.add_argument("--apps", default="results/summary_applications")
    p.add_argument("--scaling", default="results/scaling")
    p.add_argument("--sensitivity", default="results/sensitivity")
    p.add_argument("--out", default="paper/Additional_file_1.xlsx")
    args = p.parse_args()

    sheets = {}
    def add(name, path, **kw):
        if os.path.exists(path):
            sheets[name] = pd.read_csv(path, **kw)

    add("Table1_datasets", os.path.join(args.tierA, "per_dataset.csv"))
    add("Table2_summary", os.path.join(args.tierA, "summary.csv"))
    add("Table3_pairwise", os.path.join(args.tierA, "pairwise.csv"))
    add("Friedman", os.path.join(args.tierA, "friedman.csv"))
    add("Nemenyi", os.path.join(args.tierA, "nemenyi.csv"))
    add("LOO", os.path.join(args.tierA, "loo.csv"))
    add("Subgroups", os.path.join(args.tierA, "subgroups.csv"))
    add("BestOfR", os.path.join(args.tierA, "best_of.csv"))
    add("Times", os.path.join(args.tierA, "times.csv"))
    add("SeedRatio", os.path.join(args.tierA, "seed_ratio.csv"))
    add("PerDataset", os.path.join(args.tierA, "per_dataset.csv"))

    for m in ("ari", "ami", "fmi", "mse", "silhouette", "davies_bouldin",
              "n_iter"):
        add("matrix_%s" % m, os.path.join(args.tierA, "matrix_%s_mean.csv" % m))

    if os.path.exists(os.path.join(args.tierB, "per_dataset.csv")):
        sheets["TierB"] = pd.read_csv(os.path.join(args.tierB, "per_dataset.csv"))
    if os.path.exists(os.path.join(args.apps, "application_summary.csv")):
        sheets["Applications"] = pd.read_csv(
            os.path.join(args.apps, "application_summary.csv"))
        sheets["ApplicationTests"] = pd.read_csv(
            os.path.join(args.apps, "application_tests.csv"))
    if os.path.exists(os.path.join(args.scaling, "scaling_exponents.csv")):
        sheets["ScalingExponents"] = pd.read_csv(
            os.path.join(args.scaling, "scaling_exponents.csv"))
        sheets["ScalingRaw"] = pd.read_csv(
            os.path.join(args.scaling, "scaling_raw.csv"))
    if os.path.exists(os.path.join(args.sensitivity, "sensitivity_raw.csv")):
        sheets["Sensitivity"] = pd.read_csv(
            os.path.join(args.sensitivity, "sensitivity_raw.csv"))

    # all tier-A run records, if present
    run_files = [os.path.join(args.raw, f)
                 for f in sorted(os.listdir(args.raw)) if f.endswith(".csv")
                 and not f.startswith("_")] if os.path.isdir(args.raw) else []
    if run_files:
        sheets["Runs"] = pd.concat([pd.read_csv(f) for f in run_files],
                                   ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print("wrote", args.out, "with", len(sheets), "sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
