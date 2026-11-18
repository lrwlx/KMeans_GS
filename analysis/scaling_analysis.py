#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit empirical cost exponents from the controlled scaling study.

For each generating mechanism we fit, per method,

    log t = a + b_N log N + b_K log K + b_d log d

over all configurations of that mechanism, and separately fit one-dimensional
slopes within each sweep.  The one-dimensional slopes are the numbers quoted in
the paper because they are not affected by the (mild) collinearity between N,
K and d in the design.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

LABEL = {"kmeans": "K-means (random)", "kmeanspp": "k-means++",
         "kmeanspp_greedy": "k-means++ (greedy)",
         "kmeanspp_sklearn": "k-means++ (compiled)", "kmeans_par": "k-means||",
         "afkmc2": "AFK-MC2", "kmeans_gs": "KMeans-GS", "kmeans_gs_g": "KMeans-GS-G"}


def fit(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return b, 1.0 - ss_res / ss_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="results/scaling/scaling_raw.csv")
    p.add_argument("--out", default="results/scaling")
    args = p.parse_args()
    df = pd.read_csv(args.raw)
    df = df[df.status == "ok"].copy()
    rows = []
    for mech, g in df.groupby("mechanism"):
        # N sweep: fixed K, d; vary N
        for m, gm in g.groupby("method"):
            out = {"mechanism": mech, "method": m, "label": LABEL.get(m, m)}
            for sweep, var, fixed in [("N", "N", ["K", "d"]),
                                      ("K", "K", ["N", "d"]),
                                      ("d", "d", ["N", "K"])]:
                # select configurations that vary only in `var`
                grp = gm.copy()
                key = tuple(fixed)
                best = None
                for kk, gg in grp.groupby(list(key)):
                    nlev = gg[var].nunique()
                    if nlev >= 2 and (best is None or nlev > best[1]):
                        best = (kk, nlev, gg)
                if best is None:
                    out["%s_exp" % sweep] = np.nan
                    continue
                gg = best[2].groupby(var).seed_s.mean().reset_index()
                if len(gg) < 2:
                    out["%s_exp" % sweep] = np.nan
                    continue
                b, r2 = fit(np.column_stack([np.ones(len(gg)), np.log(gg[var])]),
                            np.log(gg.seed_s))
                out["%s_exp" % sweep] = float(b[1])
                out["%s_r2" % sweep] = float(r2)
            # multivariate fit over all points of the mechanism
            X = np.column_stack([np.ones(len(gm)), np.log(gm.N), np.log(gm.K),
                                 np.log(gm.d)])
            b, r2 = fit(X, np.log(gm.seed_s))
            out.update({"mv_N": float(b[1]), "mv_K": float(b[2]),
                        "mv_d": float(b[3]), "mv_r2": float(r2)})
            rows.append(out)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.out, "scaling_exponents.csv"), index=False)

    # summary ranges across mechanisms
    summ = per.groupby(["method", "label"]).agg(
        N_min=("N_exp", "min"), N_max=("N_exp", "max"),
        K_min=("K_exp", "min"), K_max=("K_exp", "max"),
        d_min=("d_exp", "min"), d_max=("d_exp", "max"),
        mv_r2_min=("mv_r2", "min")).reset_index()
    summ.to_csv(os.path.join(args.out, "scaling_summary.csv"), index=False)

    # ratio of each method's seeding time to KMeans-GS at each configuration
    piv = df.pivot_table(index=["mechanism", "N", "K", "d"], columns="method",
                         values="seed_s", aggfunc="mean")
    if "kmeans_gs" in piv:
        ratio = piv.div(piv["kmeans_gs"], axis=0)
        ratio.reset_index().to_csv(os.path.join(args.out, "scaling_ratio.csv"),
                                   index=False)
    print(summ.to_string(index=False, float_format=lambda v: "%.3f" % v))
    print("\nper-mechanism exponents\n", per.to_string(index=False,
                                                       float_format=lambda v: "%.3f" % v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
