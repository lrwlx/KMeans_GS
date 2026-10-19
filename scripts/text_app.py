#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main line 3: a different modality (sparse text) at 5.6e5 documents.

Pipeline: TF-IDF (100k features) -> TruncatedSVD (100) -> k-means variants.
Adds a sketched acceptance test: KMeans-GS is run on a 32-dimensional Gaussian random
projection, the selected indices are mapped back to the 100-dimensional space,
and Lloyd then runs in the full space.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.random_projection import GaussianRandomProjection

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from src import seeding as S  # noqa: E402
from src.kmeans import lloyd  # noqa: E402
from src.metrics import mse_to_centers  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="/home/hpclr/scdata/dbpedia_train.parquet")
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--n-features", type=int, default=100000)
    p.add_argument("--n-svd", type=int, default=100)
    p.add_argument("--sketch", type=int, default=32)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--K", type=int, default=0, help="0 means infer from labels")
    p.add_argument("--methods", default="kmeans,kmeanspp,kmeans_gs,kmeans_gs_g")
    p.add_argument("--include-sklearn-n10", action="store_true",
                   help="add an sklearn n_init=10 reference row (not in the paper table)")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", default="results/text_dbpedia.csv")
    args = p.parse_args()

    t0 = time.perf_counter()
    df = pd.read_parquet(args.path)
    print("parquet", df.shape, list(df.columns), "%.1f s" % (time.perf_counter() - t0), flush=True)
    if args.max_docs and len(df) > args.max_docs:
        df = df.sample(n=args.max_docs, random_state=0).reset_index(drop=True)
    y = df["label"].to_numpy()
    text = (df["title"].fillna("") + ". " + df["content"].fillna("")).to_numpy()
    print("docs", len(text), "classes", len(np.unique(y)), flush=True)

    t0 = time.perf_counter()
    tf = TfidfVectorizer(max_features=args.n_features, min_df=5, sublinear_tf=True,
                         stop_words="english", dtype=np.float32)
    Xs = tf.fit_transform(text)
    print("tf-idf", Xs.shape, "nnz", Xs.nnz, "%.1f s" % (time.perf_counter() - t0), flush=True)

    t0 = time.perf_counter()
    Z = TruncatedSVD(n_components=args.n_svd, random_state=0).fit_transform(Xs)
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    print("svd", Z.shape, "%.1f s" % (time.perf_counter() - t0), flush=True)
    del Xs

    t0 = time.perf_counter()
    Zs = GaussianRandomProjection(n_components=args.sketch, random_state=0).fit_transform(Z)
    Zs = np.ascontiguousarray(Zs, dtype=np.float64)
    print("sketch", Zs.shape, "%.1f s" % (time.perf_counter() - t0), flush=True)

    K = int(args.K or len(np.unique(y)))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in S.SEEDERS]
    if unknown:
        raise ValueError("Unknown seeding methods: %s" % ", ".join(unknown))
    rows = []
    for r in range(args.repeats):
        base = args.seed_base + r
        for m in methods:
            rng = np.random.default_rng(base)
            t0 = time.perf_counter()
            res = S.SEEDERS[m](Z, K, rng)
            t_seed = time.perf_counter() - t0
            t0 = time.perf_counter()
            lab, C, it = lloyd(Z, res.centers, max_iter=300, tol=1e-7)
            dt = time.perf_counter() - t0
            rows.append({"dataset": "dbpedia14", "repeat": r, "method": m,
                         "seed_s": t_seed, "total_s": t_seed + dt, "n_iter": it,
                         "ari": adjusted_rand_score(y, lab),
                         "nmi": adjusted_mutual_info_score(y, lab),
                         "mse": mse_to_centers(Z, lab, C),
                         "sil": float(silhouette_score(Z, lab, sample_size=10000, random_state=0))})
        # sketched acceptance test
        rng = np.random.default_rng(base)
        t0 = time.perf_counter()
        res = S.SEEDERS["kmeans_gs"](Zs, K, rng)
        idx = res.indices
        t_seed = time.perf_counter() - t0
        C0 = Z[idx]
        t0 = time.perf_counter()
        lab, C, it = lloyd(Z, C0, max_iter=300, tol=1e-7)
        dt = time.perf_counter() - t0
        rows.append({"dataset": "dbpedia14", "repeat": r, "method": "kmeans_gs_sketch32",
                     "seed_s": t_seed, "total_s": t_seed + dt, "n_iter": it,
                     "ari": adjusted_rand_score(y, lab),
                     "nmi": adjusted_mutual_info_score(y, lab),
                     "mse": mse_to_centers(Z, lab, C),
                     "sil": float(silhouette_score(Z, lab, sample_size=10000, random_state=0))})
        if args.include_sklearn_n10:
            t0 = time.perf_counter()
            km = KMeans(n_clusters=K, n_init=10, random_state=base).fit(Z)
            dt = time.perf_counter() - t0
            rows.append({"dataset": "dbpedia14", "repeat": r, "method": "sklearn_n10",
                         "seed_s": np.nan, "total_s": dt, "n_iter": km.n_iter_,
                         "ari": adjusted_rand_score(y, km.labels_),
                         "nmi": adjusted_mutual_info_score(y, km.labels_),
                         "mse": float(km.inertia_ / Z.shape[0]),
                         "sil": float(silhouette_score(Z, km.labels_, sample_size=10000,
                                                       random_state=0))})
        print("repeat", r, "done", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    g = out.groupby("method").agg(ari=("ari", "mean"), nmi=("nmi", "mean"),
                                  mse=("mse", "mean"), seed_s=("seed_s", "mean"),
                                  total_s=("total_s", "mean"), n_iter=("n_iter", "mean"))
    print(g.to_string(float_format=lambda v: "%.4f" % v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
