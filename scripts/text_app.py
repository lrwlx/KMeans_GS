#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DBpedia-14 text benchmark.

TF-IDF (100k features, sublinear tf, English stop words) followed by a
100-dimensional truncated SVD; the SVD embedding is cached so that every
method and repetition sees exactly the same representation.
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_score)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import seeding as S  # noqa: E402
from src.kmeans import lloyd  # noqa: E402
from src.methods import FULL_METHODS  # noqa: E402


def load_text(path, cache, n_features=100000, n_components=100, max_docs=0):
    if cache and os.path.exists(cache):
        with np.load(cache, allow_pickle=True) as f:
            return f["Z"].astype(np.float64), f["y"].astype(np.int64)
    df = pd.read_parquet(path)
    if max_docs and len(df) > max_docs:
        df = df.sample(n=max_docs, random_state=0).reset_index(drop=True)
    y = pd.factorize(df["label"])[0].astype(np.int64)
    text = (df["title"].fillna("") + ". " + df["content"].fillna("")).to_numpy()
    tf = TfidfVectorizer(max_features=n_features, min_df=5, sublinear_tf=True,
                         stop_words="english", dtype=np.float32)
    Z = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(
        tf.fit_transform(text))
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez_compressed(cache, Z=Z.astype(np.float32), y=y)
    return Z, y


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--cache", default="")
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--K", type=int, default=0)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--methods", default="kmeans,kmeanspp,kmeanspp_greedy,"
                                        "kmeanspp_sklearn,kmeans_gs,kmeans_gs_g,"
                                        "mbkmeans,sklearn1")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    Z, y = load_text(args.path, args.cache, max_docs=args.max_docs)
    print("loaded %s in %.1f s; K=%d" % (Z.shape, time.perf_counter() - t0,
                                         len(np.unique(y))), flush=True)
    K = args.K or len(np.unique(y))

    rows = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        for r in range(args.repeats):
            rng = np.random.default_rng(42 + r)
            t0 = time.perf_counter()
            if method in S.SEEDERS:
                res = S.SEEDERS[method](Z, K, rng)
                t_seed = time.perf_counter() - t0
                t0 = time.perf_counter()
                lab, C, it = lloyd(Z, res.centers, max_iter=args.max_iter,
                                   tol=args.tol)
                t_iter = time.perf_counter() - t0
                trials = float(np.mean(res.trials)) if res.trials is not None else np.nan
                fallbacks = int(res.fallbacks)
            else:
                fres = FULL_METHODS[method]["fn"](Z, K, rng)
                t_seed, t_iter = np.nan, np.nan
                lab, C, it = fres.labels, fres.centers, fres.n_iter
                trials, fallbacks = np.nan, 0
            total = t_seed + t_iter if np.isfinite(t_seed) else \
                time.perf_counter() - t0
            mse = float(np.mean(np.sum((Z - C[lab]) ** 2, axis=1)))
            idx = np.random.default_rng(0).choice(Z.shape[0],
                                                  size=min(Z.shape[0], 5000),
                                                  replace=False)
            rows.append({
                "dataset": "dbpedia14", "method": method, "repeat": r, "K": K,
                "N": int(Z.shape[0]), "d": int(Z.shape[1]), "tol": args.tol,
                "seed_s": t_seed, "iter_s": t_iter, "total_s": total,
                "n_iter": int(it), "mse": mse,
                "ari": adjusted_rand_score(y, lab),
                "nmi": normalized_mutual_info_score(y, lab),
                "ami": adjusted_mutual_info_score(y, lab),
                "silhouette": float(silhouette_score(Z[idx], lab[idx])),
                "trials": trials, "fallbacks": fallbacks,
            })
            print("  %-18s r=%2d  seed %7.1f ms  iter %5.1f s  it=%3d  ARI=%.4f"
                  % (method, r, 1e3 * t_seed if np.isfinite(t_seed) else -1,
                     t_iter if np.isfinite(t_iter) else total, it, rows[-1]["ari"]),
                  flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nwrote %s (%d rows)" % (args.out, len(df)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
