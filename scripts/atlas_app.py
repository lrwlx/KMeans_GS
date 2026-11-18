#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-cell atlas benchmark with an explicit restart protocol.

Every method is run ``--restarts`` times with paired seeds.  All restarts are
written to the output file, together with the SSE-selected one, so that the
analysis can report either the restart-averaged result or the selected restart.
Timing is the cost of the whole restart protocol (sum over restarts), not the
selected restart alone.

The embedding is read from ``obsm/<pca-key>`` when present; otherwise it is
computed from ``X`` (HVG + PCA) and cached as ``.npy`` so that repeated runs
use the same representation.
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
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_score)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import seeding as S  # noqa: E402
from src.kmeans import lloyd  # noqa: E402
from src.methods import FULL_METHODS  # noqa: E402


def _read_csr(node):
    shape = tuple(int(x) for x in node.attrs["shape"]) if "shape" not in node \
        else tuple(int(x) for x in node["shape"][:])
    return sp.csr_matrix((np.asarray(node["data"]), np.asarray(node["indices"]),
                          np.asarray(node["indptr"])), shape=shape)


def _read_cat(obs, key):
    if key not in obs:
        return None
    node = obs[key]
    if hasattr(node, "keys") and "categories" in node.keys():
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in node["categories"][:]]
        codes = np.asarray(node["codes"])
        out = np.full(codes.shape[0], "Unknown", dtype=object)
        out[codes >= 0] = [cats[c] for c in codes[codes >= 0]]
        return out
    vals = node[:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals],
                    dtype=object)


def preprocess(X, n_hvg=2000, n_pcs=50, chunk=4000):
    X = X.astype(np.float32).tocsr()
    nnz = np.asarray((X > 0).sum(axis=0)).ravel()
    X = X[:, np.nonzero(nnz >= 3)[0]]
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    X = (sp.diags(1e4 / tot) @ X).tocsr()
    X.data = np.log1p(X.data)
    n_genes = X.shape[1]
    var = np.empty(n_genes, dtype=np.float64)
    for s in range(0, n_genes, chunk):
        e = min(s + chunk, n_genes)
        blk = X[:, s:e]
        m = np.asarray(blk.mean(axis=0)).ravel()
        sq = np.asarray(blk.multiply(blk).mean(axis=0)).ravel()
        var[s:e] = sq - m ** 2
    idx = np.sort(np.argsort(var)[-min(n_hvg, n_genes):])
    dense = np.asarray(X[:, idx].todense(), dtype=np.float32)
    del X
    pcs = PCA(n_components=int(n_pcs), svd_solver="randomized",
              random_state=0).fit_transform(dense)
    return np.ascontiguousarray(pcs, dtype=np.float64)


def load_atlas(path, pca_key, cache_pca, label_key):
    import h5py

    if cache_pca and os.path.exists(cache_pca):
        Z = np.load(cache_pca)
        with h5py.File(path, "r") as f:
            labels = _read_cat(f["obs"], label_key)
        return np.ascontiguousarray(Z, dtype=np.float64), labels, "cache"

    with h5py.File(path, "r") as f:
        labels = _read_cat(f["obs"], label_key)
        if pca_key and pca_key in f["obsm"]:
            Z = np.asarray(f["obsm"][pca_key][:], dtype=np.float64)
            return Z, labels, "obsm/%s" % pca_key
        X = _read_csr(f["X"])
    Z = preprocess(X)
    del X
    if cache_pca:
        os.makedirs(os.path.dirname(os.path.abspath(cache_pca)), exist_ok=True)
        np.save(cache_pca, Z)
    return Z, labels, "own PCA"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--pca-key", default="X_pca")
    p.add_argument("--cache-pca", default="")
    p.add_argument("--label-key", default="cell_type")
    p.add_argument("--K", type=int, default=0)
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--methods", default="kmeans,kmeanspp,kmeanspp_greedy,"
                                        "kmeanspp_sklearn,kmeans_gs,kmeans_gs_g,"
                                        "mbkmeans,sklearn1")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    name = args.name or os.path.basename(args.path).split(".")[0]
    t0 = time.perf_counter()
    Z, labels, source = load_atlas(args.path, args.pca_key, args.cache_pca,
                                   args.label_key)
    labels = np.asarray(pd.factorize(pd.Series(labels))[0], dtype=np.int64)
    print("loaded %s: %s via %s (%.1f s)" % (name, Z.shape, source,
                                             time.perf_counter() - t0), flush=True)
    K = args.K or len(set(labels.tolist()))

    rows = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        method_rows = []
        for r in range(args.restarts):
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
                d0 = float(res.d0)
            else:
                fres = FULL_METHODS[method]["fn"](Z, K, rng)
                t_seed = np.nan
                t_iter = np.nan
                lab, C, it = fres.labels, fres.centers, fres.n_iter
                trials, fallbacks, d0 = np.nan, 0, np.nan
            total = t_seed + t_iter if np.isfinite(t_seed) else \
                time.perf_counter() - t0
            mse = float(np.mean(np.sum((Z - C[lab]) ** 2, axis=1)))
            sil_idx = np.random.default_rng(0).choice(
                Z.shape[0], size=min(Z.shape[0], 5000), replace=False)
            row = {
                "dataset": name, "method": method, "restart": r, "K": K,
                "N": int(Z.shape[0]), "d": int(Z.shape[1]),
                "restarts": args.restarts, "tol": args.tol,
                "seed_s": t_seed, "iter_s": t_iter, "total_s": total,
                "n_iter": int(it), "mse": mse,
                "ari": adjusted_rand_score(labels, lab),
                "nmi": normalized_mutual_info_score(labels, lab),
                "ami": adjusted_mutual_info_score(labels, lab),
                "silhouette": float(silhouette_score(Z[sil_idx], lab[sil_idx])),
                "trials": trials, "fallbacks": fallbacks, "d0": d0,
            }
            method_rows.append(row)
            print("  %-18s r=%2d  seed %8.1f ms  iter %6.1f s  it=%3d  "
                  "MSE=%.4f  ARI=%.4f"
                  % (method, r, 1e3 * t_seed if np.isfinite(t_seed) else -1,
                     t_iter if np.isfinite(t_iter) else total, it, mse, row["ari"]),
                  flush=True)
        best = int(np.argmin([x["mse"] for x in method_rows]))
        for i, x in enumerate(method_rows):
            x["selected"] = (i == best)
        rows.extend(method_rows)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nwrote %s (%d rows)" % (args.out, len(df)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
