# -*- coding: utf-8 -*-
"""Dataset registry for the KMeans-GS v2 study.

Two tiers:

* ``tier == "A"``: the quality benchmark.  Every method (including DBSCAN and
  GMM) is feasible, so all methods are compared on the same runs.
* ``tier == "B"``: large tabular data used to test scaling of quality and
  cost.  DBSCAN is infeasible (O(N^2)); the other methods are run.

Datasets are downloaded from OpenML by id and cached as float32 ``.npz`` under
``<repo>/code/data_cache`` (override with ``KMEANSGS_DATA``).
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get(
    "KMEANSGS_DATA", os.path.normpath(os.path.join(HERE, "..", "data_cache")))
LOADER_VERSION = 1

# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
DATASETS = {
    # ---- tier A: small / medium UCI, all methods feasible ----------------
    "dermatology": dict(tier="A", kind="openml", data_id=35, K=6, group="uci"),
    "wine_quality": dict(tier="A", kind="openml", data_id=40691, K=6, group="uci"),
    "wdbc": dict(tier="A", kind="openml", data_id=1510, K=2, group="uci"),
    "synthetic_control": dict(tier="A", kind="openml", data_id=377, K=6, group="uci"),
    "segment": dict(tier="A", kind="openml", data_id=36, K=7, group="uci"),
    "yeast": dict(tier="A", kind="openml", data_id=181, K=10, group="uci"),
    "vehicle": dict(tier="A", kind="openml", data_id=54, K=4, group="uci"),
    "satimage": dict(tier="A", kind="openml", data_id=182, K=6, group="uci"),
    "pendigits": dict(tier="A", kind="openml", data_id=32, K=10, group="uci"),
    "letter": dict(tier="A", kind="openml", data_id=6, K=26, group="uci"),
    "musk": dict(tier="A", kind="openml", data_id=1116, K=2, group="uci"),
    "isolet": dict(tier="A", kind="openml", data_id=300, K=26, group="uci"),
    "har": dict(tier="A", kind="openml", data_id=1478, K=6, group="uci"),
    # ---- tier A: geometry controls ---------------------------------------
    "circles": dict(tier="A", kind="synth", gen="circles", K=2, group="geometry"),
    "moons": dict(tier="A", kind="synth", gen="moons", K=2, group="geometry"),
    "synth_imbalanced": dict(tier="A", kind="synth", gen="imbalanced",
                             K=50, group="synth"),
    # ---- tier B: large tabular -------------------------------------------
    "mnist": dict(tier="B", kind="openml", data_id=554, K=10, group="large"),
    "covertype": dict(tier="B", kind="openml", data_id=1596, K=7, group="large"),
}

TIER_A = [k for k, v in DATASETS.items() if v["tier"] == "A"]
TIER_B = [k for k, v in DATASETS.items() if v["tier"] == "B"]


# --------------------------------------------------------------------------
# synthetic generators (deterministic, documented in the paper)
# --------------------------------------------------------------------------
def _gen_circles():
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=3000, factor=0.5, noise=0.05, random_state=2026)
    return X, y


def _gen_moons():
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=3000, noise=0.05, random_state=2026)
    return X, y


def _gen_imbalanced(n=15000, d=100, K=50, seed=2026):
    """Zipf-sized Gaussian clusters with at least 20 points per cluster."""
    rng = np.random.default_rng(seed)
    w = rng.zipf(1.2, K).astype(np.float64)
    w /= w.sum()
    base = 20
    counts = base + np.floor(w * (n - base * K)).astype(np.int64)
    counts[0] += n - counts.sum()
    centers = rng.normal(0.0, 1.5, size=(K, d))
    xs, ys = [], []
    for i, c in enumerate(counts):
        xs.append(centers[i] + rng.normal(0.0, 1.0, size=(int(c), d)))
        ys.append(np.full(int(c), i, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys)


_GENERATORS = {"circles": _gen_circles, "moons": _gen_moons,
               "imbalanced": _gen_imbalanced}


# --------------------------------------------------------------------------
# OpenML loading
# --------------------------------------------------------------------------
def _value_codes(uniq):
    words = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "t": 1.0,
             "f": 0.0, "y": 1.0, "n": 1.0, "": 0.0}
    if uniq and all(str(u).lower() in words for u in uniq):
        return [words[str(u).lower()] for u in uniq]
    try:
        return [float(u) for u in uniq]
    except ValueError:
        return None


def _frame_to_matrix(frame):
    """Convert an OpenML frame to floats, keeping ordinally encoded nominal
    columns and dropping identifier columns (e.g. musk molecule names)."""
    import pandas as pd

    cols, dropped = [], []
    for name in frame.columns:
        col = frame[name]
        if pd.api.types.is_bool_dtype(col):
            cols.append(col.astype(np.float64))
            continue
        if pd.api.types.is_numeric_dtype(col):
            cols.append(pd.to_numeric(col, errors="coerce").astype(np.float64))
            continue
        vals = col.astype("object").astype(str)
        uniq = sorted(vals.dropna().unique().tolist())
        codes = _value_codes(uniq)
        if codes is None:
            dropped.append(name)
            continue
        mapping = {u: codes[i] for i, u in enumerate(uniq)}
        cols.append(vals.map(mapping).astype(np.float64))
    if not cols:
        return np.zeros((len(frame), 0), dtype=np.float64), dropped
    return pd.concat(cols, axis=1).to_numpy(dtype=np.float64), dropped


def _load_openml(spec, cache_root):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    big = spec.get("as_numpy", False)
    bunch = fetch_openml(data_id=spec["data_id"], as_frame=not big,
                         parser="auto", data_home=cache_root)
    if big:
        X = np.asarray(bunch.data, dtype=np.float64)
        dropped = []
    else:
        X, dropped = _frame_to_matrix(bunch.data)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("no usable features")
    if np.any(~np.isfinite(X)):
        col = np.nanmedian(X, axis=0)
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(col, idx[1])
    y = np.asarray(bunch.target)
    y = LabelEncoder().fit_transform(y.astype(str)).astype(np.int64)
    return X, y, dropped


def load(key, use_cache=True):
    """Return (X, y, meta) for a registered dataset key."""
    if key not in DATASETS:
        raise KeyError("unknown dataset: %r" % (key,))
    spec = dict(DATASETS[key])
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "%s.v%d.npz" % (key, LOADER_VERSION))

    if use_cache and os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as f:
            X = f["X"].astype(np.float64)
            y = f["y"].astype(np.int64)
            dropped = json.loads(str(f["meta"]))["dropped"]
    else:
        if spec["kind"] == "synth":
            X, y = _GENERATORS[spec["gen"]]()
            dropped = []
        else:
            X, y, dropped = _load_openml(spec, os.path.join(CACHE_DIR, "openml"))
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if use_cache:
            np.savez_compressed(cache, X=X.astype(np.float32), y=y,
                                meta=np.array(json.dumps({"dropped": dropped})))
    meta = {
        "key": key,
        "tier": spec["tier"],
        "group": spec["group"],
        "N": int(X.shape[0]),
        "d": int(X.shape[1]),
        "K": int(spec["K"]) if spec["K"] is not None else int(len(np.unique(y))),
        "K_true": int(len(np.unique(y))),
        "source": ("OpenML data_id=%s" % spec.get("data_id")) if spec["kind"] == "openml"
                  else ("synthetic:%s" % spec.get("gen")),
        "dropped_columns": dropped,
    }
    return X, y, meta
