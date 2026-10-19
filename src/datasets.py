# -*- coding: utf-8 -*-
"""Dataset registry for the KMeans-GS paper.

The benchmark contains 12 public real datasets and one controlled synthetic
stress test. Real datasets are downloaded by OpenML data_id and cached as
cache/<key>.npz.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "cache")

# 加载逻辑一旦改变就递增，旧缓存自动失效
LOADER_VERSION = 6

# --------------------------------------------------------------------------
# 数据集登记表
#   group      : uci / synth
#   kind       : openml / synth
#   K          : 聚类数；None 表示取真实类别数
#   K_paper    : 版本2 表 2 中写的 K（用于核对，与真实类别数不一致处已在说明中标注）
# --------------------------------------------------------------------------
DATASETS = {
    # ---------------- controlled synthetic stress tests ----------------
    "synth_imbalanced": dict(group="synth", kind="synth", gen="imbalanced",
                             n=15000, d=100, K=50,
                             desc="Zipf-sized Gaussian clusters; seed=2026"),

    # ---------------- UCI 真实数据集 ----------------
    "wine_quality": dict(group="uci", kind="openml", data_id=40691, onml="wine-quality-red",
                         n=1599, d=11, K=6),
    "wdbc": dict(group="uci", kind="openml", data_id=1510, onml="wdbc", n=569, d=30, K=2),
    "dermatology": dict(group="uci", kind="openml", data_id=35, onml="dermatology",
                        n=366, d=34, K=6),
    "synthetic_control": dict(group="uci", kind="openml", data_id=377, onml="synthetic_control",
                              n=600, d=60, K=6),
    "letter": dict(group="uci", kind="openml", data_id=6, onml="letter", n=20000, d=16,
                   K=26, K_paper=9,
                   note="版本2 表 2 写 K=9，实际类别数为 26，此处用真实类别数"),
    "musk": dict(group="uci", kind="openml", data_id=1116, onml="musk", n=6598, d=166, K=2),
    "isolet": dict(group="uci", kind="openml", data_id=300, onml="isolet", n=7797, d=617, K=26),
    "segment": dict(group="uci", kind="openml", data_id=36, onml="segment",
                    n=2310, d=19, K=7),
    "yeast": dict(group="uci", kind="openml", data_id=181, onml="yeast",
                  n=1484, d=8, K=10),
    "vehicle": dict(group="uci", kind="openml", data_id=54, onml="vehicle",
                    n=846, d=18, K=4),
    "pendigits": dict(group="uci", kind="openml", data_id=32, onml="pendigits",
                      n=10992, d=16, K=10),
    "satimage": dict(group="uci", kind="openml", data_id=182, onml="satimage",
                     n=6430, d=36, K=6),
}


def _gen_imbalanced(n, d, K, seed=2026):
    """Zipf-sized clusters with a minimum of 20 points per cluster."""
    rng = np.random.default_rng(seed)
    weights = rng.zipf(1.2, K).astype(np.float64)
    weights /= weights.sum()
    base = 20
    counts = base + np.floor(weights * (n - base * K)).astype(np.int64)
    counts[0] += n - counts.sum()
    centers = rng.normal(0.0, 1.5, size=(K, d))
    xs, ys = [], []
    for i, count in enumerate(counts):
        xs.append(centers[i] + rng.normal(0.0, 1.0, size=(int(count), d)))
        ys.append(np.full(int(count), i, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys)


_GENERATORS = {"imbalanced": _gen_imbalanced}


# --------------------------------------------------------------------------
# 真实数据集
# --------------------------------------------------------------------------
def _load_openml(spec, data_home):
    """下载（或读缓存）OpenML 数据集，返回 (X, y, dropped_columns)。"""
    try:
        bunch = fetch_openml(data_id=spec["data_id"], as_frame=True, parser="auto",
                             data_home=data_home)
    except Exception:
        bunch = fetch_openml(name=spec["onml"], version=1, as_frame=True,
                             parser="auto", data_home=data_home)
    X, dropped = _frame_to_matrix(bunch.data)
    if X.shape[1] == 0:
        raise ValueError("数据集 %s 不含可用特征" % spec.get("onml"))
    if np.any(~np.isfinite(X)):
        # 用列中位数填补缺失（逐列），保证下游距离计算可用
        col = np.nanmedian(X, axis=0)
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(col, idx[1])
    y = bunch.target
    y = LabelEncoder().fit_transform(np.asarray(y).astype(str))
    return X, y.astype(np.int64), dropped


def _frame_to_matrix(frame):
    """把 OpenML 返回的 DataFrame 转成数值矩阵，返回 (X, 被丢弃的列名)。

    不能简单地用 select_dtypes(include=[np.number])：ARFF 中的名义型特征
    （zoo 的 14 个二值特征、dermatology 的 0~3 有序编码等）会被整体丢掉，
    造成维度塌缩——旧代码里 zoo 与 dermatology 都被压成 1 维，
    而论文表 2 写的是 16 维与 34 维。

    规则：
      * 数值列（含布尔）直接使用（布尔映射为 0/1）；
      * 名义列的取值本身是数字编码（如 zoo 的 '0'/'1'、dermatology 的 '0'~'3'）
        时，按该数字取值映射后保留——这样维度与论文表 2 一致（zoo 16 维、
        dermatology 34 维）；
      * 名义列的取值不是数字（如 musk 的 molecule_name）时按标识符丢弃，
        并在返回的 dropped 中列出（musk 由此保持论文表 2 的 166 维）。
    """
    import pandas as pd

    cols = []
    dropped = []
    for name in frame.columns:
        col = frame[name]
        if pd.api.types.is_bool_dtype(col):
            # openml 会把 ARFF 的二值名义特征解析为 bool（如 zoo 的 15 个 0/1 特征）
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


_BOOL_WORDS = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0,
               "t": 1.0, "f": 0.0, "y": 1.0, "n": 0.0, "": 0.0}


def _value_codes(uniq):
    """把名义列的取值映射为数值编码：布尔词表优先，其次尝试直接转数字。

    返回 None 表示该列不是数值可编码的（视为标识符，应当丢弃）。
    """
    if uniq and all(u.lower() in _BOOL_WORDS for u in uniq):
        return [_BOOL_WORDS[u.lower()] for u in uniq]
    try:
        return [float(u) for u in uniq]
    except ValueError:
        return None


def load(key, data_home=None, use_cache=True):
    """加载单个数据集，返回 (X, y, meta)。

    X: (N, d) float64；y: (N,) int64 真实标签。
    meta: dict，含 key/N/K/d/group/desc/K_paper/note。
    """
    if key not in DATASETS:
        raise KeyError("未登记的数据集：%s" % key)
    spec = dict(DATASETS[key])
    data_home = data_home or os.path.join(HERE, "..", "data_cache")
    cache_dir = os.path.normpath(CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "%s.v%d.npz" % (key, LOADER_VERSION))

    if use_cache and os.path.exists(cache_path):
        with np.load(cache_path) as f:
            X, y = f["X"], f["y"]
            dropped = [s for s in str(f["dropped"]).split(",") if s]
    else:
        if spec["kind"] == "synth":
            X, y = _GENERATORS[spec["gen"]](spec["n"], spec["d"], spec["K"])
            dropped = []
        else:
            X, y, dropped = _load_openml(spec, data_home)
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if use_cache:
            np.savez_compressed(cache_path, X=X, y=y,
                                dropped=np.array(",".join(dropped)))

    meta = {
        "key": key,
        "group": spec["group"],
        "N": int(X.shape[0]),
        "d": int(X.shape[1]),
        "K": int(len(np.unique(y)) if spec.get("K") is None else spec["K"]),
        "K_true": int(len(np.unique(y))),
        "K_paper": spec.get("K_paper", spec.get("K")),
        "desc": spec.get("desc", spec.get("onml", "")),
        "note": spec.get("note", ""),
        "dropped_columns": dropped,
    }
    return X, y, meta


def keys(group="all"):
    if group == "all":
        return list(DATASETS.keys())
    return [k for k, v in DATASETS.items() if v["group"] == group]
