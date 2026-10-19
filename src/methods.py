# -*- coding: utf-8 -*-
"""完整算法型基线（不共享本文的 Lloyd 迭代）。

这些方法各有自己的初始化/迭代机制，无法拆成"播种 + 共享迭代"两段，因此单独登记：

* `dbscan`         ：DBSCAN（密度型，需要 eps 与 min_samples）
* `birch`          ：BIRCH（层次型，需要阈值）
* `gmm`            ：高斯混合模型（EM）

统一约定
--------
* 返回 FullResult：labels 用于有标签指标（保留 DBSCAN 的噪声标签 -1），
  labels_eval 用于无标签指标（噪声点指派到最近中心，保证各方法在同一 N 上比较）。
* centers 为该算法自己得到的簇中心（DBSCAN 取各簇非噪声成员均值）。
* 所有超参数一律写进 note，便于在论文表格里如实标注。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .seeding import DEFAULT_CHUNK, _sqdist_to_points


@dataclass
class FullResult:
    labels: np.ndarray
    centers: np.ndarray
    labels_eval: Optional[np.ndarray] = None
    n_iter: int = 0
    note: str = ""
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.labels_eval is None:
            self.labels_eval = self.labels


def _centers_and_eval_labels(X, labels, chunk=65536):
    """把（可能含 -1 噪声标签的）簇标签转成中心与"可用于距离指标的"标签。

    中心取各簇成员均值；噪声点（-1）不参与中心计算，但会按最近中心指派，
    以便 MSE / 轮廓系数 / DB 这类指标在所有方法上使用同一批样本。
    """
    labels = np.asarray(labels)
    uniq = [int(u) for u in np.unique(labels) if u >= 0]
    if not uniq:
        raise ValueError("算法没有产生任何簇")
    centers = np.empty((len(uniq), X.shape[1]), dtype=np.float64)
    for j, u in enumerate(uniq):
        centers[j] = X[labels == u].mean(axis=0)
    remap = {u: j for j, u in enumerate(uniq)}
    eval_labels = np.array([remap.get(int(v), -1) for v in labels], dtype=np.int64)
    noise = np.nonzero(eval_labels < 0)[0]
    if noise.size:
        for s in range(0, noise.size, chunk):
            idx = noise[s:s + chunk]
            Xb = X[idx]
            d2 = (np.einsum("ij,ij->i", Xb, Xb)[:, None]
                  - 2.0 * (Xb @ centers.T) + np.einsum("ij,ij->i", centers, centers)[None, :])
            eval_labels[idx] = np.argmin(d2, axis=1)
    return centers, eval_labels, len(uniq), int(noise.size)


def _kdist_curve(X, min_samples, sample, seed):
    """子样本上第 min_samples 近邻距离的升序曲线（用于确定 DBSCAN 的 eps 候选）。"""
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    m = min(n, int(sample))
    rng = np.random.default_rng(seed)
    sub = np.ascontiguousarray(X[rng.choice(n, size=m, replace=False)]) if m < n else X
    nn = NearestNeighbors(n_neighbors=min(int(min_samples) + 1, m)).fit(sub)
    d, _ = nn.kneighbors(sub)
    return sub, np.sort(d[:, -1])


def _knee(sorted_vals):
    """肘部（kneedle 简化版）：取到首尾连线垂距最大的点。"""
    v = np.asarray(sorted_vals, dtype=np.float64)
    n = v.size
    if n < 3:
        return float(v[-1])
    x = np.linspace(0.0, 1.0, n)
    y = (v - v.min()) / (v.max() - v.min() + 1e-12)
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return float(v[int(np.argmax(d))])


def choose_dbscan_eps(X, *, min_samples=5, sample=3000, n_grid=8, seed=42,
                      min_clusters=2, max_noise=0.6, target_clusters=None,
                      verbose=False, **kwargs):
    """网格搜索 DBSCAN 的 eps：在**全量数据**上评估，取轮廓系数最高且不退化的候选。

    DBSCAN 的 eps 必须人为给定。两个坑都要绕开：
    * 常见的 k-距离拐点法在高维标准化数据上会把 eps 推到曲线长尾，使整份数据被判成
      一个簇（簇数 < 2，ARI 直接塌到 0）；
    * 若在子样本上调参再拿到全量数据上用，密度差异会使簇被"连通合并"（N 越大越严重）。

    因此这里用子样本的 k-距离曲线生成候选 eps（含分位点、拐点及其 0.5/0.25 倍），
    但每个候选都在全量数据上真实跑一次 DBSCAN 并按轮廓系数（噪声点指派到最近中心）
    打分；优先选满足"簇数 ≥ min_clusters 且噪声比例 ≤ max_noise"的最高分候选，若都
    不满足，则退化为"簇数最多"的候选。全程不使用真实标签，每个数据集只选一次，
    30 次重复共用同一个 eps。

    同时给出第二套 eps（eps_k）：在与簇数有关的规则下选——在噪声比例可接受的候选里，
    取"簇数最接近 target_clusters"的那个。本文所有方法都已知簇数 K（BIRCH 用
    n_clusters=K、GMM 用 n_components=K），因此这条规则并非额外信息，只是把 K 用起来，
    作为 DBSCAN 的一个更强变体（如果网格里根本没有接近 K 的候选，就退回轮廓系数最优解）。

    返回 dict：eps / silhouette / n_clusters / noise_frac / eps_k / ... / n_candidates / grid。
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    sub, kdist = _kdist_curve(X, min_samples, sample, seed)
    knee = _knee(kdist)
    qs = np.linspace(0.02, 0.90, int(n_grid))
    cands = np.concatenate([np.quantile(kdist, qs), [knee, 0.5 * knee, 0.25 * knee]])
    cands = np.unique(cands[np.isfinite(cands) & (cands > 0)])
    n = X.shape[0]
    grid = []
    for eps in cands:
        lab = DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=1).fit_predict(X)
        try:
            _, ev, n_found, n_noise = _centers_and_eval_labels(X, lab)
        except ValueError:               # 全是噪声
            n_found, n_noise, ev = 0, int(n), None
        noise_frac = n_noise / float(n)
        sil = float("nan")
        if ev is not None and n_found >= 2:
            try:
                sil = float(silhouette_score(X, ev, sample_size=min(n, 5000), random_state=seed))
            except Exception:
                sil = float("nan")
        grid.append({"eps": float(eps), "n_clusters": int(n_found),
                     "noise_frac": float(noise_frac), "silhouette": sil})
        if verbose:
            print("    eps=%.4g -> 簇数=%-4d 噪声=%.2f sil=%.3f"
                  % (eps, n_found, noise_frac, sil), flush=True)

    ok = [g for g in grid if g["n_clusters"] >= int(min_clusters)
          and g["noise_frac"] <= float(max_noise) and np.isfinite(g["silhouette"])]
    if ok:
        best = max(ok, key=lambda g: g["silhouette"])
    else:
        multi = [g for g in grid if g["n_clusters"] >= int(min_clusters)]
        best = max(multi, key=lambda g: g["n_clusters"]) if multi else \
            (grid[-1] if grid else {"eps": float(knee), "n_clusters": 0,
                                    "noise_frac": float("nan"), "silhouette": float("nan")})
    best = dict(best)
    best["n_candidates"] = int(cands.size)
    best["grid"] = grid
    best["sample"] = int(sub.shape[0])

    # 第二套：簇数最接近 K（在噪声比例可接受的候选里）
    best_k = None
    if target_clusters:
        pool = [g for g in grid if int(min_clusters) <= g["n_clusters"]
                and g["noise_frac"] <= float(max_noise)]
        if pool:
            best_k = min(pool, key=lambda g: (abs(g["n_clusters"] - int(target_clusters)),
                                              -g["n_clusters"]))
    if best_k is None:
        best_k = dict(best)
    best["eps_k"] = float(best_k["eps"])
    best["n_clusters_k"] = int(best_k["n_clusters"])
    best["noise_frac_k"] = float(best_k["noise_frac"])
    best["silhouette_k"] = float(best_k.get("silhouette", float("nan")))
    return best


def run_dbscan(X, K, rng, *, min_samples=5, eps=None, knn_sample=2000,
               max_points=20000, eps_source="k-距离拐点", **kwargs):
    """DBSCAN：eps 由外部给定的网格搜索结果（推荐）或 k-距离拐点启发式确定。

    说明：eps 与 min_samples 是 DBSCAN 必须人为设定的两个超参数。推荐做法是先跑
    scripts/choose_dbscan_eps.py 在全数据集上按轮廓系数网格搜索 eps（不使用真实
    标签，且每个数据集只选一次），再由 --dbscan-eps-file 注入；若未提供则退化为
    "子样本 k-距离拐点"启发式。N 超过 max_points 的数据集直接跳过并记录原因
    （蛮力近邻在 N 很大时不可行）。
    """
    from sklearn.cluster import DBSCAN

    n = X.shape[0]
    if n > int(max_points):
        raise ValueError("N=%d 超过 DBSCAN 上限 %d（蛮力近邻不可行）" % (n, int(max_points)))
    if eps is None:
        _, kdist = _kdist_curve(X, min_samples, knn_sample, 0)
        eps = _knee(kdist)
    db = DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=1).fit(X)
    labels = db.labels_
    centers, eval_labels, n_found, n_noise = _centers_and_eval_labels(X, labels)
    return FullResult(labels=labels, centers=centers, labels_eval=eval_labels,
                      note="DBSCAN(eps=%.4g [%s], min_samples=%d)" % (float(eps), eps_source,
                                                                    int(min_samples)),
                      extra={"eps": float(eps), "n_clusters_found": int(n_found),
                             "noise_frac": n_noise / float(n)})


def _median_nn_dist(X, rng, sample=1000, chunk=DEFAULT_CHUNK):
    """子样本内"到最近邻距离"的中位数：一个与维度、尺度都无关的局部间距尺度。

    用于给 BIRCH 的 threshold 定标。原因是 threshold 是"子簇半径"的绝对阈值：
    在标准化后的高维数据上（例如 MNIST d=784，两点典型距离约 40），原来按
    "0.25 × 稳健半径"取的阈值只有 4.7，会让 CF 树退化成几乎一个点一个子簇，
    构建时间从秒级涨到十分钟以上。用最近邻距离的中位数当尺度，等于"把彼此距离
    小于典型局部间距的点合并"，在低维和高维下都能得到规模合理的 CF 树。
    """
    n = X.shape[0]
    m = int(min(n, max(2, sample)))
    if m >= n:
        sub = X
    else:
        sub = np.ascontiguousarray(X[rng.choice(n, size=m, replace=False)])
    d2 = _sqdist_to_points(sub, sub, chunk)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(axis=1))
    med = float(np.median(nn))
    if not (med > 0.0):
        # 数据里存在完全重复的样本时最近邻距离会是 0，中位数可能整体为 0
        # （例如 zoo：101 个样本里超过一半有重复特征向量），此时改用非零距离的中位数；
        # 若整个子样本都退化，则退回到全局平均间距的一个极小比例。
        nz = nn[nn > 0.0]
        med = float(np.median(nz)) if nz.size else 0.0
        if not (med > 0.0):
            finite = d2[np.isfinite(d2)]
            scale = float(np.sqrt(np.mean(finite))) if finite.size else 1.0
            med = 1e-3 * scale
    return float(max(med, 1e-12)), int(m)


def run_birch(X, K, rng, *, threshold=None, branching_factor=50, **kwargs):
    """BIRCH：n_clusters=K；threshold 默认取"子样本最近邻距离的中位数"。

    threshold 是 BIRCH 唯一必须人为给定、且对代价影响极大的超参数：它决定 CF 树的
    粒度。默认值用与维度无关的局部间距尺度（最近邻距离中位数，1000 点子样本上估计，
    不使用标签），在低维与高维数据上都能得到规模合理的 CF 树。
    """
    from sklearn.cluster import Birch

    if threshold is None:
        threshold, _ = _median_nn_dist(X, rng)
    br = Birch(n_clusters=int(K), threshold=float(threshold),
               branching_factor=int(branching_factor))
    labels = br.fit_predict(X)
    # 注意：BIRCH 的 labels_ 已重编号为 0..K-1，不能再拿它去索引 subcluster_centers_
    # （那是"子簇"的中心，数量远多于 K）。这里按簇成员均值取中心。
    centers, eval_labels, n_found, n_noise = _centers_and_eval_labels(X, labels)
    return FullResult(labels=labels, centers=centers, labels_eval=eval_labels,
                      note="BIRCH(n_clusters=%d, threshold=%.4g, branching_factor=%d)"
                           % (int(K), float(threshold), int(branching_factor)),
                      extra={"threshold": float(threshold), "n_clusters_found": int(n_found)})


def run_gmm(X, K, rng, *, covariance=None, max_iter=100, tol=1e-3, reg_covar=1e-6, **kwargs):
    """高斯混合模型（EM）：维度 >50 时用对角协方差，否则用全协方差。"""
    from sklearn.mixture import GaussianMixture

    n, d = X.shape
    cov = covariance or ("diag" if d > 50 else "full")
    gm = GaussianMixture(n_components=int(K), covariance_type=cov, max_iter=int(max_iter),
                         tol=float(tol), reg_covar=float(reg_covar), n_init=1,
                         init_params="k-means++", random_state=int(rng.integers(2 ** 31 - 1)))
    labels = gm.fit_predict(X)
    return FullResult(labels=labels, centers=np.asarray(gm.means_, dtype=np.float64),
                      n_iter=int(gm.n_iter_),
                      note="GMM(covariance_type=%s, max_iter=%d, init=kmeans++)" % (cov, int(max_iter)),
                      extra={"lower_bound": float(gm.lower_bound_)})


def run_minibatch_kmeans(X, K, rng, *, batch_size=None, restarts=3, max_iter=100,
                         tol=1e-3, init="k-means++", **kwargs):
    """MiniBatchKMeans：大规模场景的实际默认（小批量 + 增量更新中心）。

    batch_size 默认 max(1024, 20K)（sklearn 默认 1024，K 大时按比例放大），
    n_init=3、max_iter=100、tol=1e-3，均为 sklearn 文档中的常用配置。
    """
    from sklearn.cluster import MiniBatchKMeans

    bs = int(batch_size) if batch_size else max(1024, 20 * int(K))
    mb = MiniBatchKMeans(n_clusters=int(K), init=init, n_init=int(restarts),
                         batch_size=bs, max_iter=int(max_iter), tol=float(tol),
                         reassignment_ratio=0.01,
                         random_state=int(rng.integers(2 ** 31 - 1)))
    labels = mb.fit_predict(X)
    return FullResult(labels=labels, centers=np.asarray(mb.cluster_centers_, dtype=np.float64),
                      n_iter=int(mb.n_iter_),
                      note="MiniBatchKMeans(batch_size=%d, n_init=%d, max_iter=%d, tol=%g)"
                           % (bs, int(restarts), int(max_iter), float(tol)),
                      extra={"sse": float(mb.inertia_), "batch_size": bs})


FULL_METHODS = {
    "mbkmeans": dict(label="MiniBatchKMeans", fn=run_minibatch_kmeans),
    "dbscan": dict(label="DBSCAN", fn=run_dbscan),
    "birch": dict(label="BIRCH", fn=run_birch),
    "gmm": dict(label="GMM", fn=run_gmm),
}

FULL_LABELS = {k: v["label"] for k, v in FULL_METHODS.items()}
