# -*- coding: utf-8 -*-
"""初始中心（播种）方法：random / k-means++ / KMeans-GS。

三种方法共享同一个 Lloyd 迭代实现（见 kmeans.py），区别只在"初始中心怎么选"，
因此计时差异完全来自播种阶段本身。

约定
----
* X 形状为 (N, d)，已按列标准化（见 datasets.py）。
* rng 为 np.random.Generator。同一 (数据集, 重复序号) 下三种方法使用同一种子，
  从而构成配对比较（便于后续做配对 t 检验）。
* 每个方法返回 SeedResult；KMeans-GS 的诊断量（每轮试探次数、兜底次数、阈值序列）
  一并带回，供论文报告 r 与兜底触发率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_CHUNK = 65536


@dataclass
class SeedResult:
    centers: np.ndarray
    method: str = ""
    trials: Optional[np.ndarray] = None    # KMeans-GS：每轮试探次数，长度 K-1
    fallbacks: int = 0                     # KMeans-GS：触发兜底的轮数
    taus: Optional[np.ndarray] = None      # KMeans-GS：每轮阈值
    d0: float = float("nan")               # KMeans-GS：初始阈值
    iqr_n_used: int = 0                    # KMeans-GS：估计四分位距实际使用的样本数
    indices: Optional[np.ndarray] = None   # 初始中心在 X 中的行号（用于分析"一簇一种子"）
    first_mode: str = ""                   # 首点策略（random/median/median_sub/lds）


def _sqdist_to_point(X, c, chunk=DEFAULT_CHUNK):
    """X 中每个点到 c 的平方欧氏距离（分块，避免 N x d 中间数组）。"""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = X[s:e] - c
        out[s:e] = np.einsum("ij,ij->i", diff, diff)
    return out


def _sqdist_to_points(X, P, chunk=DEFAULT_CHUNK):
    """X (N,d) 到 P (m,d) 的平方距离矩阵 (N,m)，分块计算。"""
    n = X.shape[0]
    m = P.shape[0]
    out = np.empty((n, m), dtype=np.float64)
    pn = np.einsum("ij,ij->i", P, P)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xb = X[s:e]
        blk = np.einsum("ij,ij->i", Xb, Xb)[:, None] - 2.0 * (Xb @ P.T) + pn[None, :]
        np.maximum(blk, 0.0, out=blk)
        out[s:e] = blk
    return out


def seed_random(X, K, rng, **kwargs):
    """K-means 的随机初始化：无放回随机抽取 K 个样本点。"""
    n = X.shape[0]
    if K > n:
        raise ValueError("K 不能大于样本数")
    idx = rng.choice(n, size=K, replace=False)
    return SeedResult(centers=np.array(X[idx], dtype=np.float64, copy=True), method="random",
                      indices=np.asarray(idx))


def seed_kmeanspp(X, K, rng, *, impl="optimized", chunk=DEFAULT_CHUNK, **kwargs):
    """k-means++：以 D^2 加权的概率抽样（Arthur & Vassilvitskii, 2007）。

    两种实现，用于回避"对比不公平"的质疑：
    * impl="optimized"（默认，最强对手）：维护每个点到已选中心的最近距离平方，
      每轮只与最新中心求一次距离，总代价 O(K N d)。
    * impl="classic"：每轮重新计算到全部已选中心的距离，总代价 O(K^2 N d)，
      即论文复杂度表中列出的那一档（向量化实现，不含 Python 逐样本循环）。
    """
    n, d = X.shape
    if K > n:
        raise ValueError("K 不能大于样本数")
    centers = np.empty((K, d), dtype=np.float64)
    centers[0] = X[rng.integers(n)]
    closest = _sqdist_to_point(X, centers[0], chunk)
    for k in range(1, K):
        total = float(closest.sum())
        if not np.isfinite(total) or total <= 0.0:
            centers[k] = X[rng.integers(n)]
        else:
            cum = np.cumsum(closest)
            r = rng.random() * float(cum[-1])
            idx = int(np.searchsorted(cum, r, side="left"))
            centers[k] = X[min(idx, n - 1)]
        if impl == "classic":
            closest = _sqdist_to_points(X, centers[: k + 1], chunk).min(axis=1)
        else:
            d2 = _sqdist_to_point(X, centers[k], chunk)
            np.minimum(closest, d2, out=closest)
    return SeedResult(centers=centers, method="kmeans++")


def seed_kmeans_gs(X, K, rng, *, d0_mode="paper", d0_coef=0.5, decay="inv_sqrt",
            decay_p=0.5, first="random", max_trials=100, edge_check=False,
            edge_rel=0.5, iqr_sample=2000, first_sample=2000, **kwargs):
    """KMeans-GS：基于几何阈值和拒绝采样的初始中心选择。

    每轮从 X 中独立均匀抽取候选点，只有"到所有已选中心的最近距离 > 当前阈值"
    才接受；阈值由各维四分位距自适应确定并按反平方根衰减。

    参数
    ----
    d0_mode : "paper" -> d0 = 0.5*sqrt(sum_k IQR_k^2)            （正文版本）
              "kdep"  -> d0 = sqrt(sum_k IQR_k^2)/(2+ln K)       （旧代码版本）
    decay   : "inv_sqrt" -> tau_i = d0/sqrt(i)                   （正文版本）
              "log"      -> tau_i = d0/(3+ln i)                  （旧代码版本）
    first   : "random" 随机取第一个中心；"median" 取离各维中位数最近的样本点。
    max_trials : 每轮试探上限 r；超限则取 r 个候选中"最小距离最大"的一个（兜底）。
                 兜底会破坏确定性分离保证，统计量在 SeedResult.fallbacks 中返回。
    edge_check : 是否附加"到 IQR 稳健盒边界距离 > edge_rel * d0"的接受条件。
    iqr_sample : 估计各维四分位距所用的样本数上限（默认 2000）。IQR 只用于确定
                 数据尺度，用固定规模的随机子样本估计即可：代价 O(iqr_sample * d)
                 而不是 O(N * d)，否则"播种代价与 N 无关"这条结论不成立。
                 2000 点的 IQR 估计相对误差在 3% 以内（见 scripts/iqr_sensitivity.py），
                 而 d0 只要量级正确即可，故该取值对聚类结果几乎无影响。
                 设为 0 或负数则使用全量数据（旧行为，代价随 N 线性增长）。
    """
    n, d = X.shape
    if K > n:
        raise ValueError("K 不能大于样本数")
    if iqr_sample and n > iqr_sample:
        # 有放回抽样：代价 O(iqr_sample)，与 N 无关（无放回会退化成 O(N) 的置换）
        sub = X[rng.integers(0, n, size=int(iqr_sample))]
    else:
        sub = X
    iqr_n_used = int(sub.shape[0])
    q1 = np.percentile(sub, 25, axis=0)
    q3 = np.percentile(sub, 75, axis=0)
    iqr = q3 - q1
    base = float(np.sqrt(np.sum(iqr ** 2)))
    if d0_mode == "paper":
        d0 = float(d0_coef) * base
    elif d0_mode == "kdep":
        d0 = base / (2.0 + np.log(max(K, 2)))
    else:
        raise ValueError("未知的 d0_mode: %s" % d0_mode)
    if not np.isfinite(d0) or d0 <= 0.0:
        raise ValueError("初始阈值 d0 非正（数据可能退化为常数维）")

    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    edge_tau = float(edge_rel * d0)

    centers = np.empty((K, d), dtype=np.float64)
    idx = np.empty(K, dtype=np.int64)
    if first == "median":
        med = np.median(X, axis=0)
        idx[0] = int(np.argmin(_sqdist_to_point(X, med)))
        centers[0] = X[idx[0]]
    elif first in ("median_sub", "lds"):
        # 与 N 无关的首点策略：只在 IQR 子样本里找（代价 O(m·d)）
        m = int(min(n, max(2, first_sample)))
        pick = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
        sub = np.ascontiguousarray(X[pick])
        if first == "median_sub":
            target = np.median(sub, axis=0)
        else:
            from scipy.stats import qmc
            u = qmc.Halton(d=d, scramble=False).random(1)[0]
            target = q1 + u * np.where(iqr > 0, iqr, 1e-12)
        j = int(np.argmin(_sqdist_to_point(sub, target)))
        idx[0] = int(pick[j])
        centers[0] = X[idx[0]]
    elif first == "random":
        idx[0] = int(rng.integers(n))
        centers[0] = X[idx[0]]
    else:
        raise ValueError("未知的 first: %s" % first)

    trials = np.zeros(max(K - 1, 0), dtype=np.int64)
    taus = np.zeros(max(K - 1, 0), dtype=np.float64)
    fallbacks = 0
    for k in range(1, K):
        i = k + 1
        if decay == "inv_sqrt":
            tau = d0 / np.sqrt(i)
        elif decay == "power":
            tau = d0 / (i ** float(decay_p))
        elif decay == "log":
            tau = d0 / (3.0 + np.log(i))
        else:
            raise ValueError("未知的 decay: %s" % decay)
        taus[k - 1] = tau
        best = None
        best_idx = -1
        best_min = -1.0
        t = 0
        accepted = False
        for t in range(1, max_trials + 1):
            ci = int(rng.integers(n))
            cand = X[ci]
            diff = centers[:k] - cand
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            dmin = float(dist.min())
            if dmin > best_min:
                best_min = dmin
                best = cand
                best_idx = ci
            if dmin > tau:
                if not edge_check:
                    accepted = True
                    break
                if float(np.min(np.minimum(cand - lo, hi - cand))) > edge_tau:
                    accepted = True
                    break
        if not accepted:
            fallbacks += 1
        trials[k - 1] = t
        centers[k] = best
        idx[k] = best_idx if best_idx >= 0 else int(rng.integers(n))
    return SeedResult(centers=centers, method="kmeans_gs", trials=trials,
                      fallbacks=fallbacks, taus=taus, d0=d0, iqr_n_used=iqr_n_used,
                      indices=idx, first_mode=first)


def seed_kmeans_parallel(X, K, rng, *, oversample=None, rounds=5,
                         chunk=DEFAULT_CHUNK, **kwargs):
    """k-means||（Bahmani et al., 2012）：ℓ 倍过采样 + 候选池上重新聚类。

    每一轮按概率 min(1, ℓ·d²(x,C)/φ_X(C)) 独立抽样（期望新增 ℓ 个候选），
    原文轮数为 O(log ψ)，常用实现（如 Spark 的 initializationSteps）取 5 轮。
    最后在候选池上跑一次 k-means++，得到 K 个初始中心——这正是 k-means|| 的
    "recluster" 步骤，也是它需要过采样因子 ℓ 的代价所在。

    参数
    ----
    oversample : ℓ，默认 2K（文献常用）。ℓ 越大所需采样轮数越少但候选池越大。
    rounds     : 采样轮数，默认 5。
    """
    n, d = X.shape
    if K > n:
        raise ValueError("K 不能大于样本数")
    ell = int(oversample) if oversample else max(2 * K, K)
    centers0 = X[rng.integers(n)]
    psi = _sqdist_to_point(X, centers0, chunk)
    picks = []
    for _ in range(max(1, int(rounds))):
        total = float(psi.sum())
        if not np.isfinite(total) or total <= 0.0:
            break
        p = np.clip(ell * psi / total, 0.0, 1.0)
        sel = np.nonzero(rng.random(n) < p)[0]
        if sel.size == 0:
            continue
        picks.append(sel)
        np.minimum(psi, _sqdist_to_points(X, X[sel], chunk).min(axis=1), out=psi)
    if picks:
        pool = np.concatenate(picks)
        # 候选池里同一候选出现多次等价于加权 k-means++（权重=出现次数）
        sub = seed_kmeanspp(np.ascontiguousarray(X[pool]), K, rng,
                            impl="optimized", chunk=chunk)
        out = SeedResult(centers=sub.centers, method="kmeans_par")
    else:
        out = seed_kmeanspp(X, K, rng, impl="optimized", chunk=chunk)
        out.method = "kmeans_par"
    out.trials = np.array([int(sum(len(s) for s in picks))], dtype=np.int64)
    return out


def seed_afkmc2(X, K, rng, *, chain=200, max_trials=100, chunk=DEFAULT_CHUNK,
                accept_slack=2.0, **kwargs):
    """AFK-MC²（Bachem et al., NIPS 2016）：用 MCMC 链估计正常数，再做均匀采样+拒绝。

    与 k-means++ 相比省掉两次 O(N·d) 的全量扫描：正常数 ψ ≈ (1/m)Σ_j d²(z_j,C)
    由 m 步 Metropolis-Hastings 链（目标分布 q(x) ∝ d²(x,C)，提议分布为数据上的
    均匀分布）估计得到，之后每轮只需"均匀抽一个点 + 以概率 d²(x,C)/(2ψ) 接受"，
    因此每轮代价是 O(i·d)，与 N 无关。代价是 ψ 只是估计值，且依赖链的收敛：
    m 太小则 ψ 偏差大、接受率失准（本实现用原文的 2ψ 松弛因子）。

    参数
    ----
    chain        : 链长 m，默认 200。
    max_trials   : 每轮试探上限（与 KMeans-GS 一致，记录兜底次数）。
    accept_slack : 接受概率分母的松弛系数，原文取 2。
    """
    n, d = X.shape
    if K > n:
        raise ValueError("K 不能大于样本数")
    centers = np.empty((K, d), dtype=np.float64)
    centers[0] = X[rng.integers(n)]

    cur = X[rng.integers(n)]
    d2_cur = float(np.dot(cur - centers[0], cur - centers[0]))
    total = 0.0
    for _ in range(max(1, int(chain))):
        prop = X[rng.integers(n)]
        d2_prop = float(np.dot(prop - centers[0], prop - centers[0]))
        if d2_cur <= 0.0 or rng.random() < min(1.0, d2_prop / d2_cur):
            cur, d2_cur = prop, d2_prop
        total += d2_cur
    psi = total / max(1, int(chain))
    if not np.isfinite(psi) or psi <= 0.0:
        raise ValueError("AFK-MC² 的正常数估计非正（数据可能退化为常数维）")

    trials = np.zeros(max(K - 1, 0), dtype=np.int64)
    fallbacks = 0
    for k in range(1, K):
        best, best_min2, t, accepted = None, -1.0, 0, False
        for t in range(1, int(max_trials) + 1):
            cand = X[rng.integers(n)]
            diff = centers[:k] - cand
            dmin2 = float(np.einsum("ij,ij->i", diff, diff).min())
            if dmin2 > best_min2:
                best_min2, best = dmin2, cand
            if rng.random() < dmin2 / (accept_slack * psi):
                accepted = True
                break
        trials[k - 1] = t
        if not accepted:
            fallbacks += 1
        centers[k] = best
    return SeedResult(centers=centers, method="afkmc2", trials=trials,
                      fallbacks=fallbacks, d0=float(psi), iqr_n_used=int(chain))


def seed_kmeans_gs_adapt(X, K, rng, *, d0_mode="paper", d0_coef=0.5, batch=8,
                  target_accept=0.6, iqr_sample=2000, max_batches=100,
                  **kwargs):
    """KMeans-GS 的自适应变体（A1+A2）：批量候选 + 阈值自校准 + 确定性收尾。

    与 seed_kmeans_gs 的差别只在"每轮怎么选一个中心"：
      1. 一次抽 b 个候选（而不是逐个个抽），一次性算出它们到已选中心的最近距离；
      2. 用这 b 个距离的 (1−p₀) 分位数当本轮阈值 τᵢ —— 阈值由当前几何自动校准，
         不再依赖 d₀/√i 这条手工调度；
      3. 第一个满足 dmin > τᵢ 的候选被接受；若整批都不满足，取批内 dmin 最大者
         （确定性收尾，而不是"随机试满 r 次再取最大"）。

    参数：batch=b，target_accept=p₀（目标接受比例，默认 0.6）。
    代价：每轮 O(b·i·d)，仍是 N 无关；b=8 时比逐点版本多算 8 倍距离，但少了
    Python 层的试探循环。
    """
    n, d = X.shape
    if K > n:
        raise ValueError("K 不能大于样本数")
    if iqr_sample and n > iqr_sample:
        sub = X[rng.integers(0, n, size=int(iqr_sample))]
    else:
        sub = X
    q1 = np.percentile(sub, 25, axis=0)
    q3 = np.percentile(sub, 75, axis=0)
    iqr = q3 - q1
    base = float(np.sqrt(np.sum(iqr ** 2)))
    d0 = (float(d0_coef) * base) if d0_mode == "paper" else base / (2.0 + np.log(max(K, 2)))
    if not np.isfinite(d0) or d0 <= 0:
        raise ValueError("初始尺度 d0 非正")
    tau_floor = 0.05 * d0

    centers = np.empty((K, d), dtype=np.float64)
    idx = np.empty(K, dtype=np.int64)
    idx[0] = int(rng.integers(n))
    centers[0] = X[idx[0]]

    trials = np.zeros(max(K - 1, 0), dtype=np.int64)
    taus = np.zeros(max(K - 1, 0), dtype=np.float64)
    fallbacks = 0
    b = int(max(2, batch))
    for k in range(1, K):
        i = k + 1
        best_pick, best_dmin = -1, -1.0
        for t in range(1, int(max_batches) + 1):
            cand_idx = rng.integers(0, n, size=b)
            diff = X[cand_idx][:, None, :] - centers[None, :k, :]
            dmin = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).min(axis=1)
            tau = max(float(np.quantile(dmin, 1.0 - float(target_accept))), tau_floor)
            ok = np.nonzero(dmin > tau)[0]
            if ok.size:
                best_pick = int(cand_idx[ok[0]])
                taus[k - 1] = tau
                trials[k - 1] = t
                break
            j = int(np.argmax(dmin))
            if float(dmin[j]) > best_dmin:
                best_dmin, best_pick = float(dmin[j]), int(cand_idx[j])
            taus[k - 1] = tau
            trials[k - 1] = t
        else:
            fallbacks += 1
        idx[k] = best_pick
        centers[k] = X[best_pick]
    return SeedResult(centers=centers, method="kmeans_gs_adapt", trials=trials,
                      fallbacks=fallbacks, taus=taus, d0=d0,
                      iqr_n_used=int(sub.shape[0]), indices=idx, first_mode="random")


def seed_kmeans_gs_select(X, K, rng, *, d0_mode="paper", d0_coef=0.5, batch=8,
                   max_batches=100, iqr_sample=2000, rank=1,
                   density_q=None, density_sample=1000, trim=0.0,
                   select="max", **kwargs):
    """KMeans-GS with greedy or weighted selection among admissible candidates.

    与 seed_kmeans_gs 唯一的差别：每轮一次抽 b 个候选，**保留原来的 τᵢ=d₀/√i**，
    在"通过阈值"的候选里取 min-distance 最大的那个（而不是第一个通过的）。
    整批都不过阈值时取批内 max-min 者（确定性收尾）。

    sklearn 的 k-means++ 用 n_local_trials=2+logK 个候选取最优，质量明显高于
    vanilla k-means++；这个变体是把同样的"贪心"加到 KMeans-GS 上，用于公平比较。
    代价：每轮 O(b·i·d)，仍与 N 无关。
    """
    n, d = X.shape
    if K > n:
        raise ValueError("K 不能大于样本数")
    if iqr_sample and n > iqr_sample:
        sub = X[rng.integers(0, n, size=int(iqr_sample))]
    else:
        sub = X
    q1 = np.percentile(sub, 25, axis=0)
    q3 = np.percentile(sub, 75, axis=0)
    iqr = q3 - q1
    base = float(np.sqrt(np.sum(iqr ** 2)))
    d0 = (float(d0_coef) * base) if d0_mode == "paper" else base / (2.0 + np.log(max(K, 2)))
    if not np.isfinite(d0) or d0 <= 0:
        raise ValueError("初始尺度 d0 非正")

    centers = np.empty((K, d), dtype=np.float64)
    idx = np.empty(K, dtype=np.int64)
    idx[0] = int(rng.integers(n))
    centers[0] = X[idx[0]]

    trials = np.zeros(max(K - 1, 0), dtype=np.int64)
    taus = np.zeros(max(K - 1, 0), dtype=np.float64)
    fallbacks = 0
    b = int(max(1, batch))
    rank = int(max(1, rank))
    dens_thr = None
    ref = None
    if density_q is not None:
        m_ref = int(min(n, max(50, density_sample)))
        ref = np.ascontiguousarray(X[rng.integers(0, n, size=m_ref)])
        d2 = _sqdist_to_points(ref, ref)
        np.fill_diagonal(d2, np.inf)
        dens_thr = float(np.quantile(np.sqrt(d2.min(axis=1)), float(density_q)))
    for k in range(1, K):
        i = k + 1
        tau = d0 / np.sqrt(i)
        taus[k - 1] = tau
        pick, t = -1, 0
        for t in range(1, int(max_batches) + 1):
            cand = rng.integers(0, n, size=b)
            diff = X[cand][:, None, :] - centers[None, :k, :]
            dmin = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).min(axis=1)
            ok = np.nonzero(dmin > tau)[0]
            if ok.size and dens_thr is not None:
                # drop candidates that sit in a locally sparse region (outliers)
                dref = np.sqrt(_sqdist_to_points(X[cand[ok]], ref).min(axis=1))
                keep = ok[dref <= dens_thr]
                if keep.size:
                    ok = keep
            if ok.size >= 2 and float(trim) > 0.0:
                # batch-relative trimming: discard a candidate whose minimum distance is an
                # extreme outlier with respect to the other passing candidates
                med = float(np.median(dmin[ok]))
                keep = ok[dmin[ok] <= float(trim) * med]
                if keep.size:
                    ok = keep
            if ok.size:
                if select in ("weighted", "weighted2") and ok.size > 1:
                    w = dmin[ok].astype(np.float64)
                    w = w * w if select == "weighted2" else w
                    p = w / w.sum()
                    pick = int(cand[ok[int(rng.choice(ok.size, p=p))]])
                else:
                    order = ok[np.argsort(dmin[ok])[::-1]]
                    pick = int(cand[order[min(rank - 1, len(order) - 1)]])
                break
            pick = int(cand[int(np.argmax(dmin))])
        else:
            fallbacks += 1
        trials[k - 1] = t
        idx[k] = pick
        centers[k] = X[pick]
    method = "kmeans_gs_w" if select in ("weighted", "weighted2") else "kmeans_gs_g"
    return SeedResult(centers=centers, method=method, trials=trials,
                      fallbacks=fallbacks, taus=taus, d0=d0,
                      iqr_n_used=int(sub.shape[0]), indices=idx, first_mode="random")


def seed_kmeans_gs_g(X, K, rng, **kwargs):
    """KMeans-GS-G: farthest admissible candidate, b = 8 by default."""
    kwargs.setdefault("select", "max")
    kwargs.setdefault("batch", 8)
    return seed_kmeans_gs_select(X, K, rng, **kwargs)


def seed_kmeans_gs_w(X, K, rng, **kwargs):
    """KMeans-GS-W: D^2-weighted admissible candidate, b = 8 by default."""
    kwargs.setdefault("select", "weighted2")
    kwargs.setdefault("batch", 8)
    return seed_kmeans_gs_select(X, K, rng, **kwargs)


# 播种型方法（统一走"播种 + 共享 Lloyd"）：名字 -> 函数
SEEDERS = {
    "kmeans": seed_random,
    "kmeanspp": seed_kmeanspp,
    "kmeans_gs": seed_kmeans_gs,
    "kmeans_gs_g": seed_kmeans_gs_g,
    "kmeans_par": seed_kmeans_parallel,
    "afkmc2": seed_afkmc2,
}

METHOD_LABELS = {
    "kmeans": "K-means",
    "kmeanspp": "K-means++",
    "kmeans_gs": "KMeans-GS",
    "kmeans_gs_g": "KMeans-GS-G",
    "kmeans_par": "K-means||",
    "afkmc2": "AFK-MC²",
}
