# KMeans-GS 项目任务说明

## 当前任务

本仓库对应论文初稿：

> **KMeans-GS: Geometric seeding with data-calibrated thresholds and
> sample-size-independent cost**

论文研究 K-means 的初始化阶段。目标是构造一种播种规则，使：

1. 距离阈值由数据自身的稳健尺度决定，不需要用户手工指定；
2. 播种阶段的计算量不随样本数 N 线性增长；
3. 种子之间具有可证明的几何分离下界；
4. 在大规模、高 K 场景下降低播种开销，同时保持与单次 k-means++ 相当的聚类质量。

方法主体为 KMeans-GS。它从数据子样本的 IQR 估计尺度 \(d_0\)，按
\(\tau_i = d_0/\sqrt{i}\) 收紧阈值，并用拒绝采样接受满足分离条件的候选点。
两个选择规则共享同一阈值和接受检验：

- **KMeans-GS**：接受第一个通过检验的候选点；
- **KMeans-GS-G**：在通过检验的候选中选择距离当前种子集最远者。

## 当前实验状态

主实验已经锁定为：

- 13 个数据集；
- 10 种 single-run 方法；
- 20 个配对种子（42-61）；
- 2,600 次成功运行。

论文中的主要结果包括：

- KMeans-GS 平均 ARI 0.3492，k-means++ 为 0.3457；
- KMeans-GS 平均排名 4.31，为十种方法中最好；
- 但平均 ARI 差异不显著（bootstrap 95% CI 包含 0），因此论文将结论限定为
  “小的描述性质量优势 + 统计上的质量平价”；
- 13 数据集拟合的播种时间指数为 N：0.438 vs 0.775，K：-0.060 vs 1.099；
- 在 K >= 20 的工作负载上，播种快约 7.4 倍，端到端快约 1.14 倍；
- 高 K 下 ARI 点估计低 0.0039，因此论文不声称高 K 质量更高，只声称播种成本显著降低。

30 次运行仅作为敏感性分析，不参与主排名。

## 应用实验

论文包含三类大 representation：

1. COVID-19 单细胞图谱；
2. HLCA 肺图谱；
3. DBpedia-14 文本语料。

两个图谱使用 selected-restart 结果；DBpedia 使用 10 次配对运行的均值。
这些数据不随仓库分发，需要从 CELLxGENE、10x Genomics 和 DBpedia 获取。

## 代码范围

仓库包含：

- KMeans-GS / KMeans-GS-G 的核心实现；
- Lloyd、指标和 10 方法 benchmark runner；
- 13 数据集主实验；
- N/K/d controlled scaling；
- 图谱和文本应用脚本；
- 20 次实验的聚合、统计和稳健性分析；
- 图表生成所需的数据和结果摘要。

不包含：

- 原始单细胞图谱和 DBpedia 原始数据；
- 旧的 10 次主实验版本；
- 30 次 replication 的完整 raw records；
- 缓存、下载中间文件和旧论文渲染图。

## 复现入口

```bash
python -m pip install -r requirements.txt
REPEATS=20 bash run_main_benchmark.sh
```

完整流程见 `experiments/README.md`，方法与结果说明见根目录 `README.md`。
