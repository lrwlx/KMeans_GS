# KMeans-GS

Reference code for the manuscript:

> **KMeans-GS: Geometric seeding with data-calibrated thresholds and
> sample-size-independent cost**

The repository studies the initialization stage of K-means. It implements a
rejection-sampling seeding rule whose distance threshold is calibrated from the
interquartile range (IQR) of the data, together with two selection rules:

- **KMeans-GS**: accept the first admissible candidate.
- **KMeans-GS-G**: accept the farthest admissible candidate.

The manuscript is under review at *Pattern Recognition*. The final experimental
protocol uses 13 benchmark datasets, 10 single-run methods, 20 paired seeds
(42-61), and 2,600 successful main-benchmark runs.

See [TASK_OVERVIEW.md](TASK_OVERVIEW.md) for the research task and current
status.

## What is implemented

- IQR-calibrated scale estimation with a fixed subsample.
- Inverse-square-root threshold schedule.
- First-candidate and farthest-candidate selection rules.
- A 32-dimensional sketched acceptance test.
- Shared Lloyd refinement for all seeding-based methods.
- Baselines: k-means++, k-means||, AFK-MC2, MiniBatchKMeans, BIRCH, DBSCAN and
  Gaussian mixture.
- The 13-dataset, 10-method benchmark and the 20-seed post-processing pipeline.
- Single-cell atlas and DBpedia text experiment drivers.

## Repository layout

```text
src/                 Core seeding, Lloyd, metrics and runner code
scripts/             Benchmark, scaling, application and table scripts
experiments/         End-to-end H100 experiment pipeline
analysis/            20-seed aggregation and extra statistical analysis
data/                DBSCAN epsilon table used by the benchmark
results/summary/     Packaged 20-seed summary statistics
results/applications/
                     Packaged atlas and text application records
results/scaling/     Controlled N/K/d scaling summaries
results/sensitivity/ 30-seed sensitivity summary
```

## Environment

```bash
python -m pip install -r requirements.txt
```

The code is CPU-only. Timing experiments use one BLAS thread per process:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Main benchmark

```bash
REPEATS=20 bash run_main_benchmark.sh
```

By default the run-level records are written to `../results/main_20rep` and
paper tables to `../results/tables_main_20rep`. The DBSCAN epsilon table is
included as `data/dbscan_eps_13.csv`.

The main benchmark uses these 10 methods:

`kmeans`, `kmeanspp`, `kmeans_gs`, `kmeans_gs_g`, `kmeans_par`, `afkmc2`,
`mbkmeans`, `dbscan`, `birch`, `gmm`.

## Full pipeline

```bash
bash experiments/run_all_h100.sh
```

The pipeline runs the 20-seed main comparison, seed diagnostics, N/K/d scaling,
the COVID and HLCA atlas applications, the DBpedia text application, the
smaller single-cell checks, and the 30-seed sensitivity analysis.

Atlas and DBpedia inputs are not redistributed. Their paths are supplied through
`SCDATA` in `experiments/run_all_h100.sh`.

## Post-processing

```bash
python analysis/aggregate_20rep.py \
  --raw ../results/main_20rep \
  --out ../results/summary

python analysis/analyze_extra_20rep.py \
  --stats ../results/summary \
  --out ../results/summary/extra_evidence
```

The first script validates the 13 x 10 x 20 run grid and computes ARI, ranks,
paired tests, leave-one-dataset-out results and time summaries. The second adds
the seed-time scaling exponents, large-workload checks, oracle regret and
Pareto diagnostics.

## Current evidence in brief

- Mean ARI: **0.3492** for KMeans-GS versus **0.3457** for k-means++.
- Mean rank: **4.31** for KMeans-GS versus **4.77** for k-means++.
- The paired ARI difference is small and not statistically significant
  (95% CI -0.0039 to +0.0121).
- Across the 13 datasets, the fitted seeding-time exponent in N is **0.438**
  for KMeans-GS versus **0.775** for k-means++.
- The fitted exponent in K is **-0.060** versus **1.099**.
- On K >= 20 workloads, seeding is about **7.4x** faster and end-to-end time is
  about **1.14x** faster, with a small ARI point-estimate difference of -0.0039.
- The base rule recorded no fallback in the main benchmark.

The empirical case is therefore about **seeding-cost scaling**, not universal
quality superiority or faster Lloyd iterations.

## Data

The 12 public benchmark datasets are downloaded through OpenML/UCI. The single
generated stress test, `synth_imbalanced`, is produced by the code. DBpedia-14
and the CELLxGENE atlases must be downloaded separately. No raw atlas or DBpedia
data are included in this repository.

## Notes

- The main comparison uses single-run methods. Multi-restart scikit-learn
  KMeans, Elkan KMeans and `n_init="auto"` are excluded because they answer a
  different resource question.
- The 30-seed sensitivity analysis is separate from the 20-seed main ranking.
- The recovery theorem is a sufficient condition for separable data; it is not
  a claim that real-world clusters satisfy that condition.
