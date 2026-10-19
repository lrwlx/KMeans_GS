# 20-seed post-processing

`aggregate_20rep.py` validates the 13 x 10 x 20 run grid, computes equal-dataset
ARI, ranks, paired tests, leave-one-dataset-out results and time summaries.

```bash
python analysis/aggregate_20rep.py \
  --raw ../results/main_20rep \
  --out ../results/summary
```

`analyze_extra_20rep.py` computes the seed-time scaling exponents, large-workload
checks, oracle regret, Pareto position and downside diagnostics used in the
manuscript.

```bash
python analysis/analyze_extra_20rep.py \
  --stats ../results/summary \
  --out ../results/summary/extra_evidence
```
