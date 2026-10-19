# Experiment pipeline

`run_all_h100.sh` reproduces the final manuscript:

| Stage | Manuscript | Output |
|---|---|---|
| main_20rep | Sections 6.3 and 7.1, Tables 2 and 5 | `main_20rep/` |
| seed_assignment | Section 5 diagnostics, Figure 2b | `diagnostics/seed_assignment.csv` |
| scaling | Sections 7.2-7.3, Tables 3-4, Figure 1 | `scaling/` |
| app_text | Sections 7.4-7.5, Tables 6 and 8, Figure 3 | `applications/text_dbpedia10.csv` |
| app_covid_k18 | Sections 7.4-7.5, Tables 6-7, Figure 2a | `applications/covid_k18.csv` |
| app_hlca_k25 | Section 7.5, Tables 6-7 | `applications/hlca_k25.csv` |
| app_covid_k64 | Fine-grained check in Section 7.5 | `applications/covid_k64.csv` |
| app_paul15 / app_pbmc10k | Smaller single-cell checks | `applications/` |
| sensitivity | Section 7.6, Table 9 | `sensitivity/` |
| summarize | Paper-ready CSV tables and report | `report/` |

## Run

```bash
cd code
PY=/path/to/python \
ROOT=../results/paper_reproduction \
bash experiments/run_all_h100.sh
```

The main comparison uses 13 datasets, 10 methods and 20 paired seeds
(42-61). It runs with `JOBS=1`, one BLAS thread and an optional `taskset`
core pin because its timings enter Table 2. The sensitivity analysis uses 30
seeds and is not used for the main ranking.

Every phase writes `logs/<phase>.log` and appends PASS/FAIL to
`logs/status.tsv`. A failed phase does not stop later phases; rerunning the
script resumes completed dataset files and scaling shards.

## Inputs

`atlas_app.py` and `scrna_app.py` need `h5py`; the text experiment needs
`pyarrow`. Atlas and DBpedia inputs are passed through `SCDATA`. The repository
does not redistribute those datasets.

## Deliberate exclusions

- `KMeans-GS-W` is not in the main comparison.
- scikit-learn KMeans with `n_init=10`, Elkan KMeans and `n_init="auto"` are
  excluded from the single-run table because they use a different restart
  budget.
- DBSCAN is limited to datasets with at most 20,000 points.
