#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATASETS="dermatology,wine_quality,wdbc,synthetic_control,letter,musk,isolet,segment,yeast,vehicle,pendigits,satimage,synth_imbalanced"
METHODS="kmeans,kmeanspp,kmeans_gs,kmeans_gs_g,kmeans_par,afkmc2,mbkmeans,dbscan,birch,gmm"
REPEATS="${REPEATS:-20}"
JOBS="${JOBS:-1}"
OUT="${OUT:-../results/main_20rep}"
TABLES="${TABLES:-../results/tables_main_20rep}"
EPS="${EPS:-data/dbscan_eps_13.csv}"

mkdir -p "$OUT" "$TABLES"

python scripts/run_experiment.py \
  --datasets "$DATASETS" \
  --methods "$METHODS" \
  --repeats "$REPEATS" \
  --jobs "$JOBS" \
  --dbscan-eps-file "$EPS" \
  --max-seconds 7200 \
  --out "$OUT"

python scripts/make_paper_tables.py \
  --raw "$OUT" \
  --out "$TABLES"
