#!/usr/bin/env bash
# Reproduce every experiment reported in the KMeans-GS manuscript.
#
# The script is intentionally resumable and keeps running after a failed phase.
# Every phase writes its own log and status.tsv records PASS/FAIL.
#
# Usage:
#   bash experiments/run_all_h100.sh
#   ROOT=/home/hpclr/pattern_recognition_20261019/results/paper_reproduction \
#     PY=/home/hpclr/miniconda3/envs/ai4sci/bin/python \
#     bash experiments/run_all_h100.sh

set -uo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/home/hpclr/miniconda3/envs/ai4sci/bin/python}"
SCDATA="${SCDATA:-/home/hpclr/scdata}"
ROOT="${ROOT:-../results/paper_reproduction}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
STATUS="${STATUS:-${LOG_DIR}/status.tsv}"
REPEATS="${REPEATS:-20}"
SENS_REPEATS="${SENS_REPEATS:-30}"
MAIN_CPU="${MAIN_CPU:-8}"
SCALING_JOBS="${SCALING_JOBS:-24}"
SENS_JOBS="${SENS_JOBS:-12}"

DATASETS="dermatology,wine_quality,wdbc,synthetic_control,letter,musk,isolet,segment,yeast,vehicle,pendigits,satimage,synth_imbalanced"
METHODS="kmeans,kmeanspp,kmeans_gs,kmeans_gs_g,kmeans_par,afkmc2,mbkmeans,dbscan,birch,gmm"
GS_METHODS="kmeans_gs,kmeans_gs_g"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$ROOT"/{main_20rep,tables_main,diagnostics,scaling,applications,sensitivity,report}
if [ ! -s "$STATUS" ]; then
    printf 'timestamp\tphase\trc\tseconds\tlog\n' > "$STATUS"
fi

FAILS=0

run_phase() {
    local name="$1"
    shift
    local log="$LOG_DIR/${name}.log"
    local t0 t1 rc
    t0="$(date +%s)"
    echo
    echo "============================================================"
    echo "[START] ${name}  $(date -Is)"
    echo "============================================================"
    if "$@" 2>&1 | tee "$log"; then
        rc=0
    else
        rc=$?
    fi
    t1="$(date +%s)"
    if [ "$rc" -eq 0 ]; then
        echo "[PASS] ${name}  $((t1 - t0)) s"
    else
        echo "[FAIL] ${name}  rc=${rc}  $((t1 - t0)) s  log=${log}"
        FAILS=$((FAILS + 1))
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" "$name" "$rc" "$((t1 - t0))" "$log" >> "$STATUS"
    return 0
}

PIN=()
if command -v taskset >/dev/null 2>&1 && taskset -c "$MAIN_CPU" true >/dev/null 2>&1; then
    PIN=(taskset -c "$MAIN_CPU")
fi

EPS="${ROOT}/diagnostics/dbscan_eps_13.csv"

echo "host=$(hostname)"
echo "python=${PY}"
echo "root=${ROOT}"
echo "datasets=${DATASETS}"
echo "repeats=${REPEATS}"
echo "main_cpu=${MAIN_CPU}"

run_phase env_check "$PY" scripts/env_check.py
run_phase prepare_datasets "$PY" scripts/run_experiment.py --prepare

run_phase choose_dbscan_eps "$PY" scripts/choose_dbscan_eps.py \
    --datasets "$DATASETS" --out "$EPS" --min-samples 5 \
    --sample 3000 --n-grid 10 --seed 42

# 7.1 Main quality and timing table. Keep jobs=1 and pin one core so all
# reported wall-clock times come from the same controlled process.
run_phase main_20rep "${PIN[@]}" "$PY" scripts/run_experiment.py \
    --datasets "$DATASETS" --methods "$METHODS" --repeats "$REPEATS" \
    --jobs 1 --job-mode dataset --dbscan-eps-file "$EPS" \
    --max-seconds 7200 --out "$ROOT/main_20rep"

run_phase tables_main "$PY" scripts/make_paper_tables.py \
    --raw "$ROOT/main_20rep" --out "$ROOT/tables_main"

# Seed-only diagnostics used by the separation theorem and Figure 2b.
run_phase seed_assignment "$PY" scripts/seed_assignment.py \
    --datasets "$DATASETS" --repeats "$REPEATS" \
    --methods "$GS_METHODS" \
    --out "$ROOT/diagnostics/seed_assignment.csv"

# 7.2 Controlled scaling of the seeding stage.
run_phase scaling "$PY" scripts/scaling_study.py \
    --out "$ROOT/scaling" --repeats 5 --jobs "$SCALING_JOBS" \
    --pin-base 40 --corner

# 7.4-7.5 Applications.
run_phase app_text "$PY" scripts/text_app.py \
    --path "$SCDATA/dbpedia_train.parquet" --repeats 10 \
    --methods kmeans,kmeanspp,kmeans_gs,kmeans_gs_g \
    --out "$ROOT/applications/text_dbpedia10.csv"

run_phase app_covid_k18 "${PIN[@]}" "$PY" scripts/atlas_app.py \
    --path "$SCDATA/covid_atlas.h5ad" --K 18 --restarts 3 \
    --pca-key X_pca --label-key cell_type \
    --methods kmeans,kmeanspp,kmeans_gs,kmeans_gs_g \
    --out "$ROOT/applications/covid_k18.csv"

run_phase app_hlca_k25 "${PIN[@]}" "$PY" scripts/atlas_app.py \
    --path "$SCDATA/hlca_core.h5ad" --K 25 --restarts 2 \
    --pca-key "" --label-key ann_level_3 \
    --methods kmeans,kmeanspp,kmeans_gs,kmeans_gs_g \
    --out "$ROOT/applications/hlca_k25.csv"

# Fine-grained COVID diagnostic reported in Section 7.5. It is optional
# because the label key can differ between atlas releases.
if [ "${RUN_K64:-1}" = "1" ]; then
    run_phase app_covid_k64 "${PIN[@]}" "$PY" scripts/atlas_app.py \
        --path "$SCDATA/covid_atlas.h5ad" --K 64 --restarts 1 \
        --pca-key X_pca --label-key celltype \
        --methods kmeans,kmeanspp,kmeans_gs,kmeans_gs_g \
        --out "$ROOT/applications/covid_k64.csv"
fi

run_phase app_paul15 "$PY" scripts/scrna_app.py \
    --dataset paul15 --path "$SCDATA/paul15.h5" --repeats 10 \
    --out "$ROOT/applications/paul15.csv"

run_phase app_pbmc10k "$PY" scripts/scrna_app.py \
    --dataset pbmc10k --path "$SCDATA/filtered_feature_bc_matrix" \
    --K 10 --repeats 10 --out "$ROOT/applications/pbmc10k.csv"

# 7.6 Parameter sensitivity. Quality-only runs are parallelized; all timing
# claims remain tied to the single-core main run above.
run_sens() {
    local name="$1"
    shift
    run_phase "sens_${name}" "$PY" scripts/run_experiment.py \
        --datasets "$DATASETS" --methods kmeans_gs --repeats "$SENS_REPEATS" \
        --jobs "$SENS_JOBS" --job-mode dataset --max-seconds 7200 \
        --out "$ROOT/sensitivity/$name" "$@"
}

run_sens m500 --gs-iqr-sample 500
run_sens m1000 --gs-iqr-sample 1000
run_sens m2000 --gs-iqr-sample 2000
run_sens m5000 --gs-iqr-sample 5000
run_sens mfull --gs-iqr-sample 0
run_sens d0_03 --gs-d0-coef 0.3
run_sens d0_05 --gs-d0-coef 0.5
run_sens d0_07 --gs-d0-coef 0.7
run_sens decay_03 --gs-decay power --gs-decay-p 0.3
run_sens decay_05 --gs-decay power --gs-decay-p 0.5
run_sens decay_07 --gs-decay power --gs-decay-p 0.7
run_sens first_random --gs-first random
run_sens first_median --gs-first median
run_sens first_median_sub --gs-first median_sub
run_sens first_lds --gs-first lds
run_sens trials20 --gs-max-trials 20
run_sens trials50 --gs-max-trials 50
run_sens trials100 --gs-max-trials 100
run_sens edge --gs-edge

run_phase iqr_stability "$PY" scripts/iqr_sensitivity.py \
    --datasets "$DATASETS" --m 500,1000,2000,5000 --b 100 \
    --out "$ROOT/diagnostics/iqr_stability.csv"

run_phase summarize "$PY" analysis/aggregate_20rep.py \
    --raw "$ROOT/main_20rep" --out "$ROOT/report"

run_phase extra_evidence "$PY" analysis/analyze_extra_20rep.py \
    --stats "$ROOT/report" --out "$ROOT/report/extra_evidence"

run_phase package_results bash -c \
    "cd '$ROOT' && tar -czf results_paper.tgz main_20rep tables_main diagnostics scaling applications sensitivity report"

echo
echo "============================================================"
if [ "$FAILS" -eq 0 ]; then
    echo "[ALL DONE] $(date -Is)"
else
    echo "[DONE WITH ${FAILS} FAILED PHASES] $(date -Is)"
fi
echo "status: $STATUS"
echo "root:   $ROOT"
echo "============================================================"
exit "$([ "$FAILS" -eq 0 ] && echo 0 || echo 1)"
