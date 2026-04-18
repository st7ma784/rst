#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Submit a full multi-stage RST pipeline on Slurm from a CSV manifest.

Usage:
  submit_rst_pipeline.sh --manifest FILE [options]

Required:
  --manifest FILE                 CSV manifest with per-radar/day parameters.

Optional:
  --work-root DIR                 Output working root. Default: ./slurm_runs
  --log-root DIR                  Slurm log root. Default: ./slurm_logs
  --rst-env FILE                  Bash script that exports RST environment.
  --fit-partition NAME            Slurm partition for fit stage.
  --grid-partition NAME           Slurm partition for grid stage.
  --map-partition NAME            Slurm partition for map/finalize stages.
  --fit-time HH:MM:SS             Fit walltime. Default: 02:00:00
  --grid-time HH:MM:SS            Grid walltime. Default: 01:00:00
  --map-time HH:MM:SS             Map walltime. Default: 01:00:00
  --final-time HH:MM:SS           Finalize walltime. Default: 00:30:00
  --fit-cpus N                    Fit cpus-per-task. Default: 2
  --grid-cpus N                   Grid cpus-per-task. Default: 2
  --map-cpus N                    Map cpus-per-task. Default: 2
  --final-cpus N                  Finalize cpus-per-task. Default: 1
  --fit-array-limit N             Max concurrent fit array tasks. Default: 200
  --grid-array-limit N            Max concurrent grid array tasks. Default: 200
  --account NAME                  Slurm account.
  --qos NAME                      Slurm QoS.
  --skip-exact-generator          Skip glob resolution and use manifest as-is.
  --allow-empty-inputs            Allow rows with no matching input files.
  --dry-run                       Print sbatch commands without submitting.

Manifest assumptions:
  - First line is a header.
  - Columns are comma separated without embedded commas in values.
  - Required columns are documented in scripts/slurm/README.md.
EOF
}

MANIFEST=""
WORK_ROOT="./slurm_runs"
LOG_ROOT="./slurm_logs"
RST_ENV=""
FIT_PARTITION=""
GRID_PARTITION=""
MAP_PARTITION=""
FIT_TIME="02:00:00"
GRID_TIME="01:00:00"
MAP_TIME="01:00:00"
FINAL_TIME="00:30:00"
FIT_CPUS="2"
GRID_CPUS="2"
MAP_CPUS="2"
FINAL_CPUS="1"
FIT_ARRAY_LIMIT="200"
GRID_ARRAY_LIMIT="200"
ACCOUNT=""
QOS=""
SKIP_EXACT_GENERATOR="0"
ALLOW_EMPTY_INPUTS="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --rst-env) RST_ENV="$2"; shift 2 ;;
    --fit-partition) FIT_PARTITION="$2"; shift 2 ;;
    --grid-partition) GRID_PARTITION="$2"; shift 2 ;;
    --map-partition) MAP_PARTITION="$2"; shift 2 ;;
    --fit-time) FIT_TIME="$2"; shift 2 ;;
    --grid-time) GRID_TIME="$2"; shift 2 ;;
    --map-time) MAP_TIME="$2"; shift 2 ;;
    --final-time) FINAL_TIME="$2"; shift 2 ;;
    --fit-cpus) FIT_CPUS="$2"; shift 2 ;;
    --grid-cpus) GRID_CPUS="$2"; shift 2 ;;
    --map-cpus) MAP_CPUS="$2"; shift 2 ;;
    --final-cpus) FINAL_CPUS="$2"; shift 2 ;;
    --fit-array-limit) FIT_ARRAY_LIMIT="$2"; shift 2 ;;
    --grid-array-limit) GRID_ARRAY_LIMIT="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --skip-exact-generator) SKIP_EXACT_GENERATOR="1"; shift ;;
    --allow-empty-inputs) ALLOW_EMPTY_INPUTS="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  echo "Error: --manifest is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Error: manifest not found: $MANIFEST" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_SCRIPT="$SCRIPT_DIR/rst_stage_fit.sbatch"
GRID_SCRIPT="$SCRIPT_DIR/rst_stage_grid.sbatch"
MAP_SCRIPT="$SCRIPT_DIR/rst_stage_map.sbatch"
FINAL_SCRIPT="$SCRIPT_DIR/rst_stage_finalize.sbatch"
EXACT_MANIFEST_GENERATOR="$SCRIPT_DIR/generate_exact_input_manifest.py"

for req in "$FIT_SCRIPT" "$GRID_SCRIPT" "$MAP_SCRIPT" "$FINAL_SCRIPT"; do
  [[ -f "$req" ]] || { echo "Missing script: $req" >&2; exit 1; }
done

if [[ "$SKIP_EXACT_GENERATOR" != "1" ]]; then
  [[ -f "$EXACT_MANIFEST_GENERATOR" ]] || {
    echo "Missing exact-manifest generator: $EXACT_MANIFEST_GENERATOR" >&2
    exit 1
  }
fi

RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$WORK_ROOT/$RUN_ID"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
GROUP_DIR="$RUN_DIR/manifests"
mkdir -p "$RUN_DIR" "$RUN_LOG_DIR" "$GROUP_DIR"
cp "$MANIFEST" "$RUN_DIR/input_manifest.csv"

if [[ "$SKIP_EXACT_GENERATOR" == "1" ]]; then
  RESOLVED_MANIFEST="$RUN_DIR/input_manifest.csv"
else
  RESOLVED_MANIFEST="$RUN_DIR/resolved_manifest.csv"
  INPUT_LIST_DIR="$RUN_DIR/input_lists"
  generator_args=(
    --manifest "$RUN_DIR/input_manifest.csv"
    --output "$RESOLVED_MANIFEST"
    --list-dir "$INPUT_LIST_DIR"
  )
  [[ "$ALLOW_EMPTY_INPUTS" == "1" ]] && generator_args+=(--allow-empty)
  python3 "$EXACT_MANIFEST_GENERATOR" "${generator_args[@]}"
fi

header="$(head -n 1 "$RESOLVED_MANIFEST")"

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  date_key="$(echo "$line" | cut -d, -f1)"
  hemi_key="$(echo "$line" | cut -d, -f3 | tr '[:upper:]' '[:lower:]')"
  group_file="$GROUP_DIR/${date_key}_${hemi_key}.csv"
  if [[ ! -f "$group_file" ]]; then
    echo "$header" > "$group_file"
  fi
  echo "$line" >> "$group_file"
done < <(tail -n +2 "$RESOLVED_MANIFEST")

export_list="ALL,MANIFEST,RUN_DIR,RST_ENV,WAIT_TIMEOUT_SEC,WAIT_INTERVAL_SEC"

submit_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN sbatch $*"
    return 0
  fi
  sbatch "$@"
}

extract_job_id() {
  local raw="$1"
  echo "$raw" | awk '{print $NF}'
}

common_sbatch_args=()
[[ -n "$ACCOUNT" ]] && common_sbatch_args+=("--account=$ACCOUNT")
[[ -n "$QOS" ]] && common_sbatch_args+=("--qos=$QOS")

map_job_ids=()

echo "Submitting grouped pipelines from: $GROUP_DIR"

for group_manifest in "$GROUP_DIR"/*.csv; do
  [[ -f "$group_manifest" ]] || continue
  group_name="$(basename "$group_manifest" .csv)"
  tasks=$(( $(wc -l < "$group_manifest") - 1 ))
  if [[ "$tasks" -le 0 ]]; then
    continue
  fi

  date_key="${group_name%%_*}"
  hemi_key="${group_name#*_}"

  fit_args=(
    --parsable
    --job-name="rst_fit_${date_key}_${hemi_key}"
    --output="$RUN_LOG_DIR/%x_%A_%a.out"
    --error="$RUN_LOG_DIR/%x_%A_%a.err"
    --time="$FIT_TIME"
    --cpus-per-task="$FIT_CPUS"
    --array="1-${tasks}%${FIT_ARRAY_LIMIT}"
    "${common_sbatch_args[@]}"
  )
  [[ -n "$FIT_PARTITION" ]] && fit_args+=("--partition=$FIT_PARTITION")

  fit_args+=("--export=$export_list" "$FIT_SCRIPT")

  fit_raw="$(MANIFEST="$group_manifest" RUN_DIR="$RUN_DIR" RST_ENV="$RST_ENV" submit_cmd "${fit_args[@]}")"
  fit_job_id="$(extract_job_id "$fit_raw")"
  echo "[$group_name] fit job: $fit_job_id"

  grid_args=(
    --parsable
    --job-name="rst_grid_${date_key}_${hemi_key}"
    --output="$RUN_LOG_DIR/%x_%A_%a.out"
    --error="$RUN_LOG_DIR/%x_%A_%a.err"
    --time="$GRID_TIME"
    --cpus-per-task="$GRID_CPUS"
    --array="1-${tasks}%${GRID_ARRAY_LIMIT}"
    "--dependency=aftercorr:${fit_job_id}"
    "${common_sbatch_args[@]}"
  )
  [[ -n "$GRID_PARTITION" ]] && grid_args+=("--partition=$GRID_PARTITION")

  grid_args+=("--export=$export_list" "$GRID_SCRIPT")

  grid_raw="$(MANIFEST="$group_manifest" RUN_DIR="$RUN_DIR" RST_ENV="$RST_ENV" submit_cmd "${grid_args[@]}")"
  grid_job_id="$(extract_job_id "$grid_raw")"
  echo "[$group_name] grid job: $grid_job_id"

  map_args=(
    --parsable
    --job-name="rst_map_${date_key}_${hemi_key}"
    --output="$RUN_LOG_DIR/%x_%A.out"
    --error="$RUN_LOG_DIR/%x_%A.err"
    --time="$MAP_TIME"
    --cpus-per-task="$MAP_CPUS"
    "--dependency=afterok:${grid_job_id}"
    "${common_sbatch_args[@]}"
  )
  [[ -n "$MAP_PARTITION" ]] && map_args+=("--partition=$MAP_PARTITION")

  map_args+=("--export=$export_list" "$MAP_SCRIPT")

  map_raw="$(MANIFEST="$group_manifest" RUN_DIR="$RUN_DIR" RST_ENV="$RST_ENV" submit_cmd "${map_args[@]}")"
  map_job_id="$(extract_job_id "$map_raw")"
  echo "[$group_name] map job: $map_job_id"

  map_job_ids+=("$map_job_id")
done

if [[ ${#map_job_ids[@]} -eq 0 ]]; then
  echo "No map jobs were submitted (empty manifest?)"
  exit 0
fi

dep_chain="$(IFS=:; echo "${map_job_ids[*]}")"

final_args=(
  --parsable
  --job-name="rst_finalize_${RUN_ID}"
  --output="$RUN_LOG_DIR/%x_%A.out"
  --error="$RUN_LOG_DIR/%x_%A.err"
  --time="$FINAL_TIME"
  --cpus-per-task="$FINAL_CPUS"
  "--dependency=afterok:${dep_chain}"
  "${common_sbatch_args[@]}"
)
[[ -n "$MAP_PARTITION" ]] && final_args+=("--partition=$MAP_PARTITION")

final_args+=("--export=ALL,RUN_DIR" "$FINAL_SCRIPT")

final_raw="$(RUN_DIR="$RUN_DIR" submit_cmd "${final_args[@]}")"
final_job_id="$(extract_job_id "$final_raw")"

echo "Final aggregation job: $final_job_id"
echo "Run directory: $RUN_DIR"
echo "Log directory: $RUN_LOG_DIR"
