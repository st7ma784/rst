#!/usr/bin/env bash
#SBATCH --job-name=rst-fitacf-bench
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

if [[ -n "${RST_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "$RST_ENV"
fi

ROOT_DIR="${ROOT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ITERATIONS="${ITERATIONS:-30}"
SCALE="${SCALE:-32}"
OUTPUT_FILE="${OUTPUT_FILE:-$ROOT_DIR/pythonv2/benchmark_results/fitacf_gpu_vs_cpu_slurm.json}"

cd "$ROOT_DIR/pythonv2"

"$PYTHON_BIN" scripts/benchmark_fitacf_batch.py \
  --backend both \
  --mode all \
  --require-cupy \
  --iterations "$ITERATIONS" \
  --warmup 3 \
  --scale "$SCALE" \
  --output "$OUTPUT_FILE"

printf 'FITACF benchmark complete. Results: %s\n' "$OUTPUT_FILE"
