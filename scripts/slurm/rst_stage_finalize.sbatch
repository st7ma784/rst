#!/usr/bin/env bash
#SBATCH --ntasks=1

set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be exported into the job environment" >&2
  exit 1
fi

summary="$RUN_DIR/pipeline_summary.txt"
report_csv="$RUN_DIR/pipeline_summary.csv"

fit_count=$(find "$RUN_DIR/meta" -name fit.done -type f 2>/dev/null | wc -l)
grid_count=$(find "$RUN_DIR/meta" -name grid.done -type f 2>/dev/null | wc -l)
map_count=$(find "$RUN_DIR/meta" -name map.done -type f 2>/dev/null | wc -l)

{
  echo "RST Slurm Pipeline Summary"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_dir=$RUN_DIR"
  echo "fit_completed=$fit_count"
  echo "grid_completed=$grid_count"
  echo "map_completed=$map_count"
  echo "status=complete"
} > "$summary"

{
  echo "metric,value"
  echo "fit_completed,$fit_count"
  echo "grid_completed,$grid_count"
  echo "map_completed,$map_count"
} > "$report_csv"

echo "Wrote summary: $summary"
echo "Wrote summary CSV: $report_csv"
