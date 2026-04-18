#!/usr/bin/env bash
#SBATCH --ntasks=1

set -euo pipefail

if [[ -z "${MANIFEST:-}" || -z "${RUN_DIR:-}" ]]; then
  echo "MANIFEST and RUN_DIR must be exported into the job environment" >&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script expects to run as an array job" >&2
  exit 1
fi

if [[ -n "${RST_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "$RST_ENV"
fi

WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-10}"

wait_for_file() {
  local target="$1"
  local waited=0
  while [[ ! -s "$target" ]]; do
    if [[ "$waited" -ge "$WAIT_TIMEOUT_SEC" ]]; then
      echo "Timed out waiting for file: $target" >&2
      return 1
    fi
    sleep "$WAIT_INTERVAL_SEC"
    waited=$((waited + WAIT_INTERVAL_SEC))
  done
}

require_bin() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required binary: $1" >&2; exit 1; }
}

require_bin make_grid

line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [[ -z "$line" ]]; then
  echo "No manifest row for task id $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

IFS=',' read -r date radar hemisphere input_glob fit_mode fitacf_version tdiff_method tdiff_value channel_mode scan_length_sec grid_interval_sec grid_extra_flags map_model map_order map_doping imf_mode imf_file bx by bz extra_map_flags use_cuda <<< "$line"

fit_dir="$RUN_DIR/fits/$date/$radar"
meta_dir="$RUN_DIR/meta/$date/$radar"
grid_dir="$RUN_DIR/grids/$date/$radar"
mkdir -p "$grid_dir" "$meta_dir"

in_fit="$fit_dir/${date}.${radar}.fitacf"
out_grid="$grid_dir/${date}.${radar}.grd"
state_ok="$meta_dir/grid.done"

if [[ -f "$state_ok" && -s "$out_grid" ]]; then
  echo "Grid stage already complete for $date/$radar"
  exit 0
fi

if [[ ! -s "$in_fit" ]]; then
  wait_for_file "$in_fit" || exit 2
fi

grid_args=()
if [[ -n "$scan_length_sec" && "$scan_length_sec" != "-" ]]; then
  grid_args+=("-tl" "$scan_length_sec")
fi

if [[ -n "$grid_interval_sec" && "$grid_interval_sec" != "-" ]]; then
  grid_args+=("-i" "$grid_interval_sec")
fi

case "$channel_mode" in
  a|b)
    grid_args+=("-cn" "$channel_mode")
    ;;
  all|""|-)
    ;;
  *)
    echo "Invalid channel_mode '$channel_mode' (valid: all, a, b)" >&2
    exit 3
    ;;
esac

if [[ -n "$grid_extra_flags" && "$grid_extra_flags" != "-" ]]; then
  # shellcheck disable=SC2206
  extra=( $grid_extra_flags )
  grid_args+=("${extra[@]}")
fi

echo "[$date/$radar] GRID from $in_fit"
make_grid "${grid_args[@]}" "$in_fit" > "$out_grid"

tmp_meta="$meta_dir/grid.meta.tmp"
cat > "$tmp_meta" <<EOF
stage=grid
date=$date
radar=$radar
input_fit=$in_fit
output_grid=$out_grid
EOF
mv "$tmp_meta" "$meta_dir/grid.meta"

touch "$state_ok"
