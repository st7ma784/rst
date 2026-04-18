#!/usr/bin/env bash
#SBATCH --ntasks=1

set -euo pipefail

if [[ -z "${MANIFEST:-}" || -z "${RUN_DIR:-}" ]]; then
  echo "MANIFEST and RUN_DIR must be exported into the job environment" >&2
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

require_bin combine_grid
require_bin map_grd
require_bin map_addhmb
require_bin map_addimf
require_bin map_addmodel
require_bin map_fit

header_skipped=0
rows=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  if [[ "$header_skipped" -eq 0 ]]; then
    header_skipped=1
    continue
  fi
  rows+=("$line")
done < "$MANIFEST"

if [[ ${#rows[@]} -eq 0 ]]; then
  echo "Group manifest has no rows: $MANIFEST" >&2
  exit 1
fi

first="${rows[0]}"
IFS=',' read -r date radar hemisphere input_glob fit_mode fitacf_version tdiff_method tdiff_value channel_mode scan_length_sec grid_interval_sec grid_extra_flags map_model map_order map_doping imf_mode imf_file bx by bz extra_map_flags use_cuda <<< "$first"

hemi_lc="$(echo "${hemisphere:-north}" | tr '[:upper:]' '[:lower:]')"
map_dir="$RUN_DIR/maps/$date/$hemi_lc"
stage_dir="$map_dir/stage"
meta_dir="$RUN_DIR/meta/$date/$hemi_lc"
mkdir -p "$map_dir" "$stage_dir" "$meta_dir"

combined_grid="$map_dir/${date}.${hemi_lc}.grd"
final_map="$map_dir/${date}.${hemi_lc}.map"
state_ok="$meta_dir/map.done"

if [[ -f "$state_ok" && -s "$final_map" ]]; then
  echo "Map stage already complete for $date/$hemi_lc"
  exit 0
fi

grid_files=()
for row in "${rows[@]}"; do
  IFS=',' read -r r_date r_radar r_hemi r_input_glob r_fit_mode r_fitacf_version r_tdiff_method r_tdiff_value r_channel_mode r_scan_length_sec r_grid_interval_sec r_grid_extra_flags r_map_model r_map_order r_map_doping r_imf_mode r_imf_file r_bx r_by r_bz r_extra_map_flags r_use_cuda <<< "$row"

  g="$RUN_DIR/grids/$r_date/$r_radar/${r_date}.${r_radar}.grd"
  if [[ ! -s "$g" ]]; then
    wait_for_file "$g" || {
      echo "Missing grid file for $r_date/$r_radar: $g" >&2
      exit 2
    }
  fi
  grid_files+=("$g")
done

echo "[$date/$hemi_lc] Combining ${#grid_files[@]} grid files"
combine_grid -vb "${grid_files[@]}" > "$combined_grid"

map_grd_args=()
if [[ "$hemi_lc" == "south" || "$hemi_lc" == "s" || "$hemi_lc" == "sh" ]]; then
  map_grd_args+=("-sh")
fi
if [[ -n "$extra_map_flags" && "$extra_map_flags" != "-" ]]; then
  # shellcheck disable=SC2206
  map_extra=( $extra_map_flags )
  map_grd_args+=("${map_extra[@]}")
fi

empty_map="$stage_dir/${date}.${hemi_lc}.empty.map"
hmb_map="$stage_dir/${date}.${hemi_lc}.hmb.map"
imf_map="$stage_dir/${date}.${hemi_lc}.imf.map"
model_map="$stage_dir/${date}.${hemi_lc}.model.map"

map_grd "${map_grd_args[@]}" "$combined_grid" > "$empty_map"
map_addhmb "$empty_map" > "$hmb_map"

addimf_args=()
imf_mode_lc="$(echo "${imf_mode:-none}" | tr '[:upper:]' '[:lower:]')"
case "$imf_mode_lc" in
  file)
    if [[ -z "$imf_file" || "$imf_file" == "-" || ! -f "$imf_file" ]]; then
      echo "imf_mode=file requires existing imf_file, got: $imf_file" >&2
      exit 3
    fi
    addimf_args+=("-if" "$imf_file")
    ;;
  fixed)
    addimf_args+=("-bx" "${bx:-0}" "-by" "${by:-0}" "-bz" "${bz:-0}")
    ;;
  ace)
    addimf_args+=("-ace")
    ;;
  wind)
    addimf_args+=("-wind")
    ;;
  none|""|-)
    ;;
  *)
    echo "Unknown imf_mode: $imf_mode" >&2
    exit 3
    ;;
esac

map_addimf "${addimf_args[@]}" "$hmb_map" > "$imf_map"

model_args=()
if [[ -n "$map_order" && "$map_order" != "-" ]]; then
  model_args+=("-o" "$map_order")
fi
if [[ -n "$map_doping" && "$map_doping" != "-" ]]; then
  model_args+=("-d" "$map_doping")
fi
if [[ -n "$map_model" && "$map_model" != "-" ]]; then
  model_args+=("-model" "$map_model")
fi

map_addmodel "${model_args[@]}" "$imf_map" > "$model_map"
map_fit "$model_map" > "$final_map"

tmp_meta="$meta_dir/map.meta.tmp"
cat > "$tmp_meta" <<EOF
stage=map
date=$date
hemisphere=$hemi_lc
grid_count=${#grid_files[@]}
combined_grid=$combined_grid
output_map=$final_map
EOF
mv "$tmp_meta" "$meta_dir/map.meta"

touch "$state_ok"
