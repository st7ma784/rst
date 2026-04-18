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

require_bin make_fit

line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [[ -z "$line" ]]; then
  echo "No manifest row for task id $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

IFS=',' read -r date radar hemisphere input_glob fit_mode fitacf_version tdiff_method tdiff_value channel_mode scan_length_sec grid_interval_sec grid_extra_flags map_model map_order map_doping imf_mode imf_file bx by bz extra_map_flags use_cuda <<< "$line"

if [[ -z "$date" || -z "$radar" ]]; then
  echo "Invalid manifest row: $line" >&2
  exit 1
fi

fit_dir="$RUN_DIR/fits/$date/$radar"
meta_dir="$RUN_DIR/meta/$date/$radar"
mkdir -p "$fit_dir" "$meta_dir"

out_fit="$fit_dir/${date}.${radar}.fitacf"
state_ok="$meta_dir/fit.done"

if [[ -f "$state_ok" && -s "$out_fit" ]]; then
  echo "Fit stage already complete for $date/$radar"
  exit 0
fi

shopt -s nullglob
files=()
if [[ "$input_glob" == @* ]]; then
  list_file="${input_glob#@}"
  wait_for_file "$list_file"
  while IFS= read -r p; do
    [[ -z "$p" ]] && continue
    files+=("$p")
  done < "$list_file"
else
  files=( $input_glob )
fi
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No input files matched for $date/$radar using glob: $input_glob" >&2
  exit 2
fi

IFS=$'\n' sorted=( $(printf '%s\n' "${files[@]}" | sort) )
unset IFS

fit_args=()
case "$fit_mode" in
  fitacf3|fitacf2|fitex1|fitex2|lmfit1)
    fit_args+=("-$fit_mode")
    ;;
  lmfit2)
    echo "lmfit2 requires make_lmfit2; not supported in this default fit stage" >&2
    exit 3
    ;;
  default|"")
    ;;
  *)
    echo "Unknown fit_mode '$fit_mode'" >&2
    exit 3
    ;;
esac

if [[ -n "$fitacf_version" && "$fitacf_version" != "-" ]]; then
  fit_args+=("-fitacf-version" "$fitacf_version")
fi

if [[ -n "$tdiff_method" && "$tdiff_method" != "-" ]]; then
  fit_args+=("-tdiff_method" "$tdiff_method")
fi

if [[ -n "$tdiff_value" && "$tdiff_value" != "-" ]]; then
  fit_args+=("-tdiff" "$tdiff_value")
fi

if [[ -n "$use_cuda" && "$use_cuda" != "auto" ]]; then
  if [[ "$use_cuda" == "0" || "$use_cuda" == "false" ]]; then
    export RST_DISABLE_CUDA=1
  else
    unset RST_DISABLE_CUDA
  fi
fi

stream_input() {
  local f
  for f in "${sorted[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "Missing input file: $f" >&2
      return 4
    fi
    case "$f" in
      *.bz2) bzcat "$f" ;;
      *.gz) zcat "$f" ;;
      *) cat "$f" ;;
    esac
  done
}

echo "[$date/$radar] FIT from ${#sorted[@]} rawacf files"
stream_input | make_fit "${fit_args[@]}" > "$out_fit"

tmp_meta="$meta_dir/fit.meta.tmp"
cat > "$tmp_meta" <<EOF
stage=fit
date=$date
radar=$radar
hemisphere=$hemisphere
input_count=${#sorted[@]}
output_fit=$out_fit
EOF
mv "$tmp_meta" "$meta_dir/fit.meta"

touch "$state_ok"
