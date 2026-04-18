# RST Slurm Orchestration

This directory contains a dependency-aware Slurm pipeline for large SuperDARN/RST backfills across months or years.

## What This Pipeline Solves

- Scales over long time ranges by using array jobs for per-radar work.
- Allows per-data parameter variation through a manifest CSV.
- Enforces stage dependencies:
  - fit task i -> grid task i using Slurm aftercorr
  - all grid tasks for date+hemisphere -> one map job using afterok
  - all map jobs -> one finalize job using afterok
- Supports mixed hemispheres and radar sets per day.

## Files

- submit_rst_pipeline.sh: Master submitter; builds and submits DAG.
- generate_exact_input_manifest.py: Resolves each input_glob to exact files.
- generate_pipeline_manifest_from_tree.py: Scans raw data tree and creates a manifest.
- generate_rerun_manifest.py: Builds rerun-only manifest from previous run status.
- rst_stage_fit.sbatch: Array stage for RAWACF to FITACF.
- rst_stage_grid.sbatch: Array stage for FITACF to GRD.
- rst_stage_map.sbatch: Per-group stage for combine_grid and map chain.
- rst_stage_finalize.sbatch: Final summary/aggregation stage.
- rst_fitacf_gpu_benchmark.sbatch: Standalone CuPy GPU vs NumPy FITACF benchmark/regression run.
- examples/radar_parameters.csv: Example manifest.

## Manifest Columns

Header must match exactly:

date,radar,hemisphere,input_glob,fit_mode,fitacf_version,tdiff_method,tdiff_value,channel_mode,scan_length_sec,grid_interval_sec,grid_extra_flags,map_model,map_order,map_doping,imf_mode,imf_file,bx,by,bz,extra_map_flags,use_cuda

Field notes:

- date: YYYYMMDD.
- radar: three-letter station code.
- hemisphere: north or south.
- input_glob: shell glob for RAWACF files for that row, or @/path/to/exact.list.
- fit_mode: default, fitacf3, fitacf2, fitex1, fitex2, lmfit1.
- fitacf_version: value passed to -fitacf-version, or -.
- tdiff_method: value passed to -tdiff_method, or -.
- tdiff_value: value passed to -tdiff, or -.
- channel_mode: all, a, or b.
- scan_length_sec: passed to make_grid -tl, or -.
- grid_interval_sec: passed to make_grid -i, or -.
- grid_extra_flags: extra flags for make_grid (space-separated), or -.
- map_model: map_addmodel -model value, or -.
- map_order: map_addmodel -o value, or -.
- map_doping: map_addmodel -d value, or -.
- imf_mode: none, file, fixed, ace, or wind.
- imf_file: required for imf_mode=file, else -.
- bx/by/bz: used for imf_mode=fixed.
- extra_map_flags: extra map_grd flags (space-separated), or -.
- use_cuda: auto, 1/true, or 0/false.

## Submission Example

Run from this directory or reference paths directly:

```bash
chmod +x submit_rst_pipeline.sh

./submit_rst_pipeline.sh \
  --manifest ./examples/radar_parameters.csv \
  --work-root /scratch/$USER/rst_runs \
  --log-root /scratch/$USER/rst_logs \
  --rst-env /path/to/rst_env.sh \
  --fit-partition cpu \
  --grid-partition cpu \
  --map-partition cpu \
  --fit-array-limit 400 \
  --grid-array-limit 400 \
  --account my_account \
  --qos normal
```

By default, submitter-side preprocessing runs generate_exact_input_manifest.py,
which converts each row's input_glob into an exact per-row list file and rewrites
the row to use input_glob=@/path/to/listfile. This makes reruns deterministic and
prevents compute-node glob expansion surprises.

If you want to skip this and use raw globs directly:

```bash
./submit_rst_pipeline.sh --manifest my_manifest.csv --skip-exact-generator
```

To allow empty matches during exact-list generation:

```bash
./submit_rst_pipeline.sh --manifest my_manifest.csv --allow-empty-inputs
```

You can also run the generator directly:

```bash
python3 generate_exact_input_manifest.py \
  --manifest ./examples/radar_parameters.csv \
  --output ./examples/radar_parameters.exact.csv \
  --list-dir ./examples/input_lists
```

## Build Manifest from Raw Data Tree

If you want to start from a multi-year data directory and auto-build a manifest:

```bash
python3 generate_pipeline_manifest_from_tree.py \
  --raw-root /data/superdarn/rawacf \
  --output ./manifests/auto_manifest.csv \
  --hemisphere-map ./examples/radar_hemisphere_map.csv \
  --start-date 20080101 \
  --end-date 20101231 \
  --fit-mode fitacf3 \
  --fitacf-version 3.0 \
  --channel-mode all \
  --scan-length-sec 60 \
  --grid-interval-sec 120 \
  --map-model PSR \
  --map-order 8 \
  --map-doping l \
  --imf-mode none \
  --use-cuda auto
```

Then submit normally:

```bash
./submit_rst_pipeline.sh \
  --manifest ./manifests/auto_manifest.csv \
  --rst-env /shared/env/rst_slurm_env.sh \
  --work-root /scratch/$USER/rst_runs \
  --log-root /scratch/$USER/rst_logs
```

The submitter will resolve exact filenames from the generated input_glob patterns.

## Output Layout

Each submission creates a run directory:

- fits/YYYYMMDD/RADAR/YYYYMMDD.RADAR.fitacf
- grids/YYYYMMDD/RADAR/YYYYMMDD.RADAR.grd
- maps/YYYYMMDD/HEMISPHERE/YYYYMMDD.HEMISPHERE.map
- meta/.../*.done and *.meta markers
- pipeline_summary.txt
- pipeline_summary.csv

## Operational Notes

- If you have very large manifests, split by year and run multiple submitters.
- afterok dependencies mean a failed upstream job prevents downstream jobs from starting.
- To re-run failed rows, fix the issue and resubmit with a manifest containing only failed rows.
- For rows whose outputs already exist and have *.done markers, stages are skipped.
- Grid and map stages wait for dependent files with timeout (default 900s) to handle
  distributed filesystem visibility lag; tune with WAIT_TIMEOUT_SEC and WAIT_INTERVAL_SEC.
- For cluster installation and shared-node environment setup, see docs/guides/slurm-rst-install.md.

## Recovery: Generate Rerun Manifest

After a run completes or partially fails, generate a rerun manifest from run state:

```bash
python3 generate_rerun_manifest.py \
  --run-dir /scratch/$USER/rst_runs/run_YYYYMMDD_HHMMSS \
  --output ./manifests/rerun_manifest.csv
```

Then resubmit only failed rows:

```bash
./submit_rst_pipeline.sh \
  --manifest ./manifests/rerun_manifest.csv \
  --rst-env /shared/env/rst_slurm_env.sh \
  --work-root /scratch/$USER/rst_runs \
  --log-root /scratch/$USER/rst_logs
```

By default, rerun generation flags rows as failed if fit/grid markers are missing, and also
includes entire date+hemisphere groups if map.done is missing. Use --skip-map-check when
you only want row-local fit/grid failures.

## FITACF GPU Benchmark Job

To benchmark CuPy FITACF batch performance against NumPy and validate against the fixed
legacy reference dataset, submit:

```bash
sbatch \
  --export=ALL,ROOT_DIR=/path/to/rst,RST_ENV=/path/to/rst_env.sh,ITERATIONS=40,SCALE=64 \
  ./rst_fitacf_gpu_benchmark.sbatch
```

Output is written to:

- pythonv2/benchmark_results/fitacf_gpu_vs_cpu_slurm.json
