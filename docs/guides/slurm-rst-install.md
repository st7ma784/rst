# RST on Slurm Clusters

This guide describes a practical installation model so every compute node can run RST jobs consistently.

## Deployment Model

Use a shared, read-mostly RST installation mounted on all nodes, plus modulefiles for environment loading.

Recommended layout:

- /opt/rst/versions/rst-<version>/         immutable install trees
- /opt/rst/current -> /opt/rst/versions/... symlink used by modulefile
- /opt/modulefiles/rst/<version>            environment module definitions
- /scratch/$USER/rst_runs                   per-user job outputs

## 1. Build RST Once on a Build Node

Example (adjust for your site):

```bash
# On a build/admin node
sudo mkdir -p /opt/rst/versions
sudo chown -R $USER:$USER /opt/rst/versions

git clone https://github.com/SuperDARN/rst.git /tmp/rst-src
cd /tmp/rst-src

# Optional: checkout a tagged release
# git checkout <tag>

# Build/install prefix
export RST_PREFIX=/opt/rst/versions/rst-2026.04
mkdir -p "$RST_PREFIX"

# Site-specific build command (example)
# Replace with your validated build flow for this repo/site.
make
make install PREFIX="$RST_PREFIX"
```

If your build system does not support PREFIX directly, install to a staging directory and copy the final bin/lib/include/profile files into the versioned prefix.

## 2. Provide Runtime Dependencies Cluster-Wide

Ensure required runtime tools are available on all compute nodes:

- Shell tools: bash, awk, sed, sort, findutils
- Compression tools: gzip, bzip2 (for zcat/bzcat)
- RST dependencies from your local build (e.g., libpng, CDF, optional CUDA runtime)

If CUDA acceleration is desired:

- NVIDIA driver and CUDA runtime must be present on GPU nodes.
- Keep CPU partitions available; RST can run with CUDA disabled.

## 3. Create an Environment Module

Create /opt/modulefiles/rst/2026.04:

```tcl
#%Module1.0
proc ModulesHelp { } {
    puts stderr "Loads Radar Software Toolkit (RST)"
}
module-whatis "RST radar processing toolkit"

set root /opt/rst/versions/rst-2026.04

prepend-path PATH            $root/bin
prepend-path LD_LIBRARY_PATH $root/lib
prepend-path CPATH           $root/include
setenv RSTPATH               $root
```

Optionally expose default via module alias or symlink.

## 4. Verify on Compute Nodes

Run these checks through Slurm on each target partition:

```bash
srun -p cpu -N1 -n1 --pty bash -lc 'module load rst/2026.04 && which make_fit && make_fit --help | head'
srun -p cpu -N1 -n1 --pty bash -lc 'module load rst/2026.04 && which make_grid && make_grid --help | head'
srun -p cpu -N1 -n1 --pty bash -lc 'module load rst/2026.04 && which map_fit && map_fit --help | head'
```

For GPU nodes:

```bash
srun -p gpu -N1 -n1 --pty bash -lc 'module load rst/2026.04 && nvidia-smi'
```

## 5. Create a Job Environment Script for Pipeline Runs

Create a small script consumed by submit_rst_pipeline.sh via --rst-env.

Example: /shared/env/rst_slurm_env.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

module purge
module load rst/2026.04

# Optional: site-specific overrides
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# export RST_DISABLE_CUDA=1
```

Make it executable:

```bash
chmod +x /shared/env/rst_slurm_env.sh
```

## 6. Submit the Pipeline

```bash
cd /home/user/rst/scripts/slurm

./submit_rst_pipeline.sh \
  --manifest ./examples/radar_parameters.csv \
  --rst-env /shared/env/rst_slurm_env.sh \
  --work-root /scratch/$USER/rst_runs \
  --log-root /scratch/$USER/rst_logs \
  --fit-partition cpu \
  --grid-partition cpu \
  --map-partition cpu \
  --account <account> \
  --qos <qos>
```

If you need to auto-build manifests from a raw data tree first, use:

```bash
cd /home/user/rst/scripts/slurm
python3 generate_pipeline_manifest_from_tree.py --help
python3 generate_exact_input_manifest.py --help
python3 generate_rerun_manifest.py --help
```

For failure recovery, create a rerun-only manifest from a prior run:

```bash
python3 generate_rerun_manifest.py \
  --run-dir /scratch/$USER/rst_runs/run_YYYYMMDD_HHMMSS \
  --output /scratch/$USER/rst_runs/rerun_manifest.csv
```

## 7. Operations Checklist

- Pin an RST version for each production campaign.
- Keep modulefiles versioned and immutable.
- Test new RST versions in a staging partition before promoting.
- Capture sbatch logs to durable storage.
- For multi-year runs, submit in chunks (e.g., per year) for easier recovery.

## FITACF GPU Benchmark Validation Job

Use this when you want a reproducible NumPy-vs-CuPy FITACF benchmark plus fixed-reference regression on a GPU partition.

Script:

- `scripts/slurm/rst_fitacf_gpu_benchmark.sbatch`

Example submission:

```bash
cd /home/user/rst/scripts/slurm

sbatch \
  --partition=gpu \
  --export=ALL,ROOT_DIR=/home/user/rst,RST_ENV=/shared/env/rst_slurm_env.sh,ITERATIONS=40,SCALE=64 \
  ./rst_fitacf_gpu_benchmark.sbatch
```

Outputs:

- JSON report written to `pythonv2/benchmark_results/fitacf_gpu_vs_cpu_slurm.json`
- Includes regression pass/fail checks and benchmark speedup metrics.

## Troubleshooting

- Jobs fail with command not found:
  - Confirm --rst-env loads module and PATH is updated.
- Library load errors:
  - Confirm LD_LIBRARY_PATH includes RST lib and required system libs.
- Intermittent missing upstream files:
  - Increase WAIT_TIMEOUT_SEC in job environment to tolerate slower filesystem propagation.
