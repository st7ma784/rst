# Updating RST

Guide for updating RST to new versions.

## Update Overview

### Before Updating

1. **Check release notes** for breaking changes
2. **Backup current installation and data**
3. **Test in non-production first** if possible
4. **Schedule maintenance window** for production

### Update Paths

| From | To | Method |
|------|-----|--------|
| Any | Latest | Full update |
| Minor version | Next minor | In-place update |
| Major version | Next major | Clean install recommended |

---

## Quick Update

### Git-based Installation

```bash
cd /opt/rst

# Backup current version
cp -r . /backup/rst_$(date +%Y%m%d)

# Fetch updates
git fetch origin

# Check what's new
git log HEAD..origin/main --oneline

# Apply update
git pull origin main

# Rebuild
source .profile.bash
cd build && make clean && make

# Rebuild CUDA modules
../scripts/build_all_cuda_modules.sh

# Verify
./scripts/ecosystem_validation.sh
```

### Docker Update

```bash
# Pull new image
docker pull superdarn-rst:latest

# Stop existing container
docker stop rst-processor

# Remove old container
docker rm rst-processor

# Start with new image
docker run -d --name rst-processor \
    --gpus all \
    -v /data:/data \
    superdarn-rst:latest

# Verify
docker exec rst-processor make_fit --help
```

---

## Detailed Update Process

### Step 1: Preparation

```bash
# Create backup
./scripts/backup_full.sh

# Check current version
cat .rst.version

# Document current settings
env | grep RST > /tmp/rst_env_backup.txt

# Stop services
sudo systemctl stop rst-processor
```

### Step 2: Update Code

```bash
cd /opt/rst

# Clean build artifacts
make -C build clean
rm -f lib/*.a lib/*.so

# Update from git
git stash  # Save local changes
git pull origin main
git stash pop  # Restore local changes (if compatible)
```

### Step 3: Rebuild

```bash
# Source environment
source .profile.bash

# Build core
cd build
make

# Build CUDA modules (if using GPU)
cd ..
./scripts/build_all_cuda_modules.sh
```

### Step 4: Verify

```bash
# Run tests
./scripts/ecosystem_validation.sh

# Test processing
make_fit test_data/sample.rawacf > /tmp/test_output.fitacf
echo $?  # Should be 0

# Compare with known-good output
diff /tmp/test_output.fitacf test_data/expected_output.fitacf
```

### Step 5: Restart Services

```bash
# Restart service
sudo systemctl start rst-processor

# Check status
sudo systemctl status rst-processor

# Monitor logs
journalctl -u rst-processor -f
```

---

## Rollback Procedure

### If Update Fails

```bash
# Stop services
sudo systemctl stop rst-processor

# Restore from backup
rm -rf /opt/rst
mv /backup/rst_YYYYMMDD /opt/rst

# Rebuild
cd /opt/rst
source .profile.bash
cd build && make

# Restart
sudo systemctl start rst-processor

# Verify
./scripts/ecosystem_validation.sh
```

### Docker Rollback

```bash
# List available images
docker images superdarn-rst

# Rollback to previous tag
docker stop rst-processor
docker rm rst-processor
docker run -d --name rst-processor \
    superdarn-rst:previous-tag
```

---

## Version-Specific Updates

### Updating CUDA Components

```bash
# Check current CUDA version
nvcc --version

# If CUDA toolkit updated, rebuild all CUDA modules
cd /opt/rst
./scripts/build_all_cuda_modules.sh --clean

# Test CUDA functionality
cd CUDArst
make test
./test_integration
```

### Updating Python Bindings

```bash
cd pythonv2

# Uninstall old version
pip uninstall superdarn-gpu

# Install new version
pip install -e .

# Verify
python -c "import superdarn_gpu; print(superdarn_gpu.__version__)"
```

### Configuration Migration

Some updates may require configuration changes:

```bash
# Compare configurations
diff /opt/rst/config/default.cfg /opt/rst/config/default.cfg.new

# Merge changes
# (manually review and update your configuration)

# Validate new config
./scripts/validate_config.sh
```

---

## Automated Updates

### Update Script

```bash
#!/bin/bash
# update_rst.sh - Automated RST update

set -e  # Exit on error

LOG_FILE="/var/log/rst/update_$(date +%Y%m%d).log"
exec > >(tee -a $LOG_FILE) 2>&1

echo "=== RST Update Started: $(date) ==="

# Backup
echo "Creating backup..."
./scripts/backup_full.sh

# Stop services
echo "Stopping services..."
sudo systemctl stop rst-processor

# Update
echo "Pulling updates..."
cd /opt/rst
git fetch origin
git pull origin main

# Build
echo "Building..."
source .profile.bash
cd build && make clean && make
cd .. && ./scripts/build_all_cuda_modules.sh

# Test
echo "Testing..."
./scripts/ecosystem_validation.sh

# Restart
echo "Restarting services..."
sudo systemctl start rst-processor

echo "=== RST Update Complete: $(date) ==="
```

### Scheduled Updates

```bash
# Check for updates weekly, apply monthly (example)

# /etc/cron.d/rst-update

# Check for updates every Sunday
0 6 * * 0 root /opt/rst/scripts/check_updates.sh

# Apply updates first Sunday of month (during maintenance window)
0 2 1-7 * 0 root /opt/rst/scripts/update_rst.sh
```

---

## Health Checks Post-Update

### Validation Checklist

```bash
#!/bin/bash
# post_update_check.sh

echo "Running post-update checks..."

# 1. Version check
echo "1. Version:"
cat /opt/rst/.rst.version

# 2. Binary check
echo "2. Binaries:"
for cmd in make_fit make_grid map_grd; do
    which $cmd && echo "  $cmd: OK" || echo "  $cmd: MISSING"
done

# 3. Library check
echo "3. Libraries:"
ldd $(which make_fit) | grep "not found" && echo "  MISSING" || echo "  OK"

# 4. CUDA check
echo "4. CUDA:"
nvidia-smi &>/dev/null && \
    ./CUDArst/test_integration && echo "  OK" || echo "  NOT AVAILABLE"

# 5. Processing test
echo "5. Processing:"
make_fit test_data/sample.rawacf > /dev/null 2>&1 && \
    echo "  OK" || echo "  FAILED"

# 6. Service check
echo "6. Service:"
systemctl is-active rst-processor && echo "  Running" || echo "  Not running"

echo "Post-update checks complete."
```

---

## Troubleshooting Updates

### Build Fails After Update

```bash
# Clean and rebuild
make -C build clean
make -C build

# If still fails, check dependencies
./scripts/check_dependencies.sh

# May need to update dependencies
sudo apt-get update && sudo apt-get upgrade
```

### Tests Fail After Update

```bash
# Run verbose tests
RST_VERBOSE=1 ./scripts/ecosystem_validation.sh

# Check for configuration changes
git diff HEAD~1 config/

# Review release notes for breaking changes
cat CHANGELOG.md | head -100
```

### Service Won't Start

```bash
# Check logs
journalctl -u rst-processor -n 100

# Verify environment
sudo -u rst env | grep RST

# Test manually
sudo -u rst make_fit --help
```
