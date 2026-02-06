# Troubleshooting Guide

Solutions for common RST issues.

## Quick Diagnostics

```bash
# Run comprehensive check
./scripts/ecosystem_validation.sh

# Check specific components
make_fit --help         # Processing tools
nvidia-smi              # GPU status
echo $RSTPATH           # Environment
```

---

## Installation Issues

### "Command not found"

**Problem:** RST commands not recognized.

**Solution:**
```bash
# Ensure environment is sourced
source /path/to/rst/.profile.bash

# Verify PATH
echo $PATH | grep -q rst && echo "OK" || echo "RST not in PATH"

# Add to bashrc for persistence
echo 'source /path/to/rst/.profile.bash' >> ~/.bashrc
```

### Build Failures

**Problem:** `make` fails with errors.

**Solutions:**

1. **Missing dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential gcc make \
       libnetcdf-dev libhdf5-dev libpng-dev zlib1g-dev
   
   # Check versions
   gcc --version   # Needs 7+
   make --version
   ```

2. **Wrong compiler version:**
   ```bash
   # Install specific version
   sudo apt-get install gcc-11
   export CC=gcc-11
   make clean && make
   ```

3. **Library path issues:**
   ```bash
   # Find libraries
   ldconfig -p | grep netcdf
   
   # Set paths if needed
   export CPATH=/usr/include/hdf5/serial
   export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial
   ```

### Library Errors

**Problem:** `error while loading shared libraries`

**Solution:**
```bash
# Check library path
echo $LD_LIBRARY_PATH

# Add RST libraries
export LD_LIBRARY_PATH=$RSTPATH/lib:$LD_LIBRARY_PATH

# Update ldconfig
sudo ldconfig

# Verify
ldd $(which make_fit) | grep "not found"
```

---

## CUDA Issues

### "CUDA driver version insufficient"

**Problem:** Driver mismatch with CUDA toolkit.

**Solution:**
```bash
# Check versions
nvidia-smi  # Shows driver version
nvcc --version  # Shows toolkit version

# Update driver
sudo apt-get update
sudo apt-get install nvidia-driver-535

# Reboot required
sudo reboot
```

### "No CUDA-capable device"

**Problem:** GPU not detected.

**Solutions:**

1. **Check hardware:**
   ```bash
   lspci | grep -i nvidia
   # Should show NVIDIA GPU
   ```

2. **Check driver:**
   ```bash
   nvidia-smi
   # Should show GPU info
   ```

3. **Reinstall driver:**
   ```bash
   sudo apt-get purge nvidia-*
   sudo apt-get install nvidia-driver-535
   sudo reboot
   ```

### CUDA Build Errors

**Problem:** CUDA compilation fails.

**Solutions:**

1. **Architecture mismatch:**
   ```bash
   # Find your GPU architecture
   nvidia-smi --query-gpu=compute_cap --format=csv
   
   # Build with correct arch
   CUDA_ARCH=sm_75 make -f makefile.cuda
   ```

2. **GCC version incompatibility:**
   ```bash
   # CUDA 12.x needs GCC <= 12
   sudo apt-get install gcc-11 g++-11
   export CC=gcc-11
   export CXX=g++-11
   make -f makefile.cuda
   ```

### Poor CUDA Performance

**Problem:** CUDA slower than expected.

**Diagnosis:**
```bash
# Profile kernel execution
nvprof make_fit input.rawacf > /dev/null

# Check GPU utilization
nvidia-smi dmon -s u
```

**Solutions:**

1. **Data too small:**
   ```bash
   # Check data size
   dmapdump input.rawacf | grep -c "^"
   
   # Batch multiple files for better performance
   cat file1.rawacf file2.rawacf | make_fit > output.fitacf
   ```

2. **Memory transfer bottleneck:**
   ```bash
   # Enable memory pooling
   export RST_MEMORY_POOL=1
   ```

---

## Processing Issues

### "No output" or Empty Results

**Problem:** Processing completes but no data output.

**Diagnosis:**
```bash
# Check input file
dmapdump input.rawacf | head -20

# Run with verbose
RST_VERBOSE=1 make_fit input.rawacf > output.fitacf 2>&1
```

**Solutions:**

1. **Wrong file format:**
   ```bash
   # Check file type
   file input.rawacf
   
   # Try old format flag
   make_fit -old input.rawacf > output.fitacf
   ```

2. **Corrupted input:**
   ```bash
   # Validate file
   dmapdump input.rawacf > /dev/null
   # Check exit code
   echo $?  # 0 = OK
   ```

### Incorrect Results

**Problem:** Output values seem wrong.

**Diagnosis:**
```bash
# Compare CPU vs CUDA
RST_DISABLE_CUDA=1 make_fit input.rawacf > cpu_result.fitacf
make_fit input.rawacf > cuda_result.fitacf

# Compare
diff cpu_result.fitacf cuda_result.fitacf
```

**Solutions:**

1. **Version mismatch:**
   ```bash
   # Specify algorithm version
   make_fit -fitacf-version 3.0 input.rawacf
   ```

2. **CUDA numerical differences:**
   ```
   Small differences (< 0.1%) are expected due to floating point
   If differences are large, try forcing CPU:
   export RST_DISABLE_CUDA=1
   ```

### Slow Processing

**Problem:** Processing takes too long.

**Diagnosis:**
```bash
# Time the operation
time make_fit input.rawacf > output.fitacf

# Check if CUDA is being used
RST_VERBOSE=1 make_fit input.rawacf 2>&1 | grep -i cuda
```

**Solutions:**

1. **CUDA not enabled:**
   ```bash
   # Verify CUDA is working
   nvidia-smi
   ./CUDArst/test_integration
   ```

2. **System resource contention:**
   ```bash
   # Check system load
   top -bn1 | head -20
   
   # Check GPU usage
   nvidia-smi
   ```

---

## Memory Issues

### Out of Memory (RAM)

**Problem:** Process killed or memory errors.

**Solutions:**

1. **Process in smaller batches:**
   ```bash
   # Split large files
   split -l 10000 large_file.rawacf chunk_
   
   # Process chunks
   for f in chunk_*; do
       make_fit $f >> output.fitacf
   done
   ```

2. **Increase swap:**
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Out of GPU Memory

**Problem:** CUDA out of memory error.

**Solutions:**

1. **Reduce batch size:**
   ```bash
   export RST_BATCH_SIZE=500
   ```

2. **Use different GPU:**
   ```bash
   export CUDA_VISIBLE_DEVICES=1
   ```

3. **Fall back to CPU:**
   ```bash
   export RST_DISABLE_CUDA=1
   ```

---

## Docker Issues

### Container Won't Start

**Problem:** Container exits immediately.

**Diagnosis:**
```bash
docker logs container_name
```

**Solutions:**

1. **Missing GPU runtime:**
   ```bash
   # Install nvidia-container-toolkit
   sudo apt-get install nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Permission issues:**
   ```bash
   # Run with user mapping
   docker run -u $(id -u):$(id -g) ...
   ```

### GPU Not Available in Container

**Problem:** `nvidia-smi` fails in container.

**Solution:**
```bash
# Verify docker GPU support
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi

# If that fails, reinstall nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Getting Help

### Collect Diagnostic Info

```bash
# Create diagnostic report
{
    echo "=== System Info ==="
    uname -a
    cat /etc/os-release
    
    echo "=== RST Version ==="
    cat $RSTPATH/.rst.version
    
    echo "=== Environment ==="
    env | grep -E "RST|CUDA|PATH"
    
    echo "=== GPU Info ==="
    nvidia-smi 2>/dev/null || echo "No NVIDIA GPU"
    
    echo "=== Library Info ==="
    ldd $(which make_fit) 2>/dev/null
    
} > diagnostic_report.txt
```

### Report an Issue

1. Run diagnostics above
2. Include error messages
3. Provide sample data if possible
4. Submit at: https://github.com/SuperDARN/rst/issues
