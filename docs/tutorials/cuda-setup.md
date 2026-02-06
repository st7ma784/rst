# CUDA Setup Guide

Enable GPU acceleration for significant performance improvements (up to 16x speedup).

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with Compute Capability 3.5+
- Recommended: GTX 1060 or better, or any Tesla/Quadro card

### Software Requirements

- NVIDIA Driver 450.0+
- CUDA Toolkit 11.0+ (12.x recommended)
- GCC compatible with your CUDA version

## Check Current Setup

```bash
# Check for NVIDIA GPU
lspci | grep -i nvidia

# Check driver version
nvidia-smi

# Check CUDA installation
nvcc --version
```

## Installing CUDA

### Ubuntu/Debian

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-12-0

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Fedora/RHEL

```bash
# Enable NVIDIA repo
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install
sudo dnf install cuda-toolkit-12-0
```

### Verify Installation

```bash
# Compile and run sample
nvcc --version
nvidia-smi

# Test CUDA
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

## Building RST with CUDA

### Quick Build

```bash
cd /path/to/rst

# Build all CUDA modules
./scripts/build_all_cuda_modules.sh

# Or build specific module
cd codebase/superdarn/src.lib/tk/fitacf_v3.0
make -f makefile.cuda
```

### Manual Build

```bash
# Set CUDA paths
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Build CUDA library
cd CUDArst
make clean
make

# Run tests
make test
```

### Build Options

```makefile
# In makefile.cuda or Makefile
CUDA_ARCH ?= sm_75      # Target GPU architecture
CUDA_DEBUG ?= 0         # Enable debug symbols
CUDA_PROFILE ?= 0       # Enable profiling
```

Common architectures:
| GPU Family | Architecture |
|------------|--------------|
| GTX 10xx | sm_61 |
| RTX 20xx | sm_75 |
| RTX 30xx | sm_86 |
| RTX 40xx | sm_89 |
| Tesla V100 | sm_70 |
| A100 | sm_80 |

## Verifying CUDA Integration

### Run Tests

```bash
cd CUDArst

# Unit tests
./tests/run_tests.sh

# Integration test
./test_integration

# Real-world data test
./test_real_world_mixing
```

### Benchmark Performance

```bash
# Run performance comparison
cd scripts
python compare_performance.py

# Or use benchmark script
./scripts/comprehensive_cuda_performance.sh
```

### Expected Results

```
CUDA Performance Test Results:
============================
Data Copying:      16.1x speedup
Power Computation: 12.5x speedup
Statistical Ops:   12.4x speedup
ACF Processing:    8.3x speedup

All tests PASSED
Numerical accuracy: < 0.1% difference
```

## Using CUDA in RST

### Automatic Detection

RST automatically uses CUDA when available:

```bash
# CUDA used automatically
make_fit input.rawacf > output.fitacf
```

### Manual Control

```bash
# Disable CUDA (use CPU only)
export RST_DISABLE_CUDA=1
make_fit input.rawacf > output.fitacf

# Re-enable CUDA
unset RST_DISABLE_CUDA

# Verbose mode (shows CUDA usage)
export RST_VERBOSE=1
make_fit input.rawacf > output.fitacf
```

### Python Interface

```python
from superdarn_gpu import FitACF, CUDAConfig

# Check CUDA availability
config = CUDAConfig()
print(f"CUDA available: {config.cuda_available}")
print(f"GPU: {config.device_name}")

# Force CPU mode
processor = FitACF(use_cuda=False)

# Use CUDA with specific device
processor = FitACF(use_cuda=True, device_id=0)
```

## Docker with CUDA

### Prerequisites

Install nvidia-container-toolkit:

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

### Run with GPU

```bash
# Build image
docker build -t superdarn-rst:cuda .

# Run with GPU access
docker run --gpus all -it superdarn-rst:cuda

# Specify GPU
docker run --gpus '"device=0"' -it superdarn-rst:cuda
```

## Troubleshooting

### "CUDA driver version insufficient"

Update NVIDIA driver:
```bash
sudo apt-get install nvidia-driver-535
sudo reboot
```

### "No CUDA-capable device"

Check GPU detection:
```bash
lspci | grep -i nvidia
nvidia-smi
```

### Build errors

Verify CUDA/GCC compatibility:
```bash
nvcc --version
gcc --version

# CUDA 12.x requires GCC <= 12
# CUDA 11.x requires GCC <= 10
```

### Runtime errors

Check library paths:
```bash
echo $LD_LIBRARY_PATH
ldconfig -p | grep cuda
```

### Performance issues

Check GPU utilization:
```bash
nvidia-smi dmon -s u
```

## Advanced Configuration

### Multi-GPU Setup

```bash
# List GPUs
nvidia-smi -L

# Select specific GPU
export CUDA_VISIBLE_DEVICES=0

# Use multiple GPUs (if supported)
export CUDA_VISIBLE_DEVICES=0,1
```

### Memory Management

```bash
# Limit GPU memory
export RST_CUDA_MEMORY_LIMIT=4096  # MB

# Enable memory pooling
export RST_CUDA_MEMORY_POOL=1
```

## Next Steps

- [First Processing](first-processing.md) - Process data with CUDA
- [Architecture Guide](../architecture/cuda-implementation.md) - Technical details
- [Benchmarks](../guides/benchmarks.md) - Performance analysis
