# Installation Guide

This guide covers installing RST on Linux and macOS systems.

## Quick Install (Docker)

The fastest way to get started is using Docker:

```bash
# Clone the repository
git clone https://github.com/SuperDARN/rst.git
cd rst

# Build the Docker image
docker build -t superdarn-rst .

# Run interactively
docker run -it superdarn-rst

# Or with GPU support (requires nvidia-docker)
docker run --gpus all -it superdarn-rst
```

## Linux Installation

### Prerequisites

Install required packages:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    gcc \
    make \
    libnetcdf-dev \
    libhdf5-dev \
    libpng-dev \
    libx11-dev \
    zlib1g-dev

# Fedora/RHEL
sudo dnf install -y \
    gcc \
    make \
    netcdf-devel \
    hdf5-devel \
    libpng-devel \
    libX11-devel \
    zlib-devel
```

### Build RST

```bash
# Clone repository
git clone https://github.com/SuperDARN/rst.git
cd rst

# Source the environment
source .profile.bash

# Build the toolkit
cd build
make

# Verify installation
make_fit --help
```

### Environment Setup

Add to your `~/.bashrc`:

```bash
# RST Environment
export RSTPATH=/path/to/rst
source $RSTPATH/.profile.bash
```

## macOS Installation

### Prerequisites

Install Xcode Command Line Tools and Homebrew packages:

```bash
# Install Xcode CLI tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install netcdf hdf5 libpng
```

### Build RST

```bash
# Clone and build
git clone https://github.com/SuperDARN/rst.git
cd rst
source .profile.bash
cd build && make
```

## CUDA Installation (Optional)

For GPU acceleration, see [CUDA Setup Tutorial](cuda-setup.md).

### Quick CUDA Setup

```bash
# Verify CUDA is installed
nvcc --version

# Build CUDA modules
./scripts/build_all_cuda_modules.sh

# Test CUDA functionality
cd CUDArst && make test
```

## Verifying Installation

Run the verification script:

```bash
# Check all components
./scripts/ecosystem_validation.sh

# Or test individual tools
make_fit --help
make_grid --help
map_plot --help
```

## Common Issues

### "Command not found"

Ensure environment is sourced:
```bash
source /path/to/rst/.profile.bash
```

### Library errors

Check library paths:
```bash
echo $LD_LIBRARY_PATH
# Should include: /path/to/rst/lib
```

### Build failures

Check compiler version:
```bash
gcc --version  # Requires GCC 7+
```

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Run your first commands
- [CUDA Setup](cuda-setup.md) - Enable GPU acceleration
- [First Processing](first-processing.md) - Process real data
