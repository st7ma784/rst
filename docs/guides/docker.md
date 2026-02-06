# Docker Guide

Complete guide for running RST in Docker containers.

## Quick Start

```bash
# Build image
docker build -t superdarn-rst .

# Run interactively
docker run -it superdarn-rst

# Run with GPU support
docker run --gpus all -it superdarn-rst
```

## Docker Images

### Available Tags

| Tag | Description | Size |
|-----|-------------|------|
| `latest` | Latest stable release | ~2.5 GB |
| `cuda` | CUDA-enabled build | ~4.5 GB |
| `slim` | Minimal CPU-only | ~1.2 GB |
| `dev` | Development version | ~3.0 GB |

### Pulling Images

```bash
# From Docker Hub (when available)
docker pull superdarn/rst:latest

# Build locally
docker build -t superdarn-rst .

# Build CUDA version
docker build -f Dockerfile.cuda -t superdarn-rst:cuda .
```

## Running Containers

### Interactive Mode

```bash
# Basic interactive shell
docker run -it superdarn-rst

# With your data mounted
docker run -v /path/to/data:/data -it superdarn-rst

# With GPU support
docker run --gpus all -v /path/to/data:/data -it superdarn-rst
```

### Processing Data

```bash
# Process single file
docker run -v $(pwd):/data superdarn-rst \
    make_fit /data/input.rawacf > output.fitacf

# Complete pipeline
docker run -v $(pwd):/data superdarn-rst bash -c "
    make_fit /data/input.rawacf > /data/output.fitacf && \
    make_grid /data/output.fitacf > /data/output.grid && \
    map_grd /data/output.grid > /data/output.map
"
```

### Batch Processing

```bash
# Process all files in directory
docker run -v $(pwd)/data:/data superdarn-rst bash -c "
    for f in /data/*.rawacf; do
        make_fit \$f > \${f%.rawacf}.fitacf
    done
"
```

## GPU Support

### Prerequisites

1. **NVIDIA Driver** (450.0+) on host
2. **nvidia-container-toolkit** installed

### Installing nvidia-container-toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Running with GPU

```bash
# Use all GPUs
docker run --gpus all -it superdarn-rst:cuda

# Use specific GPU
docker run --gpus '"device=0"' -it superdarn-rst:cuda

# Verify GPU access
docker run --gpus all superdarn-rst:cuda nvidia-smi
```

## Docker Compose

### Basic Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  rst:
    build: .
    volumes:
      - ./data:/data
      - ./output:/output
    environment:
      - RST_VERBOSE=1
    
  rst-cuda:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    volumes:
      - ./data:/data
      - ./output:/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Usage

```bash
# Start services
docker-compose up -d

# Run command
docker-compose exec rst make_fit /data/input.rawacf

# With GPU
docker-compose exec rst-cuda make_fit /data/input.rawacf

# Stop services
docker-compose down
```

## Building Custom Images

### Dockerfile

```dockerfile
# Dockerfile
FROM ubuntu:22.04 AS base

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make \
    libnetcdf-dev \
    libhdf5-dev \
    libpng-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy RST source
COPY . /opt/rst
WORKDIR /opt/rst

# Build RST
RUN source .profile.bash && \
    cd build && make

# Set environment
ENV RSTPATH=/opt/rst
ENV PATH=$RSTPATH/build/bin:$PATH

# Default command
CMD ["/bin/bash"]
```

### CUDA Dockerfile

```dockerfile
# Dockerfile.cuda
FROM nvidia/cuda:12.0-devel-ubuntu22.04 AS base

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make \
    libnetcdf-dev \
    libhdf5-dev \
    libpng-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy RST source
COPY . /opt/rst
WORKDIR /opt/rst

# Build RST with CUDA
RUN source .profile.bash && \
    cd build && make && \
    ./scripts/build_all_cuda_modules.sh

# Set environment
ENV RSTPATH=/opt/rst
ENV PATH=$RSTPATH/build/bin:$PATH
ENV LD_LIBRARY_PATH=$RSTPATH/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

CMD ["/bin/bash"]
```

### Multi-stage Build (Smaller Image)

```dockerfile
# Dockerfile.slim
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential gcc make \
    libnetcdf-dev libhdf5-dev

COPY . /opt/rst
WORKDIR /opt/rst
RUN source .profile.bash && cd build && make

# Runtime image
FROM ubuntu:22.04 AS runtime

RUN apt-get update && apt-get install -y \
    libnetcdf19 libhdf5-103 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/rst/build/bin /usr/local/bin/
COPY --from=builder /opt/rst/lib /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

CMD ["/bin/bash"]
```

## Volume Mounts

### Data Directories

```bash
# Mount single directory
docker run -v /host/data:/data superdarn-rst

# Mount multiple directories
docker run \
    -v /host/input:/input:ro \
    -v /host/output:/output \
    superdarn-rst
```

### Configuration Files

```bash
# Mount custom configuration
docker run \
    -v /host/config:/opt/rst/config:ro \
    superdarn-rst
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RST_DISABLE_CUDA` | Disable CUDA | 0 |
| `RST_VERBOSE` | Verbose output | 0 |
| `RST_CUDA_DEVICE` | GPU device ID | 0 |

```bash
# Set environment variables
docker run \
    -e RST_VERBOSE=1 \
    -e RST_CUDA_DEVICE=0 \
    superdarn-rst
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/docker.yml
name: Docker Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build image
        run: docker build -t superdarn-rst .
      
      - name: Test
        run: |
          docker run superdarn-rst make_fit --help
          docker run superdarn-rst ./scripts/ecosystem_validation.sh
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test

build:
  stage: build
  script:
    - docker build -t superdarn-rst .
  
test:
  stage: test
  script:
    - docker run superdarn-rst ./scripts/ecosystem_validation.sh
```

## Troubleshooting

### "nvidia-smi not found"

```bash
# Ensure nvidia-container-toolkit is installed
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Out of Memory

```bash
# Limit memory usage
docker run --memory=4g superdarn-rst

# Check usage
docker stats
```

### Permission Denied

```bash
# Run with user mapping
docker run -u $(id -u):$(id -g) -v $(pwd):/data superdarn-rst
```

### Slow Performance

```bash
# Use volume instead of bind mount (macOS/Windows)
docker volume create rst-data
docker run -v rst-data:/data superdarn-rst

# Enable host networking (Linux)
docker run --network host superdarn-rst
```
