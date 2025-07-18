# SUPERDARN RST Unified Docker Container
# ====================================
# Multi-stage build supporting both CPU and CUDA configurations
# Includes all dependencies for SUPERDARN processing and CUDA acceleration

ARG CUDA_VERSION=12.0
ARG UBUNTU_VERSION=22.04

# =============================================================================
# Stage 1: Base Dependencies
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-9 \
    gcc-10 \
    gcc-11 \
    g++ \
    gfortran \
    make \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libfftw3-dev \
    libgsl-dev \
    zlib1g-dev \
    libbz2-dev \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    pkg-config \
    valgrind \
    gdb \
    vim \
    nano \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for documentation and testing
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    h5py \
    netcdf4 \
    sphinx \
    sphinx-rtd-theme \
    sphinx-autodoc-typehints \
    myst-parser \
    pytest \
    pytest-cov \
    pytest-benchmark

# Set up CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# Stage 2: SUPERDARN RST Build Environment
# =============================================================================
FROM base as rst-build

# Create working directory
WORKDIR /opt/rst

# Copy source code
COPY codebase/ ./codebase/
COPY test_data/ ./test_data/
COPY build/ ./build/
COPY docs/ ./docs/

# Set up RST environment variables
ENV RST_ROOT=/opt/rst
ENV SYSTEM=linux
ENV MAKECFG=${RST_ROOT}/build/make/makecfg.${SYSTEM}
ENV MAKELIB=${RST_ROOT}/build/make/makelib.${SYSTEM}
ENV IPATH=${RST_ROOT}/codebase/include
ENV LIBPATH=${RST_ROOT}/codebase/lib/${SYSTEM}
ENV BINPATH=${RST_ROOT}/codebase/bin/${SYSTEM}
ENV PATH=${BINPATH}:${PATH}
ENV LD_LIBRARY_PATH=${LIBPATH}:${LD_LIBRARY_PATH}

# Create necessary directories
RUN mkdir -p ${LIBPATH} ${BINPATH} codebase/include codebase/lib/${SYSTEM}

# Build configuration
ARG BUILD_TYPE=optimized
ARG ENABLE_CUDA=ON
ARG ENABLE_TESTING=ON
ARG ENABLE_BENCHMARKS=ON

ENV BUILD_TYPE=${BUILD_TYPE}
ENV ENABLE_CUDA=${ENABLE_CUDA}
ENV ENABLE_TESTING=${ENABLE_TESTING}
ENV ENABLE_BENCHMARKS=${ENABLE_BENCHMARKS}

# =============================================================================
# Stage 3: CPU-Only Build (for compatibility testing)
# =============================================================================
FROM rst-build as cpu-build

ENV ENABLE_CUDA=OFF

# Build RST libraries (CPU only)
RUN cd codebase && \
    find . -name "makefile" -exec make -f {} clean \; || true && \
    find . -name "makefile" -exec make -f {} \; && \
    echo "CPU-only build completed"

# =============================================================================
# Stage 4: CUDA-Enabled Build (for performance testing)
# =============================================================================
FROM rst-build as cuda-build

ENV ENABLE_CUDA=ON

# Verify CUDA installation
RUN nvcc --version && \
    nvidia-smi || echo "GPU not available in build environment"

# Build RST libraries with CUDA support
RUN cd codebase && \
    find . -name "makefile.cuda" -exec make -f {} clean \; || true && \
    find . -name "makefile.cuda" -exec make -f {} \; && \
    echo "CUDA-enabled build completed"

# Build CUDA-specific components
RUN cd codebase/superdarn/src.lib/tk/fitacf_v3.0 && \
    make -f makefile.cuda all && \
    make -f makefile.cuda tests && \
    echo "CUDA components built successfully"

# =============================================================================
# Stage 5: Testing Environment
# =============================================================================
FROM cuda-build as testing

# Copy test scripts and data
COPY test_data/ ./test_data/
COPY codebase/superdarn/src.lib/tk/fitacf_v3.0/tests/ ./tests/

# Set up test environment
ENV TEST_DATA_PATH=/opt/rst/test_data/rawacf_samples
ENV RAWACF_PATH=${TEST_DATA_PATH}

# Create test runner script
RUN cat > /opt/rst/run_all_tests.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Running SUPERDARN RST Test Suite"
echo "=================================="

cd /opt/rst

# Environment check
echo "📋 Environment Check:"
echo "  CUDA Available: $(nvidia-smi > /dev/null 2>&1 && echo "YES" || echo "NO")"
echo "  Test Data: $(ls -la ${TEST_DATA_PATH}/ | wc -l) files"
echo "  Build Type: ${BUILD_TYPE}"
echo ""

# Run tests based on arguments
TEST_SUITE=${1:-standard}

case $TEST_SUITE in
    "quick")
        echo "🏃 Running Quick Tests..."
        cd tests && ./run_tests.sh --quick
        ;;
    "standard")
        echo "🔧 Running Standard Tests..."
        cd tests && ./run_tests.sh
        ;;
    "comprehensive")
        echo "🔬 Running Comprehensive Tests..."
        cd tests && ./run_tests.sh --comprehensive
        ;;
    "benchmark")
        echo "⚡ Running Benchmark Tests..."
        cd tests && ./run_tests.sh --benchmark
        ;;
    *)
        echo "❌ Unknown test suite: $TEST_SUITE"
        echo "Available options: quick, standard, comprehensive, benchmark"
        exit 1
        ;;
esac

echo "✅ Test suite completed!"
EOF

RUN chmod +x /opt/rst/run_all_tests.sh

# =============================================================================
# Stage 6: Documentation Build
# =============================================================================
FROM testing as docs-build

# Install additional documentation dependencies
RUN pip3 install --no-cache-dir \
    sphinx-copybutton \
    sphinx-tabs \
    sphinxcontrib-bibtex \
    nbsphinx \
    jupyter

# Create documentation structure
RUN mkdir -p docs/source docs/build

# Copy documentation source
COPY docs/ ./docs/

# Build documentation
RUN cd docs && \
    sphinx-build -b html source build/html && \
    echo "Documentation built successfully"

# =============================================================================
# Stage 7: Production Image
# =============================================================================
FROM cuda-build as production

# Copy built libraries and binaries
COPY --from=cuda-build /opt/rst/codebase/lib/ ./codebase/lib/
COPY --from=cuda-build /opt/rst/codebase/bin/ ./codebase/bin/

# Copy test data and documentation
COPY --from=testing /opt/rst/test_data/ ./test_data/
COPY --from=docs-build /opt/rst/docs/build/ ./docs/

# Create entrypoint script
RUN cat > /opt/rst/entrypoint.sh << 'EOF'
#!/bin/bash

# SUPERDARN RST Container Entrypoint
echo "🌟 SUPERDARN RST Container"
echo "========================="
echo "Build Type: ${BUILD_TYPE}"
echo "CUDA Enabled: ${ENABLE_CUDA}"
echo "Container: $(hostname)"
echo ""

# Check GPU availability
if nvidia-smi > /dev/null 2>&1; then
    echo "🚀 GPU Available:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "💻 CPU-only mode"
fi
echo ""

# Execute command or start interactive shell
if [ $# -eq 0 ]; then
    echo "Starting interactive shell..."
    echo "Available commands:"
    echo "  ./run_all_tests.sh [quick|standard|comprehensive|benchmark]"
    echo "  cd codebase && make -f makefile.cuda"
    echo "  cd docs && python -m http.server 8080"
    echo ""
    exec /bin/bash
else
    exec "$@"
fi
EOF

RUN chmod +x /opt/rst/entrypoint.sh

# Set working directory and entrypoint
WORKDIR /opt/rst
ENTRYPOINT ["/opt/rst/entrypoint.sh"]

# Expose documentation port
EXPOSE 8080

# Add labels for metadata
LABEL maintainer="SUPERDARN RST Team"
LABEL description="SUPERDARN RST with CUDA acceleration and comprehensive testing"
LABEL version="1.0"
LABEL cuda.version="${CUDA_VERSION}"
LABEL ubuntu.version="${UBUNTU_VERSION}"
