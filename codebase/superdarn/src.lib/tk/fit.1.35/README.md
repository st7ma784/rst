# SuperDARN fit.1.35 Module

This module is part of the SuperDARN radar data processing toolkit. It provides functionality for fitting ACF (Auto-Correlation Function) data to theoretical models.

## New Build System

We've implemented a modern CMake-based build system that supports:
- Multiple build types (Debug, Release, etc.)
- AVX/AVX2/AVX-512 optimizations
- CUDA acceleration
- Unit tests
- Benchmarks
- Easy installation

### Prerequisites

- CMake 3.10 or higher
- C/C++ compiler with C++11 support
- (Optional) CUDA Toolkit for GPU acceleration
- (Optional) Google Test for unit testing
- (Optional) Google Benchmark for performance testing

### Building

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd fit.1.35

# Make the build script executable
chmod +x build.sh

# Build with default options (Release, AVX enabled, no CUDA)
./build.sh

# Build with CUDA support
./build.sh --cuda

# Build with debug symbols
./build.sh --debug

# Build without AVX optimizations
./build.sh --no-avx

# Build with custom installation prefix
./build.sh --prefix=/path/to/install

# Build with specific number of parallel jobs
./build.sh -j8
```

### Installation

After building, you can install the library:

```bash
cd build
cmake --build . --target install
```

### Running Tests

If tests were built, you can run them with:

```bash
cd build
ctest --output-on-failure
```

### Running Benchmarks

If benchmarks were built, you can run them with:

```bash
cd build
./benchmarks/fit_benchmarks
```

### Using the Library

To use the library in your CMake project:

```cmake
find_package(fit 1.35 REQUIRED)
target_link_libraries(your_target PRIVATE fit::fit_optimized)  # or fit_original
```

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `BUILD_ORIGINAL` | Build original implementation | ON |
| `ENABLE_AVX` | Enable AVX/AVX2/AVX-512 optimizations | ON |
| `ENABLE_CUDA` | Enable CUDA acceleration | OFF |
| `BUILD_TESTS` | Build test programs | ON |
| `BUILD_BENCHMARKS` | Build benchmark programs | OFF |

### Directory Structure

- `include/` - Public header files
- `src/` - Source files
  - `avx/` - AVX-optimized implementations
- `cuda/` - CUDA kernel implementations
- `tests/` - Unit tests
- `benchmarks/` - Performance benchmarks
- `cmake/` - CMake modules and configuration

### License

This software is licensed under the GNU General Public License v3.0. See the LICENSE file for details.
