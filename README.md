[![DOI](https://zenodo.org/badge/74060190.svg)](https://zenodo.org/badge/latestdoi/74060190)
[![Documentation](https://img.shields.io/badge/docs-Sphinx-blue)](https://superdarn.github.io/rst/)
[![License](https://img.shields.io/badge/license-LGPL--3.0-green)](LICENSE)

# SuperDARN Radar Software Toolkit (RST)

The **Radar Software Toolkit (RST)** is a comprehensive data processing system for SuperDARN (Super Dual Auroral Radar Network) radar data. This version includes **CUDA GPU acceleration** for significant performance improvements (up to 16x speedup).

Maintained by the [SuperDARN Data Analysis Working Group (DAWG)](https://superdarn.github.io/dawg/).

---

## Quick Start

### Using Docker (Recommended)

```bash
# Clone and build
git clone https://github.com/SuperDARN/rst.git && cd rst
docker build -t superdarn-rst .

# Run with GPU support
docker run --gpus all -it superdarn-rst

# Or without GPU (CPU-only)
docker run -it superdarn-rst
```

### Native Installation

```bash
# Prerequisites: GCC, Make, CUDA Toolkit (optional)
git clone https://github.com/SuperDARN/rst.git && cd rst

# Build standard toolkit
make -C build

# Build with CUDA acceleration (requires NVIDIA GPU + CUDA Toolkit)
./scripts/build_all_cuda_modules.sh
```

See [Installation Guide](docs/tutorials/installation.md) for detailed platform-specific instructions.

---

## Documentation

📚 **Full documentation**: [`docs/`](docs/) | [Online Docs](https://superdarn.github.io/rst/)

| Section | Description |
|---------|-------------|
| [**Tutorials**](docs/tutorials/) | Step-by-step guides for getting started |
| [**User Guide**](docs/user_guide/) | Data processing workflows and examples |
| [**Architecture**](docs/architecture/) | Technical deep-dive: original vs CUDA implementation |
| [**Guides**](docs/guides/) | Comprehensive reference for all components |
| [**Maintenance**](docs/maintenance/) | Deployment, troubleshooting, and administration |
| [**API Reference**](docs/references/) | Function and module documentation |

---

## Key Features

### CUDA GPU Acceleration

| Operation | CPU Time | CUDA Time | Speedup |
|-----------|----------|-----------|---------|
| Data Copying | 45.2 ms | 2.8 ms | **16.1x** |
| Power/Phase Computation | 38.7 ms | 3.1 ms | **12.5x** |
| Statistical Reduction | 52.1 ms | 4.2 ms | **12.4x** |
| ACF Processing | - | - | **20-60x** |

**Accelerated Modules**: FITACF v3.0, LMFIT v2.0, Grid, Raw, Scan, Fit, ACF, IQ, ConvMap

### Core Capabilities

- **Data Processing**: Complete pipeline from raw ACF to convection maps
- **Fitting Algorithms**: Advanced auto-correlation and cross-correlation fitting
- **Visualization**: Built-in plotting for fields, grids, and maps
- **Backward Compatible**: CPU fallback when GPU unavailable
- **Extensive Testing**: Validated CUDA vs CPU numerical accuracy

---

## Repository Structure

```
rst/
├── README.md              # This file
├── docs/                  # 📚 All documentation (Sphinx)
│   ├── tutorials/         # Getting started guides
│   ├── user_guide/        # Usage workflows
│   ├── architecture/      # Technical design docs
│   ├── guides/            # Comprehensive references
│   └── maintenance/       # Deployment & admin
├── codebase/              # 🔧 Core RST source code
│   ├── superdarn/         # SuperDARN-specific modules
│   ├── general/           # General utilities
│   └── base/              # Base libraries
├── CUDArst/               # 🚀 Unified CUDA library
├── pythonv2/              # 🐍 Python bindings
├── scripts/               # 📜 Build and utility scripts
├── build/                 # Compiled binaries
├── include/               # Header files
└── tables/                # Reference data tables
```

---

## Data Policy & Citation

### Citing the RST

See [Zenodo](https://doi.org/10.5281/zenodo.801458) for citation formats, or use:

```bibtex
@software{superdarn_rst,
  author = {{SuperDARN Data Analysis Working Group}},
  title = {Radar Software Toolkit},
  url = {https://github.com/SuperDARN/rst},
  doi = {10.5281/zenodo.801458}
}
```

### Data Access

SuperDARN data available on [FRDR](https://www.frdr-dfdr.ca/repo/collection/superdarn) under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

**Required Acknowledgement:**
> The authors acknowledge the use of SuperDARN data. SuperDARN is a collection of radars funded by national scientific funding agencies of Australia, Canada, China, France, Italy, Japan, Norway, South Africa, United Kingdom and the United States of America.

---

## Contributing

We welcome contributions! See [Contributing Guide](docs/guides/contributing.md).

- 🐛 [Report Issues](https://github.com/SuperDARN/rst/issues)
- 💡 [Request Features](https://github.com/SuperDARN/rst/discussions)
- 🔧 [Submit Pull Requests](https://github.com/SuperDARN/rst/pulls)

---

## License

Licensed under [LGPL-3.0](LICENSE). See [AUTHORS.md](AUTHORS.md) for contributors
 - Discuss [issues](https://github.com/SuperDARN/rst/issues) and answer questions
 - **Test CUDA optimizations**: Help validate GPU acceleration on different hardware configurations
 - Become a developer: if you would like to contribute code to the RST, please submit a new [issue](https://github.com/SuperDARN/rst/issues) on Github, or contact us at darn-dawg *at* isee *dot* nagoya-u *dot* ac *dot* jp.
