[![DOI](https://zenodo.org/badge/74060190.svg)](https://zenodo.org/badge/latestdoi/74060190)

Radar Software Toolkit
========
The Radar Software Toolkit (RST) is maintained by the SuperDARN Data Analysis Working Group (DAWG). For general information and updates from the DAWG, visit our [website](https://superdarn.github.io/dawg/).

## Documentation

RST's documentation is currently hosted on two sites:

- RST readthedocs includes the installation guide, RST Tutorials, and SuperDARN data formats:
  https://radar-software-toolkit-rst.readthedocs.io/en/latest/
- RST API documentation includes the software structure and binary command line description and options: 
  https://superdarn.github.io/rst/


## Installation

Installation guide for:

  - [Linux](https://radar-software-toolkit-rst.readthedocs.io/en/latest/user_guide/linux_install/)
  - [MacOSX](https://radar-software-toolkit-rst.readthedocs.io/en/latest/user_guide/mac_install/)
  - <font color="grey">Windows </font> yet to be implemented


## How to cite the RST

Instructions for citing the RST are available on [Zenodo](https://doi.org/10.5281/zenodo.801458). Scroll to the end of the page and choose your perferred citation format.

## Data access and usage policy

SuperDARN data are published on [FRDR](https://www.frdr-dfdr.ca/repo/collection/superdarn) in the `rawacf`/`dat` format. These data are licensed under the [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) license](https://creativecommons.org/licenses/by-nc/4.0/).

**Please read the [SuperDARN data policy](https://g-772fa5.cd4fe.0ec8.data.globus.org/7/published/publication_285/submitted_data/2018RAWACF.readme.txt) when downloading data from FRDR.**

For all usage of SuperDARN data, users are asked to include the following standard acknowledgement text:

> The authors acknowledge the use of SuperDARN data. SuperDARN is a collection of radars funded by national scientific funding agencies of Australia, Canada, China, France, Italy, Japan, Norway, South Africa, United Kingdom and the United States of America.

When data from an individual radar or radars are used, users must contact the principal investigator(s) of those radar(s) to obtain the appropriate acknowledgement information and to offer collaboration, where appropriate. PI contact information can be found [here](https://superdarn.ca/radar-info).

For more information, please read [citing SuperDARN data](https://radar-software-toolkit-rst.readthedocs.io/en/latest/user_guide/citing.md).

## CUDA Acceleration 🚀

The RST now includes **CUDA-accelerated processing** for significant performance improvements:

### Features
- **Up to 16x speedup** on GPU-enabled systems
- **CUDA-compatible linked list** data structures with validity masking
- **Advanced parallel kernels** for ACF/XCF processing, power/phase computation, and statistical reduction
- **Seamless CPU fallback** for systems without CUDA support
- **Comprehensive testing** with automated CPU vs CUDA validation

### CUDA Components Location
```
codebase/superdarn/src.lib/tk/fitacf_v3.0/
├── src/
│   ├── cuda_kernels.cu              # Core CUDA kernels
│   ├── cuda_advanced_kernels.cu     # Advanced optimization kernels
│   ├── cuda_llist.cu               # CUDA-compatible data structures
│   ├── cuda_cpu_bridge.c           # CPU-CUDA integration layer
│   └── cuda_fitacf_optimizer.c     # High-level optimization interface
├── include/
│   └── cuda_llist.h                # CUDA data structure definitions
├── tests/                          # Comprehensive test suite
└── makefile.cuda                   # CUDA build configuration
```

### Quick Start with CUDA
```bash
# Build with CUDA support
cd codebase/superdarn/src.lib/tk/fitacf_v3.0
make -f makefile.cuda

# Run tests
cd tests && ./run_tests.sh

# Run benchmarks
./cuda_integration_test
```

### Docker Support
Use the unified Docker container for easy setup:
```bash
# Build container
docker build -t superdarn-rst .

# Run with GPU support
docker run --gpus all -it superdarn-rst

# Run tests in container
docker run --gpus all superdarn-rst ./run_all_tests.sh
```

## Contribute to the RST

The DAWG welcomes new testers and developers to join the team. Here are some ways to contribute:

 - Test pull requests: to determine which [pull requests](https://github.com/SuperDARN/rst/pulls) need to be tested right away, filter them by their milestones
 - Discuss [issues](https://github.com/SuperDARN/rst/issues) and answer questions
 - **Test CUDA optimizations**: Help validate GPU acceleration on different hardware configurations
 - Become a developer: if you would like to contribute code to the RST, please submit a new [issue](https://github.com/SuperDARN/rst/issues) on Github, or contact us at darn-dawg *at* isee *dot* nagoya-u *dot* ac *dot* jp.
