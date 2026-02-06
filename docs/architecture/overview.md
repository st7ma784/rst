# Architecture Overview

This document provides a high-level overview of the RST architecture, covering both the original CPU-based design and the new CUDA-accelerated implementation.

## System Architecture

### High-Level View

```
┌─────────────────────────────────────────────────────────────────┐
│                     RST Application Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ make_fit │ │make_grid │ │ map_grd  │ │ map_plot │  ...      │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
┌───────▼────────────▼────────────▼────────────▼──────────────────┐
│                    RST Library Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ fitacf_v3.0  │  │  grid.1.24   │  │ cnvmap.1.17  │  ...      │
│  │   (tk)       │  │    (tk)      │  │    (tk)      │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
┌─────────▼─────────────────▼─────────────────▼───────────────────┐
│                  Computation Backend                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    CUDArst Library                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │ │
│  │  │ CUDA Kernels│  │ Memory Mgmt │  │  CPU Bridge │         │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│         ┌────────┐     ┌──────────┐    ┌──────────┐            │
│         │  GPU   │     │   CPU    │    │ Fallback │            │
│         │ (CUDA) │     │ (OpenMP) │    │ (Serial) │            │
│         └────────┘     └──────────┘    └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Component Layers

#### 1. Application Layer
Command-line tools that users interact with directly:
- `make_fit` - Process raw data to fitted parameters
- `make_grid` - Create spatial grids
- `map_grd` - Generate convection maps
- `*_plot` - Visualization tools

#### 2. Library Layer (`codebase/`)
Core algorithm implementations organized by function:
- `superdarn/` - SuperDARN-specific processing
- `general/` - General utilities
- `base/` - Foundational libraries

#### 3. Computation Backend
Handles actual computation with automatic backend selection:
- **CUDA kernels** - GPU-accelerated implementations
- **CPU fallback** - Standard CPU processing
- **Memory management** - Unified CPU/GPU memory handling

## Data Flow Architecture

### Processing Pipeline

```
Input Data                 Processing Stages                    Output
──────────                ──────────────────                   ──────

 RAWACF     ──────────▶   ┌─────────────────┐
 (raw IQ)                 │   fitacf_v3.0   │
                          │  ┌───────────┐  │
                          │  │ACF fitting│  │  ──────────▶   FITACF
                          │  │Noise est. │  │               (fits)
                          │  │Power calc │  │
                          │  └───────────┘  │
                          └─────────────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │   grid.1.24     │
                          │  ┌───────────┐  │
                          │  │Gridding   │  │  ──────────▶   GRID
                          │  │Merge      │  │               (spatial)
                          │  │Statistics │  │
                          │  └───────────┘  │
                          └─────────────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │  cnvmap.1.17    │
                          │  ┌───────────┐  │
                          │  │Sph. Harm. │  │  ──────────▶   MAP
                          │  │Potential  │  │               (convection)
                          │  │Fitting    │  │
                          │  └───────────┘  │
                          └─────────────────┘
```

### Data Formats

| Format | Content | Size | Processing |
|--------|---------|------|------------|
| RAWACF | Raw I/Q samples, ACF | ~10-100 MB/hr | Heavy |
| FITACF | Fitted velocity, power, width | ~1-10 MB/hr | Moderate |
| GRID | Spatial grid cells | ~100 KB-1 MB | Light |
| MAP | Convection patterns | ~10-100 KB | Light |

## Module Organization

### Directory Structure

```
codebase/
├── superdarn/
│   └── src.lib/
│       └── tk/                    # Toolkit libraries
│           ├── fitacf_v3.0/       # ★ CUDA accelerated
│           │   ├── src/
│           │   │   ├── fitacf.c          # Original CPU
│           │   │   ├── cuda_kernels.cu   # CUDA kernels
│           │   │   └── cuda_cpu_bridge.c # Integration
│           │   ├── include/
│           │   └── makefile.cuda
│           ├── grid.1.24/         # ★ CUDA accelerated
│           ├── raw.1.22/          # ★ CUDA accelerated
│           ├── scan.1.7/          # ★ CUDA accelerated
│           └── ...
├── general/                       # General utilities
└── base/                         # Base libraries
```

### Module Naming Convention

```
<name>.<major>.<minor>
  │       │       │
  │       │       └── Minor version (bug fixes)
  │       └────────── Major version (API changes)
  └────────────────── Functionality name
```

## Build System

### Compilation Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Top-Level Makefile                         │
│  (build/make/makefile)                                       │
└──────────────────────────────┬───────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Common    │         │   Module    │         │   CUDA      │
│  Libraries  │         │  Libraries  │         │  Libraries  │
│ (base, etc) │         │ (fitacf,..) │         │ (CUDArst)   │
└─────────────┘         └─────────────┘         └─────────────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Final Binaries    │
                    │   (build/bin/)      │
                    └─────────────────────┘
```

### Dual Build Support

Each CUDA-enabled module supports both builds:

```makefile
# Standard CPU build
make

# CUDA-accelerated build  
make -f makefile.cuda
```

## Runtime Architecture

### Backend Selection

```c
// Automatic backend selection (pseudocode)
if (cuda_available() && data_size > CUDA_THRESHOLD) {
    // Use CUDA implementation
    cuda_process_data(data);
} else {
    // Fall back to CPU
    cpu_process_data(data);
}
```

### Memory Management

```
┌─────────────────────────────────────────────────────────┐
│              Unified Memory Manager                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────────┐        ┌──────────────┐              │
│   │  Host Memory │◀──────▶│Device Memory │              │
│   │   (CPU RAM)  │  sync  │  (GPU VRAM)  │              │
│   └──────────────┘        └──────────────┘              │
│           │                       │                      │
│           ▼                       ▼                      │
│   ┌──────────────┐        ┌──────────────┐              │
│   │ CPU Compute  │        │ GPU Compute  │              │
│   └──────────────┘        └──────────────┘              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Backward Compatibility
- All original APIs preserved
- Existing code works unchanged
- CUDA is opt-in and transparent

### 2. Graceful Degradation
- Automatic CPU fallback
- No CUDA? No problem
- Same results, different speed

### 3. Data Structure Evolution
- Linked lists → Arrays + masks
- Enables GPU parallelization
- Preserves semantic behavior

### 4. Modular Acceleration
- Each module independently accelerated
- Mix CPU and CUDA modules
- Incremental migration path

## Next Steps

- [Original Design](original-design.md) - Deep dive into legacy architecture
- [CUDA Implementation](cuda-implementation.md) - GPU acceleration details
- [Data Structures](data-structures.md) - Old vs new comparison
