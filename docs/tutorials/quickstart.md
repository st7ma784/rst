# Quick Start Guide

Get up and running with RST in 10 minutes.

## Prerequisites

- RST installed ([Installation Guide](installation.md))
- Environment sourced: `source .profile.bash`

## Your First Commands

### 1. Check Installation

```bash
# Verify RST is accessible
which make_fit
# Output: /path/to/rst/build/bin/make_fit

# View help for key commands
make_fit --help
make_grid --help
```

### 2. Understand RST Data Flow

```
Raw Data (RAWACF)
       ↓
   make_fit      → Fitted data (FITACF)
       ↓
   make_grid     → Gridded data (GRID)
       ↓
   map_grd       → Convection maps (MAP)
       ↓
   map_plot      → Visualizations (PNG/PS)
```

### 3. Explore Sample Data

RST includes test data for learning:

```bash
# List available test data
ls test_data/

# View data file information
dmapdump test_data/sample.fitacf | head -50
```

### 4. Process Sample Data

```bash
# Convert FITACF to grid
make_grid -old test_data/sample.fitacf > output.grid

# Create a simple plot
grid_plot -png output.png output.grid
```

## Key RST Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `make_fit` | Raw → Fit | `make_fit input.rawacf > output.fitacf` |
| `make_grid` | Fit → Grid | `make_grid input.fitacf > output.grid` |
| `map_grd` | Grid → Map | `map_grd input.grid > output.map` |
| `grid_plot` | Visualize grid | `grid_plot -png out.png input.grid` |
| `map_plot` | Visualize map | `map_plot -png out.png input.map` |
| `dmapdump` | Inspect files | `dmapdump input.fitacf \| head` |

## Using CUDA Acceleration

If you have CUDA installed:

```bash
# Check CUDA availability
nvidia-smi

# Run with CUDA acceleration (automatic when available)
make_fit input.rawacf > output.fitacf

# Force CPU-only processing
RST_DISABLE_CUDA=1 make_fit input.rawacf > output.fitacf
```

## Python Interface

RST includes Python bindings:

```python
# Install Python package
pip install -e pythonv2/

# Use in scripts
from superdarn_gpu import FitACF

# Process data
processor = FitACF()
results = processor.process("input.rawacf")
```

## Docker Quick Start

```bash
# Run RST in container with your data
docker run -v $(pwd)/data:/data -it superdarn-rst

# Inside container
make_fit /data/input.rawacf > /data/output.fitacf
```

## Next Steps

- [First Processing Tutorial](first-processing.md) - Complete data workflow
- [CUDA Setup](cuda-setup.md) - GPU acceleration details
- [User Guide](../user_guide/) - Comprehensive usage documentation

## Getting Help

```bash
# Command help
<command> --help

# Man pages (if installed)
man make_fit
```
