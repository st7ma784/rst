# Python API Guide

Documentation for RST Python bindings (`superdarn_gpu`).

## Installation

```bash
# Install from source
cd pythonv2
pip install -e .

# Or install dependencies and package
pip install -r requirements.txt
pip install .
```

## Quick Start

```python
from superdarn_gpu import FitACF, GridProcessor, CUDAConfig

# Check CUDA availability
config = CUDAConfig()
print(f"CUDA available: {config.cuda_available}")
print(f"Device: {config.device_name}")

# Process FITACF data
processor = FitACF()
results = processor.process_file("data.rawacf")

# Access results
print(f"Velocities: {results.velocity}")
print(f"Powers: {results.power}")
```

## Core Classes

### FitACF

Process raw ACF data to extract fitted parameters.

```python
from superdarn_gpu import FitACF

# Initialize processor
processor = FitACF(
    use_cuda=True,      # Enable GPU acceleration
    device_id=0,        # GPU device ID
    verbose=False       # Debug output
)

# Process file
results = processor.process_file("input.rawacf")

# Process numpy arrays directly
import numpy as np

acf_real = np.random.randn(75, 100).astype(np.float32)
acf_imag = np.random.randn(75, 100).astype(np.float32)

results = processor.process_arrays(
    acf_real=acf_real,
    acf_imag=acf_imag,
    n_ranges=75,
    n_lags=100
)
```

#### FitACF Results

```python
class FitACFResults:
    velocity: np.ndarray      # [n_ranges] Velocity (m/s)
    velocity_error: np.ndarray
    power: np.ndarray         # [n_ranges] Power (dB)
    power_error: np.ndarray
    width: np.ndarray         # [n_ranges] Spectral width (m/s)
    width_error: np.ndarray
    quality: np.ndarray       # [n_ranges] Quality flags
    
    # Metadata
    n_ranges: int
    processing_time: float    # Seconds
    used_cuda: bool
```

### GridProcessor

Create spatial grids from fitted data.

```python
from superdarn_gpu import GridProcessor

# Initialize
grid = GridProcessor(
    resolution=1.0,     # Degrees
    method='median',    # Averaging method
    use_cuda=True
)

# Process fit data
grid_data = grid.process_file("data.fitacf")

# Or from FitACF results
grid_data = grid.process_results(fitacf_results)

# Access grid
print(f"Grid shape: {grid_data.shape}")
print(f"Lat range: {grid_data.lat_range}")
print(f"Lon range: {grid_data.lon_range}")
```

### ConvectionMap

Generate convection maps using spherical harmonics.

```python
from superdarn_gpu import ConvectionMap

# Initialize
mapper = ConvectionMap(
    order=8,            # Spherical harmonic order
    hemisphere='north', # 'north' or 'south'
    use_cuda=True
)

# Process grid data
conv_map = mapper.process(grid_data)

# Access results
potential = conv_map.potential      # 2D potential field
velocity = conv_map.velocity        # 2D velocity field
```

### CUDAConfig

Manage CUDA configuration.

```python
from superdarn_gpu import CUDAConfig

config = CUDAConfig()

# Check availability
if config.cuda_available:
    print(f"Device: {config.device_name}")
    print(f"Compute capability: {config.compute_capability}")
    print(f"Memory: {config.total_memory / 1e9:.1f} GB")
    print(f"Cores: {config.cuda_cores}")

# Set device
config.set_device(0)

# Get memory info
free, total = config.get_memory_info()
print(f"Free: {free / 1e9:.1f} GB / {total / 1e9:.1f} GB")
```

## File I/O

### Reading Files

```python
from superdarn_gpu.io import read_rawacf, read_fitacf, read_grid

# Read RAWACF
raw_data = read_rawacf("data.rawacf")
print(f"Records: {len(raw_data)}")
print(f"Time range: {raw_data.time_start} - {raw_data.time_end}")

# Read FITACF
fit_data = read_fitacf("data.fitacf")

# Read Grid
grid_data = read_grid("data.grid")
```

### Writing Files

```python
from superdarn_gpu.io import write_fitacf, write_grid

# Write FITACF results
write_fitacf(results, "output.fitacf")

# Write grid data
write_grid(grid_data, "output.grid")
```

## Visualization

### Quick Plots

```python
from superdarn_gpu import plot

# Field plot
plot.field(fitacf_results, 
           param='velocity',
           save='field.png')

# Grid plot
plot.grid(grid_data,
          param='velocity',
          coastlines=True,
          save='grid.png')

# Time series
plot.timeseries(fitacf_results,
                beam=7, range_gate=30,
                param='velocity',
                save='timeseries.png')
```

### Custom Plots

```python
import matplotlib.pyplot as plt
from superdarn_gpu import plot

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Velocity field
plot.field(results, param='velocity', ax=axes[0, 0])

# Power field
plot.field(results, param='power', ax=axes[0, 1])

# Time series
plot.timeseries(results, beam=7, ax=axes[1, 0])

# Statistics
plot.histogram(results.velocity, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('analysis.png')
```

## Batch Processing

### Process Multiple Files

```python
from superdarn_gpu import FitACF
from pathlib import Path
import concurrent.futures

def process_file(filepath):
    processor = FitACF()
    return processor.process_file(filepath)

# Get all RAWACF files
files = list(Path("data/").glob("*.rawacf"))

# Process in parallel (CPU parallelism, GPU per-file)
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))
```

### Pipeline Processing

```python
from superdarn_gpu import Pipeline

# Create processing pipeline
pipeline = Pipeline([
    ('fit', FitACF(use_cuda=True)),
    ('grid', GridProcessor(resolution=1.0)),
    ('map', ConvectionMap(order=8))
])

# Process through pipeline
result = pipeline.process("data.rawacf")

# Access intermediate results
fit_result = result['fit']
grid_result = result['grid']
map_result = result['map']
```

## Performance Control

### Disable CUDA

```python
# Per-processor
processor = FitACF(use_cuda=False)

# Global
import os
os.environ['RST_DISABLE_CUDA'] = '1'

from superdarn_gpu import FitACF
processor = FitACF()  # Will use CPU
```

### Memory Management

```python
from superdarn_gpu import CUDAConfig

# Pre-allocate memory pool
config = CUDAConfig()
config.allocate_pool(size_mb=1024)

# Process multiple files with pooled memory
for file in files:
    results = processor.process_file(file)

# Release pool
config.release_pool()
```

### Benchmarking

```python
from superdarn_gpu import benchmark

# Compare CPU vs CUDA
results = benchmark.compare(
    data_file="large_data.rawacf",
    iterations=10
)

print(f"CPU: {results.cpu_time:.2f} ms")
print(f"CUDA: {results.cuda_time:.2f} ms")
print(f"Speedup: {results.speedup:.1f}x")
```

## Error Handling

```python
from superdarn_gpu import FitACF
from superdarn_gpu.exceptions import (
    CUDAError,
    FileFormatError,
    ProcessingError
)

try:
    processor = FitACF(use_cuda=True)
    results = processor.process_file("data.rawacf")

except CUDAError as e:
    print(f"CUDA error: {e}")
    # Fall back to CPU
    processor = FitACF(use_cuda=False)
    results = processor.process_file("data.rawacf")

except FileFormatError as e:
    print(f"Invalid file format: {e}")

except ProcessingError as e:
    print(f"Processing failed: {e}")
```

## Examples

### Complete Processing Script

```python
#!/usr/bin/env python3
"""Process SuperDARN data with CUDA acceleration."""

import argparse
from pathlib import Path
from superdarn_gpu import FitACF, GridProcessor, ConvectionMap, plot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input RAWACF file')
    parser.add_argument('-o', '--output', default='output',
                        help='Output directory')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Process data
    print("Processing FITACF...")
    fit = FitACF(use_cuda=use_cuda)
    fit_results = fit.process_file(args.input)
    
    print("Creating grid...")
    grid = GridProcessor(use_cuda=use_cuda)
    grid_data = grid.process_results(fit_results)
    
    print("Generating map...")
    mapper = ConvectionMap(use_cuda=use_cuda)
    conv_map = mapper.process(grid_data)
    
    # Create visualizations
    print("Creating plots...")
    plot.field(fit_results, save=output_dir / 'field.png')
    plot.grid(grid_data, save=output_dir / 'grid.png')
    plot.convection(conv_map, save=output_dir / 'map.png')
    
    print(f"Done! Output in {output_dir}")

if __name__ == '__main__':
    main()
```

### Jupyter Notebook Example

```python
# Cell 1: Setup
from superdarn_gpu import FitACF, plot, CUDAConfig
import matplotlib.pyplot as plt

config = CUDAConfig()
print(f"Using GPU: {config.device_name if config.cuda_available else 'None'}")

# Cell 2: Process data
processor = FitACF()
results = processor.process_file("data.rawacf")

print(f"Processed {results.n_ranges} ranges")
print(f"Time: {results.processing_time*1000:.1f} ms")
print(f"CUDA: {results.used_cuda}")

# Cell 3: Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot.field(results, param='velocity', ax=axes[0], title='Velocity')
plot.field(results, param='power', ax=axes[1], title='Power')
plot.field(results, param='width', ax=axes[2], title='Width')
plt.tight_layout()
```
