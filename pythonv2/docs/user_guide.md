# SuperDARN GPU User Guide

This guide provides comprehensive instructions for using SuperDARN GPU for radar data processing.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Data Processing Workflows](#data-processing-workflows)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for large datasets)
- GPU with 4GB+ VRAM (optional but recommended)

### Step-by-Step Installation

1. **Install CUDA Dependencies**
   ```bash
   # Check CUDA installation
   nvidia-smi
   nvcc --version
   ```

2. **Install SuperDARN GPU**
   ```bash
   # From source (recommended for latest features)
   git clone https://github.com/SuperDARN/rst.git
   cd rst/pythonv2
   pip install -e .
   
   # With all optional dependencies
   pip install -e .[viz,ml,dev]
   ```

3. **Verify Installation**
   ```python
   import superdarn_gpu as sd
   print(f"Version: {sd.__version__}")
   print(f"GPU Available: {sd.GPU_AVAILABLE}")
   ```

## Quick Start

### Basic Processing Pipeline

```python
import superdarn_gpu as sd
from datetime import datetime

# Load raw data
rawacf_files = ['20231201.120000.cly.rawacf', '20231201.120200.cly.rawacf']
rawacf_data = sd.io.load_batch(rawacf_files)

# Process ACF → FITACF → Grid
fitacf_results = []
for rawacf in rawacf_data:
    fitacf = sd.processing.fitacf(rawacf)
    fitacf_results.append(fitacf)

# Create spatial grid
grid_data = sd.processing.grid(fitacf_results, resolution=1.0)

# Generate convection map  
conv_map = sd.processing.mapping(grid_data, model_order=8)

# Save results
sd.io.save('processed_data.h5', {
    'fitacf': fitacf_results,
    'grid': grid_data,
    'convection_map': conv_map
})

print(f"Processed {len(fitacf_results)} records")
print(f"Grid coverage: {sd.tools.calculate_coverage(grid_data):.1%}")
```

### Performance-Optimized Processing

```python
import superdarn_gpu as sd

# Configure for optimal GPU usage
sd.set_backend('cupy')
sd.core.memory.memory_pool.set_memory_limit(limit_fraction=0.8)

# Create processing pipeline
pipeline = sd.ProcessingPipeline("High-Performance Pipeline")

# Add optimized stages
pipeline.add_stage(sd.processing.ACFProcessor(
    config=sd.processing.ACFConfig(
        use_gpu_kernels=True,
        batch_size=1024,
        dc_offset_removal=True
    )
))

pipeline.add_stage(sd.processing.FitACFProcessor(
    config=sd.processing.FitACFConfig(
        algorithm='v3.0',
        batch_size=512,
        use_shared_memory=True
    )
))

# Process data with automatic optimization
results = pipeline.run_batch(data_files, batch_size=10)

# Get performance statistics
stats = pipeline.get_performance_summary()
print(f"Processing time: {stats['total_time']:.2f}s")
print(f"GPU memory used: {stats['total_memory_used'] / (1024**3):.2f} GB")
print(f"Average speed: {stats['total_time'] / len(data_files):.3f}s per file")
```

## Core Concepts

### Data Structures

SuperDARN GPU uses GPU-optimized data structures that automatically manage memory between CPU and GPU:

```python
# All data structures support GPU/CPU operation
rawacf = sd.RawACF(nrang=75, mplgs=18, nave=50)
print(f"Data on GPU: {rawacf.use_gpu}")

# Seamless transfer between devices
rawacf_cpu = rawacf.to_cpu()    # Transfer to CPU
rawacf_gpu = rawacf.to_gpu()    # Transfer to GPU

# Backend context switching
with sd.BackendContext('numpy'):  # Force CPU processing
    cpu_result = sd.processing.fitacf(rawacf)

with sd.BackendContext('cupy'):   # Force GPU processing  
    gpu_result = sd.processing.fitacf(rawacf)
```

### Memory Management

SuperDARN GPU includes sophisticated memory management for efficient processing:

```python
from superdarn_gpu.core.memory import memory_pool, gpu_memory_context

# Check memory usage
info = memory_pool.get_memory_info()
print(f"GPU memory usage: {memory_pool.get_memory_usage_percent():.1f}%")

# Automatic memory management
with gpu_memory_context(limit_fraction=0.7):
    # Processing will use max 70% of GPU memory
    results = process_large_dataset(files)
    
# Memory optimization
memory_pool.optimize_memory()  # Free unused memory
```

### Processing Configuration

Configure algorithms for optimal performance and accuracy:

```python
# ACF processing configuration
acf_config = sd.processing.ACFConfig(
    bad_sample_threshold=1e6,      # Saturation threshold
    dc_offset_removal=True,        # Remove DC bias
    xcf_processing=True,           # Process interferometer data
    use_gpu_kernels=True,          # Enable custom CUDA kernels
    batch_size=1024                # GPU batch size
)

# FITACF processing configuration  
fitacf_config = sd.processing.FitACFConfig(
    algorithm='v3.0',               # Latest algorithm version
    min_power_threshold=3.0,        # Power threshold (dB)
    max_velocity=2000.0,           # Velocity limit (m/s)
    ground_scatter_threshold=0.3,   # Ground scatter detection
    batch_size=512,                # Processing batch size
    elevation_correction=True       # Elevation angle correction
)

# Grid processing configuration
grid_config = sd.processing.GridConfig(
    lat_resolution=1.0,            # Latitude resolution (degrees)
    lon_resolution=2.0,            # Longitude resolution (degrees) 
    coordinate_system='magnetic',   # Coordinate system
    grid_method='weighted_mean',    # Interpolation method
    min_vectors_per_cell=3,        # Quality control
    use_gpu_interpolation=True     # GPU spatial interpolation
)
```

## Data Processing Workflows

### Single File Processing

```python
import superdarn_gpu as sd

# Load and process single file
rawacf = sd.io.load('data.rawacf')
fitacf = sd.processing.fitacf(rawacf)

# Access fitted parameters
velocities = fitacf.velocity[fitacf.qflg > 0]  # Valid velocities
print(f"Velocity range: {velocities.min():.1f} to {velocities.max():.1f} m/s")
print(f"Mean velocity: {velocities.mean():.1f} ± {velocities.std():.1f} m/s")

# Quality statistics
total_ranges = len(fitacf.qflg)
fitted_ranges = sum(fitacf.qflg > 0)
ground_scatter = sum(fitacf.gflg > 0)

print(f"Data quality: {fitted_ranges}/{total_ranges} ranges fitted")
print(f"Ground scatter: {ground_scatter} ranges")
```

### Batch Processing

```python
import superdarn_gpu as sd
from pathlib import Path

# Find all data files
data_dir = Path('/data/superdarn/2023/12')
rawacf_files = list(data_dir.glob('*.rawacf'))

# Set up batch processing
batch_size = sd.tools.suggest_batch_size(
    total_records=len(rawacf_files),
    record_memory=sd.tools.estimate_memory_requirement(nrang=75, mplgs=18)
)

print(f"Processing {len(rawacf_files)} files in batches of {batch_size}")

# Process in batches with progress tracking
results = []
for i in range(0, len(rawacf_files), batch_size):
    batch_files = rawacf_files[i:i+batch_size]
    
    print(f"Processing batch {i//batch_size + 1}/{len(rawacf_files)//batch_size + 1}")
    
    # Load batch
    rawacf_batch = sd.io.load_batch(batch_files)
    
    # Process batch
    fitacf_batch = [sd.processing.fitacf(rawacf) for rawacf in rawacf_batch]
    
    results.extend(fitacf_batch)
    
    # Memory cleanup between batches
    sd.core.memory.memory_pool.optimize_memory()

# Save batch results
sd.io.save_batch(results, 'processed_batch.h5')
print(f"Processed {len(results)} files total")
```

### Advanced Pipeline Processing

```python
import superdarn_gpu as sd

# Custom processing stage
class CustomQualityFilter(sd.core.pipeline.Stage):
    def __init__(self, min_snr=3.0):
        super().__init__(name="Quality Filter")
        self.min_snr = min_snr
    
    def process(self, fitacf):
        # Apply custom quality filtering
        snr = fitacf.power / fitacf.noise  
        fitacf.qflg[snr < self.min_snr] = 0
        return fitacf

# Build custom pipeline
pipeline = sd.ProcessingPipeline("Custom Analysis Pipeline")
pipeline.add_stage(sd.processing.ACFProcessor())
pipeline.add_stage(sd.processing.FitACFProcessor())
pipeline.add_stage(CustomQualityFilter(min_snr=5.0))
pipeline.add_stage(sd.processing.GridProcessor())

# Process with custom pipeline
results = pipeline.run(input_data)
```

### Real-Time Processing

```python
import superdarn_gpu as sd
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SuperDARNFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.processor = sd.processing.FitACFProcessor()
    
    def on_created(self, event):
        if event.src_path.endswith('.rawacf'):
            print(f"Processing new file: {event.src_path}")
            
            # Load and process immediately
            rawacf = sd.io.load(event.src_path)
            fitacf = self.processor.process(rawacf)
            
            # Save result
            output_path = event.src_path.replace('.rawacf', '.fitacf.h5')
            sd.io.save(output_path, fitacf)
            
            print(f"Processed and saved: {output_path}")

# Set up real-time monitoring
event_handler = SuperDARNFileHandler()
observer = Observer()
observer.schedule(event_handler, path='/data/incoming', recursive=False)
observer.start()

print("Monitoring /data/incoming for new files...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

## Performance Optimization

### GPU Memory Optimization

```python
import superdarn_gpu as sd

# Monitor memory usage
def print_memory_stats():
    info = sd.core.memory.memory_pool.get_memory_info()
    print(f"GPU Memory: {info['used_pool']/(1024**3):.2f}/{info['total']/(1024**3):.2f} GB")

# Optimize for large datasets
with sd.core.memory.gpu_memory_context(limit_fraction=0.9):
    print_memory_stats()
    
    # Process large dataset
    results = []
    for data_file in large_dataset:
        rawacf = sd.io.load(data_file)
        fitacf = sd.processing.fitacf(rawacf)
        results.append(fitacf)
        
        # Periodic cleanup
        if len(results) % 50 == 0:
            print_memory_stats()
            sd.core.memory.memory_pool.optimize_memory()
```

### Batch Size Optimization

```python
import superdarn_gpu as sd

# Find optimal batch size
def benchmark_batch_sizes(data, batch_sizes=[64, 128, 256, 512, 1024]):
    results = {}
    
    for batch_size in batch_sizes:
        config = sd.processing.FitACFConfig(batch_size=batch_size)
        processor = sd.processing.FitACFProcessor(config=config)
        
        start_time = time.time()
        result = processor.process(data)
        processing_time = time.time() - start_time
        
        results[batch_size] = processing_time
        print(f"Batch size {batch_size}: {processing_time:.3f}s")
    
    optimal_batch_size = min(results, key=results.get)
    print(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

# Use optimal settings
optimal_batch = benchmark_batch_sizes(sample_data)
config = sd.processing.FitACFConfig(batch_size=optimal_batch)
```

### Parallel Processing

```python
import superdarn_gpu as sd
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

def process_file(filename):
    """Process single file"""
    rawacf = sd.io.load(filename)
    fitacf = sd.processing.fitacf(rawacf)
    
    # Save result
    output_file = filename.replace('.rawacf', '.fitacf.h5')
    sd.io.save(output_file, fitacf)
    return output_file

# Parallel processing with multiple GPUs
file_list = ['file1.rawacf', 'file2.rawacf', 'file3.rawacf']

# Method 1: ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, file_list))

# Method 2: Multiprocessing Pool  
with Pool(processes=4) as pool:
    results = pool.map(process_file, file_list)

print(f"Processed {len(results)} files in parallel")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. GPU Memory Errors

```python
# Problem: CUDA out of memory
# Solution: Reduce batch size and use memory management

sd.core.memory.memory_pool.set_memory_limit(limit_fraction=0.7)
config = sd.processing.FitACFConfig(batch_size=256)  # Reduce from default 512
```

#### 2. Slow Processing Performance

```python
# Problem: Slower than expected processing
# Solution: Check GPU utilization and optimize settings

# Check if GPU is being used
print(f"Current backend: {sd.get_backend()}")
print(f"GPU available: {sd.GPU_AVAILABLE}")

# Force GPU usage
sd.set_backend('cupy')

# Use performance profiling
with sd.tools.PerformanceProfiler() as profiler:
    result = sd.processing.fitacf(rawacf)
    
profiler.print_summary()
```

#### 3. Installation Issues

```bash
# Problem: CuPy installation fails
# Solution: Install compatible CUDA version

# Check CUDA version
nvidia-smi

# Install specific CuPy version
pip install cupy-cuda11x  # for CUDA 11.x
pip install cupy-cuda12x  # for CUDA 12.x
```

#### 4. Data Format Issues

```python
# Problem: Unsupported data format
# Solution: Convert data or check format

try:
    data = sd.io.load('mysterious_file.dat')
except ValueError as e:
    print(f"Format error: {e}")
    
    # Try explicit format specification
    data = sd.io.load('mysterious_file.dat', data_type='rawacf')
    
    # Or convert format
    sd.tools.convert_format('mysterious_file.dat', 'output.h5', 
                           input_format='dmap', output_format='hdf5')
```

### Performance Monitoring

```python
import superdarn_gpu as sd

# Built-in performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.stats = []
    
    def monitor_processing(self, data_files):
        for file in data_files:
            start_time = time.time()
            memory_start = sd.core.memory.memory_pool.get_memory_usage_percent()
            
            # Process file
            rawacf = sd.io.load(file)
            fitacf = sd.processing.fitacf(rawacf)
            
            # Record stats
            processing_time = time.time() - start_time
            memory_peak = sd.core.memory.memory_pool.get_memory_usage_percent()
            
            self.stats.append({
                'file': file,
                'processing_time': processing_time,
                'memory_usage': memory_peak,
                'fitted_ranges': sum(fitacf.qflg > 0)
            })
    
    def print_summary(self):
        total_time = sum(s['processing_time'] for s in self.stats)
        avg_memory = sum(s['memory_usage'] for s in self.stats) / len(self.stats)
        
        print(f"Processed {len(self.stats)} files in {total_time:.2f}s")
        print(f"Average processing time: {total_time/len(self.stats):.3f}s per file")
        print(f"Average memory usage: {avg_memory:.1f}%")

# Use monitoring
monitor = PerformanceMonitor()
monitor.monitor_processing(data_files)
monitor.print_summary()
```

### Getting Help

- **Documentation**: Check the [API Reference](../api/) for detailed function documentation
- **Examples**: See the [examples](../examples/) directory for complete working examples  
- **Issues**: Report bugs on [GitHub Issues](https://github.com/SuperDARN/rst/issues)
- **Community**: Join discussions on [GitHub Discussions](https://github.com/SuperDARN/rst/discussions)
- **Email**: Contact the development team at darn-dawg@isee.nagoya-u.ac.jp

For more advanced topics, see the [Developer Guide](developer_guide.md).