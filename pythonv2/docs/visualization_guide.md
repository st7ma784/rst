# SuperDARN GPU Visualization System

## Overview

The SuperDARN GPU visualization system provides comprehensive tools for scientists to understand, monitor, and optimize SuperDARN data processing workflows. The system offers real-time monitoring, interactive exploration, scientific plotting, performance analysis, and quality validation.

## Key Features

### 🔄 Real-time Processing Monitoring
- **Live GPU utilization tracking** - Monitor GPU memory usage, utilization, and temperature
- **Processing stage visualization** - Step through each stage of the processing pipeline
- **Performance metrics** - Real-time throughput and processing efficiency

### 📊 Scientific Visualization
- **Range-time plots** - Traditional SuperDARN data visualization
- **Fan plots** - Field-of-view visualization at specific times
- **ACF analysis** - Auto-correlation function magnitude, phase, and spectrum
- **Statistical summaries** - Data completeness, quality metrics, distributions

### 🎛️ Interactive Exploration
- **Time navigation** - Scrub through observation periods with sliders
- **Beam selection** - Focus on specific radar beams
- **Parameter switching** - Toggle between velocity, power, spectral width
- **Animation controls** - Auto-play through time sequences

### ⚙️ Parameter Tuning
- **Algorithm comparison** - Side-by-side comparison of processing methods
- **Quality assessment** - Automated validation of processing results
- **Performance optimization** - Find optimal parameters for your hardware

## Installation

The visualization system is included with the SuperDARN GPU package:

```python
import superdarn_gpu as sd
from superdarn_gpu.visualization import *
```

### Dependencies
- **Core**: NumPy, matplotlib, datetime
- **GPU acceleration**: CuPy (optional, falls back to NumPy)
- **Interactive widgets**: ipywidgets (for Jupyter notebooks)
- **Performance monitoring**: psutil

## Quick Start

### 1. Basic Scientific Plots

```python
import superdarn_gpu as sd
from superdarn_gpu.visualization import plot_range_time, plot_fan

# Load your SuperDARN data
data = sd.load_data('your_file.fitacf')

# Create range-time plot
plot_range_time(data, parameter='velocity', beam=8)

# Create fan plot at specific time
plot_fan(data, parameter='velocity', time_idx=100)
```

### 2. Interactive Exploration

```python
from superdarn_gpu.visualization import InteractiveExplorer

# Create interactive explorer
explorer = InteractiveExplorer(data, parameters=['velocity', 'power', 'width'])
explorer.show()
```

### 3. Processing Pipeline Monitoring

```python
from superdarn_gpu.visualization import ProcessingViewer

# Create processing viewer
viewer = ProcessingViewer()

# Add processing stages as you process data
viewer.add_processing_stage("Raw Data", raw_data)
viewer.add_processing_stage("ACF Calculation", acf_results)
viewer.add_processing_stage("FitACF Processing", fitacf_results)
viewer.add_processing_stage("Quality Control", final_results)
```

### 4. Performance Dashboard

```python
from superdarn_gpu.visualization import PerformanceDashboard

# Start performance monitoring
dashboard = PerformanceDashboard()
dashboard.start_monitoring()

# Your processing code here
# ...

# Log processing stages
dashboard.log_stage_performance("ACF Calculation", duration_ms=250, data_size_mb=15.2)

# Generate report
dashboard.generate_performance_report()
dashboard.stop_monitoring()
```

## Detailed Usage

### Scientific Plotting Functions

#### `plot_range_time(data, parameter, beam=None, **kwargs)`
Creates traditional range-time plots showing parameter evolution over time.

**Parameters:**
- `data`: SuperDARN data dictionary
- `parameter`: 'velocity', 'power', 'width', or 'elevation'  
- `beam`: Specific beam number (default: average across all beams)
- `time_range`: Tuple of (start_idx, end_idx) for time selection
- `range_limits`: Tuple of (min_range_km, max_range_km)
- `colormap`: Matplotlib colormap name (default: 'RdBu_r' for velocity)
- `ax`: Matplotlib axis object (optional)

**Returns:**
- Matplotlib figure object

#### `plot_fan(data, parameter, time_idx, **kwargs)`
Creates fan plots showing spatial distribution at a specific time.

**Parameters:**
- `data`: SuperDARN data dictionary
- `parameter`: Parameter to plot
- `time_idx`: Time index to plot
- `coords`: 'beam-range' or 'geographic' (if coordinate info available)
- `ax`: Matplotlib axis object (optional)

#### `plot_acf(acf_data, beam, range_gate, **kwargs)`
Visualizes auto-correlation functions.

**Parameters:**
- `acf_data`: Complex ACF array [time, beam, range, lag]
- `beam`: Beam number to plot
- `range_gate`: Range gate number to plot
- `plot_type`: 'magnitude', 'phase', 'real', 'imag', or 'all'

### Interactive Classes

#### `InteractiveExplorer(data, parameters=['velocity'])`
Creates interactive data exploration interface.

**Methods:**
- `show()`: Display the interactive interface
- `set_time_range(start, end)`: Limit time range
- `set_beam_range(start, end)`: Limit beam range
- `add_parameter(name, array)`: Add custom parameter

#### `ParameterTuner(processing_function, data, **kwargs)`
Interactive parameter tuning interface.

**Parameters:**
- `processing_function`: Function that takes data and parameters
- `data`: Input data
- `param_ranges`: Dictionary of parameter names and (min, max, default) values

#### `ProcessingComparison(data, algorithms, labels, **kwargs)`
Side-by-side algorithm comparison.

**Parameters:**
- `data`: Input data
- `algorithms`: List of processing functions
- `labels`: List of algorithm names
- `metrics`: List of comparison metrics to calculate

### Dashboard Creation

#### `create_processing_dashboard(data, results, **kwargs)`
Creates comprehensive processing overview dashboard.

**Parameters:**
- `data`: Original SuperDARN data
- `results`: Dictionary with processing results
- `title`: Dashboard title
- `save_path`: Path to save dashboard image

**Returns:**
- Matplotlib figure with multiple subplots

#### `create_validation_dashboard(original, processed, metrics, **kwargs)`
Creates validation dashboard comparing original and processed data.

**Parameters:**
- `original`: Original data
- `processed`: Processed data
- `metrics`: Validation metrics dictionary
- `statistical_tests`: Enable additional statistical tests

### Real-time Monitoring

#### `RealtimeMonitor(**kwargs)`
Real-time processing and performance monitor.

**Methods:**
- `start_monitoring()`: Begin monitoring
- `update_processing_stage(stage_name, data)`: Update current stage
- `log_performance(metrics_dict)`: Log performance metrics
- `stop_monitoring()`: End monitoring session

#### `GPUMonitor()`
GPU-specific monitoring capabilities.

**Methods:**
- `get_gpu_utilization()`: Current GPU utilization percentage
- `get_memory_usage()`: GPU memory usage in MB
- `get_temperature()`: GPU temperature in Celsius
- `print_gpu_status()`: Print comprehensive GPU status

## Configuration

### Visualization Defaults

You can customize default visualization settings:

```python
import superdarn_gpu.visualization as sdvis

# Set default colormaps
sdvis.config.velocity_colormap = 'RdBu_r'
sdvis.config.power_colormap = 'plasma'
sdvis.config.width_colormap = 'viridis'

# Set default figure sizes
sdvis.config.default_figsize = (12, 8)
sdvis.config.dashboard_figsize = (16, 12)

# Performance monitoring settings
sdvis.config.monitor_update_interval = 1.0  # seconds
sdvis.config.gpu_poll_interval = 0.5  # seconds
```

### Backend Selection

The visualization system automatically detects and uses the best available backend:

```python
import superdarn_gpu as sd

# Check current backend
print(f"Using backend: {sd.get_backend().__name__}")

# Force CPU backend (for testing)
sd.use_cpu_backend()

# Re-enable GPU if available
if sd.GPU_AVAILABLE:
    sd.use_gpu_backend()
```

## Examples

### Complete Processing Pipeline with Visualization

```python
import superdarn_gpu as sd
from superdarn_gpu.visualization import *

# Load data
data = sd.load_fitacf('20220315.1201.00.sas.fitacf')

# Start monitoring
monitor = RealtimeMonitor()
monitor.start_monitoring()

# Processing with visualization
acf_results = sd.calculate_acf(data)
monitor.update_processing_stage("ACF Calculation", acf_results)

fitacf_results = sd.process_fitacf(acf_results)
monitor.update_processing_stage("FitACF Processing", fitacf_results)

filtered_results = sd.apply_quality_filter(fitacf_results)
monitor.update_processing_stage("Quality Control", filtered_results)

# Create comprehensive dashboard
dashboard = create_processing_dashboard(
    data, 
    {
        'acf': acf_results,
        'fitacf': fitacf_results, 
        'filtered': filtered_results
    },
    title="SuperDARN Processing Results"
)

# Save dashboard
dashboard.savefig('processing_results.png', dpi=300, bbox_inches='tight')

monitor.stop_monitoring()
```

### Jupyter Notebook Interactive Analysis

```python
# In Jupyter notebook
%matplotlib widget
import superdarn_gpu as sd
from superdarn_gpu.visualization import *

# Load data
data = sd.load_fitacf('data.fitacf')

# Create interactive explorer
explorer = InteractiveExplorer(data, parameters=['velocity', 'power', 'width'])

# Create parameter tuning interface
def my_processing_algorithm(data, threshold=0.1, smoothing=0.5):
    # Your processing algorithm here
    return processed_data

tuner = ParameterTuner(
    my_processing_algorithm, 
    data,
    param_ranges={
        'threshold': (0.05, 0.5, 0.1),
        'smoothing': (0.0, 1.0, 0.5)
    }
)
```

### Performance Benchmarking

```python
import time
from superdarn_gpu.visualization import PerformanceDashboard

# Create performance dashboard
perf = PerformanceDashboard()
perf.start_monitoring()

# Benchmark different processing approaches
data_sizes = [1, 5, 10, 20, 50]  # MB
results = {}

for size in data_sizes:
    # Generate test data
    test_data = generate_test_data(size_mb=size)
    
    # Time GPU processing
    start_time = time.time()
    gpu_results = sd.process_fitacf_gpu(test_data)
    gpu_time = time.time() - start_time
    
    # Time CPU processing  
    start_time = time.time()
    cpu_results = sd.process_fitacf_cpu(test_data)
    cpu_time = time.time() - start_time
    
    # Log performance
    perf.log_stage_performance(f"GPU_{size}MB", gpu_time*1000, size)
    perf.log_stage_performance(f"CPU_{size}MB", cpu_time*1000, size)
    
    results[size] = {
        'gpu_time': gpu_time,
        'cpu_time': cpu_time,
        'speedup': cpu_time / gpu_time
    }

# Generate performance report
perf.generate_performance_report()
perf.plot_scaling_analysis(results)
```

## Best Practices

### 1. Memory Management
- Use `create_processing_dashboard()` for large datasets instead of individual plots
- Call `plt.close()` on figures you don't need to keep open
- Monitor GPU memory usage with `GPUMonitor` during processing

### 2. Interactive Performance
- Limit time ranges in interactive explorers for better responsiveness
- Use data subsampling for initial exploration of large datasets
- Enable GPU acceleration when available for smoother interactions

### 3. Scientific Visualization
- Use appropriate colormaps for each parameter (velocity: RdBu_r, power: plasma)
- Include proper axis labels and units in all plots
- Add quality flags and data availability information to plots
- Save high-resolution versions (300 DPI) for publications

### 4. Real-time Monitoring
- Use background monitoring for long processing runs
- Log intermediate results for debugging processing issues
- Set appropriate update intervals based on processing speed

## Troubleshooting

### Common Issues

**Import errors with CuPy:**
```python
# The system handles this automatically with fallback to NumPy
import superdarn_gpu as sd
if not sd.GPU_AVAILABLE:
    print("GPU not available, using CPU backend")
```

**Slow interactive performance:**
```python
# Reduce data size for interactive exploration
data_subset = {
    'velocity': data['velocity'][::2, ::2, ::2],  # Subsample
    'power': data['power'][::2, ::2, ::2],
    # ... other parameters
}
explorer = InteractiveExplorer(data_subset)
```

**Memory issues with large datasets:**
```python
# Process data in chunks
chunk_size = 100  # time steps
for i in range(0, data['velocity'].shape[0], chunk_size):
    chunk = extract_time_chunk(data, i, i+chunk_size)
    results = process_chunk(chunk)
    visualize_chunk_results(results)
```

**Dashboard layout issues:**
```python
# Adjust figure size and layout
dashboard = create_processing_dashboard(
    data, results,
    figsize=(20, 14),  # Larger figure
    tight_layout=True
)
```

### Performance Optimization

1. **Enable GPU acceleration** - Install CuPy for significant speedup
2. **Use appropriate data types** - float32 instead of float64 for GPU processing
3. **Batch processing** - Process multiple time periods together
4. **Optimize visualization** - Use data subsampling for large datasets
5. **Monitor resources** - Use performance dashboard to identify bottlenecks

## API Reference

### Core Functions
- `plot_range_time()` - Range-time plotting
- `plot_fan()` - Fan plot visualization  
- `plot_acf()` - Auto-correlation function plots
- `plot_spectrum()` - Spectral analysis plots

### Interactive Classes
- `InteractiveExplorer` - Interactive data exploration
- `ParameterTuner` - Algorithm parameter optimization
- `ProcessingComparison` - Side-by-side algorithm comparison
- `ValidationViewer` - Processing validation interface

### Dashboard Creation
- `create_processing_dashboard()` - Comprehensive processing overview
- `create_performance_dashboard()` - Performance analysis dashboard
- `create_validation_dashboard()` - Validation and quality assessment

### Monitoring Classes
- `RealtimeMonitor` - Real-time processing monitoring
- `PerformanceDashboard` - Performance tracking and analysis
- `GPUMonitor` - GPU-specific monitoring utilities

## Support

For issues, questions, or feature requests:
- Check the examples in `examples/` directory
- Review the Jupyter notebook tutorial
- Submit issues on the project repository
- Consult the API documentation

## Contributing

The visualization system is designed to be extensible:
- Add new plot types in `scientific.py`
- Create custom interactive widgets in `interactive.py`  
- Extend dashboard layouts in `dashboards.py`
- Contribute new monitoring capabilities in `performance.py`

See `CONTRIBUTING.md` for development guidelines.
