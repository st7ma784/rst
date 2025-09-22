# SuperDARN GPU Visualization Examples

This directory contains comprehensive examples demonstrating the visualization capabilities of the SuperDARN GPU processing system.

## 🚀 Quick Start

### For Scientists New to the System

**Start here:** Run the quick start script to see what the visualization system can do:

```bash
cd examples/
python quick_start_visualization.py
```

This will show you:
- ✅ Basic scientific plots (range-time, fan plots)
- ✅ Comprehensive processing dashboard  
- ✅ Interactive data exploration
- ✅ Performance monitoring

**Arguments:**
- `--save-plots`: Save plots to PNG files
- `--gpu-demo`: Enable GPU-specific features (requires CuPy)
- `--no-interactive`: Skip interactive widgets

### For Interactive Analysis

**Jupyter Notebook Tutorial:** Open the comprehensive tutorial in Jupyter:

```bash
jupyter notebook superdarn_visualization_tutorial.ipynb
```

This notebook covers:
- 📊 Scientific visualization techniques
- 🎛️ Interactive data exploration
- 🔄 Processing pipeline monitoring
- ⚡ Performance optimization
- ✅ Data quality validation
- ⚖️ Algorithm comparison

### For Complete Processing Demonstration

**Full Pipeline Demo:** See the entire processing pipeline with visualization:

```bash
python demo_processing_visualization.py
```

Features:
- 🔬 Synthetic SuperDARN data generation
- 📡 ACF processing simulation
- 🎯 FitACF parameter extraction
- 📈 Real-time progress monitoring
- 🎨 Integrated dashboard creation

**Arguments:**
- `--duration HOURS`: Set simulation duration (default: 2 hours)
- `--beams N`: Number of radar beams (default: 16)
- `--ranges N`: Number of range gates (default: 75)
- `--gpu`: Force GPU processing (requires CuPy)
- `--save-dashboard`: Save dashboard as PNG
- `--interactive`: Enable interactive elements

## 📁 File Overview

### Core Examples

| File | Purpose | Difficulty | GPU Required |
|------|---------|------------|-------------|
| `quick_start_visualization.py` | Introduction to visualization system | Beginner | No |
| `superdarn_visualization_tutorial.ipynb` | Interactive Jupyter tutorial | Beginner | No |
| `demo_processing_visualization.py` | Complete processing pipeline demo | Intermediate | Optional |

### Utility Functions

Each example includes helper functions that you can reuse in your own code:

#### `quick_start_visualization.py`
- `create_sample_data()` - Generate realistic synthetic SuperDARN data
- `demo_basic_plots()` - Create standard scientific visualizations
- `demo_dashboard()` - Build comprehensive processing dashboard
- `demo_interactive_explorer()` - Launch interactive data explorer
- `demo_performance_monitoring()` - Monitor processing performance

#### `demo_processing_visualization.py`
- `generate_synthetic_superdarn_data()` - Advanced synthetic data generation
- `simulate_acf_processing()` - ACF calculation simulation
- `simulate_fitacf_processing()` - FitACF parameter extraction simulation
- `demonstrate_processing_pipeline()` - End-to-end pipeline demonstration

## 🔧 Setup Instructions

### Basic Requirements

```bash
# Install core dependencies
pip install numpy matplotlib datetime

# For SuperDARN GPU package
cd ../
pip install -e .
```

### Optional GPU Acceleration

For significant performance improvements, install CuPy:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x  
pip install cupy-cuda12x

# Verify GPU installation
python -c "import cupy; print('GPU available:', cupy.cuda.is_available())"
```

### Interactive Features

For full interactive capabilities in Jupyter notebooks:

```bash
pip install jupyter ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## 🎯 Usage Patterns

### 1. Exploring New Data

**Start with quick visualization:**
```python
import superdarn_gpu as sd
from superdarn_gpu.visualization import plot_range_time, plot_fan

# Load your data
data = sd.load_fitacf('your_file.fitacf')

# Quick overview
plot_range_time(data, 'velocity')
plot_fan(data, 'velocity', time_idx=100)
```

### 2. Interactive Analysis

**Launch interactive explorer:**
```python
from superdarn_gpu.visualization import InteractiveExplorer

explorer = InteractiveExplorer(data, parameters=['velocity', 'power', 'width'])
explorer.show()
```

### 3. Processing Pipeline Development

**Monitor your processing:**
```python
from superdarn_gpu.visualization import ProcessingViewer

viewer = ProcessingViewer()
viewer.add_processing_stage("Raw Data", raw_data)
# ... your processing steps ...
viewer.add_processing_stage("Final Results", final_data)
```

### 4. Performance Optimization

**Track processing performance:**
```python
from superdarn_gpu.visualization import PerformanceDashboard

perf = PerformanceDashboard()
perf.start_monitoring()
# ... your processing code ...
perf.generate_performance_report()
```

### 5. Algorithm Comparison

**Compare different approaches:**
```python
from superdarn_gpu.visualization import ProcessingComparison

algorithms = [algorithm_v1, algorithm_v2, algorithm_v3]
labels = ['Method A', 'Method B', 'Method C']

comparison = ProcessingComparison(data, algorithms, labels)
comparison.show()
```

## 📊 Example Outputs

### Range-Time Plots
- Traditional SuperDARN visualization showing parameter evolution over time
- Color-coded by velocity, power, or spectral width
- Customizable time and range limits

### Fan Plots  
- Spatial distribution at specific times
- Shows field-of-view coverage
- Useful for understanding ionospheric patterns

### Interactive Dashboards
- Real-time parameter adjustment
- Multiple visualization panels
- Animation controls for time evolution

### Processing Pipelines
- Step-by-step visualization of processing stages
- Before/after comparisons
- Quality metrics at each stage

### Performance Analysis
- GPU utilization and memory usage
- Processing throughput measurements
- Bottleneck identification

## 🔍 Troubleshooting

### Common Issues

**"ImportError: No module named 'superdarn_gpu'"**
```bash
# Make sure you're in the examples/ directory and the package is installed
cd examples/
python -c "import sys; sys.path.insert(0, '..'); import superdarn_gpu"
```

**Interactive widgets not working in Jupyter:**
```bash
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
```

**GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi
# Install appropriate CuPy version
pip install cupy-cuda11x  # or cupy-cuda12x
```

**Matplotlib display issues:**
```python
# For non-interactive environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# For Jupyter notebooks
%matplotlib inline
# or for interactive widgets:
%matplotlib widget
```

**Memory issues with large datasets:**
```python
# Use data subsampling for exploration
data_subset = {
    'velocity': data['velocity'][::2, ::2, ::2],  # Every other point
    'power': data['power'][::2, ::2, ::2],
    # ... other parameters
}
```

### Performance Tips

1. **Enable GPU acceleration** - Install CuPy for 10-100x speedup
2. **Use data subsampling** - For initial exploration of large datasets  
3. **Limit time ranges** - Focus on specific periods of interest
4. **Close unused figures** - Call `plt.close()` to free memory
5. **Use appropriate backends** - Interactive for exploration, Agg for batch processing

## 🎓 Learning Path

### Beginner (New to SuperDARN visualization)
1. Run `quick_start_visualization.py`
2. Open the Jupyter notebook tutorial
3. Modify examples with your own data

### Intermediate (Familiar with SuperDARN data)
1. Run `demo_processing_visualization.py`
2. Experiment with different visualization parameters
3. Create custom processing pipelines with monitoring

### Advanced (Developing new algorithms)
1. Use `ProcessingComparison` for algorithm development
2. Implement custom visualization functions
3. Integrate monitoring into production processing systems

## 📚 Additional Resources

- **Visualization Guide**: `../docs/visualization_guide.md` - Comprehensive API documentation
- **SuperDARN GPU Documentation**: `../README.md` - Main package documentation  
- **Architecture Design**: `../ARCHITECTURE_DESIGN.md` - System design principles
- **Performance Benchmarks**: `../benchmarks/` - Processing performance analysis

## 🤝 Contributing

Found a bug or want to add a new example?

1. Check existing examples for similar functionality
2. Follow the established code structure and documentation style
3. Add appropriate error handling and user-friendly messages
4. Test with both CPU and GPU backends
5. Submit a pull request with description of changes

### Example Template

When creating new examples, use this template:

```python
#!/usr/bin/env python3
"""
Brief description of what this example demonstrates.

Usage:
    python your_example.py [arguments]
"""

import sys
import argparse
sys.path.insert(0, '..')

import superdarn_gpu as sd
from superdarn_gpu.visualization import *

def main():
    """Main function with argument parsing and error handling"""
    parser = argparse.ArgumentParser(description='Your example description')
    # Add arguments...
    
    try:
        # Your example code here
        pass
    except Exception as e:
        print(f"❌ Example failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 📞 Support

For questions about these examples:
- Review the visualization guide documentation
- Check the main package README
- Submit issues with detailed error messages and system information
