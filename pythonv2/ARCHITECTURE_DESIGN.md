# SuperDARN Python v2 Architecture Design

## Design Philosophy
Maximize GPU utilization and data throughput while maintaining scientific accuracy and providing a clean, modern Python interface.

## Core Architecture

### 1. **GPU-First Data Structures**
```python
# All data structures default to GPU memory with CUPy arrays
class RadarDataGPU:
    def __init__(self):
        self.data = cp.zeros(shape, dtype=cp.complex64)  # GPU by default
        self.metadata = {}  # CPU metadata
        
class ProcessingPipeline:
    def __init__(self, use_gpu=True):
        self.backend = cp if use_gpu else np  # CUPy or NumPy backend
```

### 2. **Unified Processing Framework**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Loader   │ -> │ Processing Chain │ -> │   Data Writer   │
│  (GPU Upload)   │    │   (CUPy Kernels) │    │ (GPU Download)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────────────┐
                       │ Visualization│
                       │   (GPU Accel)│
                       └──────────────┘
```

### 3. **Module Hierarchy**

```
superdarn_gpu/
├── core/
│   ├── datatypes.py        # GPU-native data structures
│   ├── memory.py          # Memory management & pools
│   ├── pipeline.py        # Processing pipeline framework
│   └── backends.py        # CUPy/NumPy abstraction
├── io/
│   ├── readers.py         # Multi-format data readers
│   ├── writers.py         # Output format writers
│   └── streaming.py       # Streaming I/O for large datasets
├── processing/
│   ├── acf.py            # ACF calculation (GPU kernels)
│   ├── fitacf.py         # FITACF processing (advanced GPU)
│   ├── grid.py           # Spatial gridding (GPU interpolation)
│   ├── mapping.py        # Convection mapping (GPU fitting)
│   └── filters.py        # Signal processing filters
├── algorithms/
│   ├── fitting.py        # Least-squares & optimization (cuSOLVER)
│   ├── interpolation.py  # Spatial interpolation (GPU kernels)
│   ├── statistics.py     # Statistical operations (cuML)
│   └── signal.py         # Signal processing (cuSignal)
├── visualization/
│   ├── plots.py          # GPU-accelerated plotting
│   ├── maps.py           # Geomagnetic coordinate plotting
│   └── interactive.py    # Real-time visualization
└── tools/
    ├── benchmarks.py     # Performance testing
    ├── validation.py     # Scientific validation
    └── conversion.py     # Legacy format conversion
```

## GPU Optimization Strategy

### 1. **Memory Management**
- **Unified Memory**: Use CUPy memory pools for efficient allocation
- **Streaming**: Process large datasets in GPU-resident chunks
- **Caching**: Keep frequently accessed data (calibration, hardware configs) in GPU memory

### 2. **Computational Acceleration**
- **CuPy Kernels**: Custom CUDA kernels for SuperDARN-specific operations
- **cuBLAS**: Matrix operations for fitting algorithms
- **cuFFT**: Fast Fourier transforms for signal processing
- **cuSOLVER**: Linear algebra for least-squares fitting
- **cuSPARSE**: Sparse matrix operations for grid interpolation

### 3. **Algorithmic Improvements**
- **Vectorization**: Process multiple beams/ranges simultaneously
- **Batching**: Group operations for better GPU utilization
- **Async Processing**: Overlap computation with I/O using CUDA streams

## Key Design Decisions

### 1. **Backend Abstraction**
```python
# Seamless CPU/GPU switching
class SuperDARNProcessor:
    def __init__(self, backend='cupy'):
        if backend == 'cupy' and cp.cuda.is_available():
            self.xp = cp
            self.device = 'gpu'
        else:
            self.xp = np
            self.device = 'cpu'
```

### 2. **Data Pipeline Design**
- **Lazy Loading**: Load data only when needed
- **GPU Persistence**: Keep processing chains on GPU
- **Smart Caching**: Cache intermediate results in GPU memory
- **Batch Processing**: Process multiple time periods simultaneously

### 3. **Scientific Validation**
- **Bit-identical Results**: Match original C implementation exactly
- **Comprehensive Testing**: Compare all outputs with reference data
- **Performance Monitoring**: Built-in benchmarking and profiling

## Performance Targets

### Expected Speedups (vs original C code)
- **ACF Processing**: 10-20x (highly parallel)
- **FITACF Processing**: 15-30x (complex math, already shows 16x in CUDA C)
- **Grid Processing**: 8-15x (spatial operations)
- **Mapping**: 5-12x (matrix operations)
- **Overall Pipeline**: 10-25x end-to-end speedup

### Memory Efficiency
- **GPU Memory Usage**: Target <50% of available GPU memory
- **Memory Throughput**: Optimize for GPU memory bandwidth
- **Minimal CPU-GPU Transfers**: Keep data on GPU throughout pipeline

## API Design Philosophy

### Simple and Pythonic
```python
import superdarn_gpu as sd

# Load and process data
data = sd.load('20231201.rawacf')
fitacf = sd.process.fitacf(data, algorithm='v3')
grid = sd.process.grid(fitacf, resolution=0.5)
map_data = sd.process.convection_map(grid)

# Visualize results
sd.plot.range_time(fitacf, parameter='velocity')
sd.plot.convection_map(map_data, timestamp='2023-12-01T12:00:00')
```

### Advanced Usage
```python
# Custom processing pipeline
pipeline = sd.Pipeline()
pipeline.add_stage(sd.ACFProcessor(method='optimized'))
pipeline.add_stage(sd.FitACFProcessor(algorithm='v3', gpu_batch_size=1024))
pipeline.add_stage(sd.GridProcessor(interpolation='gpu_texture'))

# Process with automatic batching and GPU memory management
results = pipeline.process_batch(file_list, batch_size=10)
```

## Migration Path

### Phase 1: Core Infrastructure (Week 1-2)
1. GPU data structures and memory management
2. I/O system with format support
3. Backend abstraction layer

### Phase 2: Processing Modules (Week 3-6)
1. ACF processing with CUPy kernels
2. FITACF processing (port existing CUDA optimizations)
3. Grid processing with GPU interpolation
4. Basic validation framework

### Phase 3: Advanced Features (Week 7-8)
1. Convection mapping
2. Visualization system
3. Performance optimization
4. Comprehensive testing and documentation

This architecture prioritizes GPU performance while maintaining scientific rigor and providing a modern, user-friendly Python interface.