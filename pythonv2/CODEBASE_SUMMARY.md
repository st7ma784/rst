# RST SuperDARN Codebase Summary

## Overview
The Radar Software Toolkit (RST) is a comprehensive C-based system for processing SuperDARN (Super Dual Auroral Radar Network) radar data. The system processes raw radar measurements through multiple stages to produce scientific data products.

## Current Architecture

### Core Data Flow
1. **Raw Data (rawacf)** → **ACF Calculation** → **FITACF Processing** → **Grid Generation** → **Convection Maps**
2. **IQ Data** → **ACF Processing** → **Fitted Parameters** → **Scientific Products**

### Key Components

#### 1. **Data Structures**
- **RadarParm**: Core radar parameters (beam, frequency, power, timing)
- **RawACF**: Auto-correlation functions from raw radar data
- **FITACF**: Fitted parameters from ACF data (velocity, power, spectral width)
- **GridData**: Spatially gridded radar measurements
- **ConvMap**: Global convection maps derived from multiple radars

#### 2. **Processing Modules**

**ACF Processing (`acf.1.16`)**
- Calculates auto-correlation functions from IQ data
- Handles power calculation, lag normalization, bad lag detection
- Highly computational with array operations

**FITACF Processing (`fitacf_v3.0`)**  
- Fits Lorentzian curves to ACF data to extract Doppler parameters
- Uses least-squares fitting, phase unwrapping, ground scatter detection
- Most computationally intensive module
- Already has CUDA acceleration with up to 16x speedup

**Grid Processing (`grid.1.24`)**
- Spatial interpolation of radar measurements onto regular grids
- Median filtering, range-time processing
- Significant CUDA optimizations available

**Map Generation (`cnvmap.1.17`)**
- Generates global convection maps from gridded data
- Uses spherical harmonic fitting and statistical models
- Complex mathematical operations suitable for GPU acceleration

#### 3. **CUDA Integration Status**
- **13 modules** already CUDA-enabled with sophisticated optimizations
- **28 high-priority modules** identified for CUDA conversion
- Advanced CUDA features: custom kernels, memory management, CPU fallback
- Comprehensive testing framework for CPU vs CUDA validation

### Current Limitations
1. **Complex Build System**: Multiple makefiles, dependencies, version conflicts
2. **Memory Management**: Manual C memory management across large datasets
3. **Scalability**: Limited parallelization beyond CUDA modules
4. **Interoperability**: Difficult integration with modern Python scientific ecosystem
5. **Data Access**: Limited to custom binary formats (DMAP)

## Key Computational Patterns
1. **Array Processing**: Massive parallel operations on radar data arrays
2. **Complex Number Operations**: Phase/amplitude calculations
3. **Matrix Operations**: Linear algebra for fitting algorithms  
4. **Signal Processing**: FFT, filtering, correlation functions
5. **Spatial Interpolation**: Grid mapping and convolution operations
6. **Statistical Fitting**: Least-squares, curve fitting, optimization
7. **Time Series Processing**: Temporal averaging and integration

## Data Formats
- **Input**: Raw IQ data, RAWACF files (binary DMAP format)
- **Intermediate**: FITACF files (fitted parameters)
- **Output**: GRID files, convection maps, scientific visualizations
- **Configuration**: Hardware description files, lookup tables, models

## Performance Characteristics
- **CPU Processing**: Single-threaded, moderate performance
- **CUDA Acceleration**: 5-16x speedup on suitable modules
- **Memory Usage**: Moderate to high for large datasets
- **I/O Bound**: Significant file I/O operations

This codebase is scientifically mature but architecturally complex, making it an excellent candidate for Python modernization with CUPy GPU acceleration.