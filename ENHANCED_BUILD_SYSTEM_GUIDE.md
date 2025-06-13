# SuperDARN RST Enhanced Build System with Dynamic Optimization

## Overview

The SuperDARN RST Enhanced Build System provides automated detection and use of optimized modules during compilation. Instead of maintaining hard-coded lists of optimized modules, the system dynamically discovers optimized versions using naming conventions, making it scalable as new optimized modules are added.

## Features

### Dynamic Module Detection
- Automatically discovers optimized modules using `*_optimized*` naming patterns
- No need to manually update configuration files for new optimized modules
- Supports multiple optimization levels per module
- Fallback to standard modules when optimized versions are unavailable

### Hardware-Aware Optimization
- Automatic hardware detection (CPU cores, AVX2, OpenMP support)
- Intelligent optimization level recommendations
- Cross-platform compatibility (Linux, macOS, Windows)

### Flexible Build Options
- Multiple optimization levels (none, opt1, opt2, opt3)
- Pattern-based module filtering
- Verbose logging for debugging
- Backward compatibility with existing build system

## Quick Start

### 1. Basic Usage

```bash
# Standard build (same as make.code)
make.code.optimized

# Auto-detect optimal optimization level
make.code.optimized --auto-optimize

# Use specific optimization level
make.code.optimized -o opt2

# Build only libraries with optimization
make.code.optimized -o opt1 lib

# Build specific modules with optimization
make.code.optimized -o opt2 -p grid
```

### 2. Hardware Detection

```bash
# Show hardware capabilities
make.code.optimized --hardware-info

# Get optimization recommendations
make.code.optimized --auto-optimize --help
```

### 3. List Available Optimizations

```bash
# Show all detected optimized modules
make.code.optimized --list-optimizations
```

## Optimization Levels

### none (Default)
- Standard RST build with maximum compatibility
- No special optimizations applied
- Uses original modules only

### opt1 (Basic Optimization)
- OpenMP parallelization where available
- Safe compiler optimizations (-O2)
- Recommended for: Multi-core systems with OpenMP support

### opt2 (Advanced Optimization)  
- OpenMP + SIMD instructions
- AVX2 support where available
- Aggressive compiler optimizations (-O3, -march=native)
- Recommended for: Modern CPUs with AVX2 support

### opt3 (Maximum Optimization)
- All opt2 features plus experimental optimizations
- Link-time optimization (LTO)
- CUDA support where available
- May be unstable, use with caution

## Module Detection System

### Automatic Discovery Patterns

The system searches for optimized modules using these patterns:
- `{module_name}_optimized*`
- `{module_name}.*_optimized*`
- `*{module_name}*_optimized*`
- `{module_name}.optimized*`

### Example Detected Modules

```
grid.1.24 -> grid.1.24_optimized.1
acf.1.16 -> acf.1.16_optimized.2.0
fitacf.2.5 -> fitacf_v3.0_optimized
binplotlib.1.0 -> binplotlib.1.0_optimized.2.0
```

### Validation Requirements

For a module to be considered valid for optimization:
1. Directory must exist in the codebase
2. Must contain `src/makefile`, `CMakeLists.txt`, or `Makefile`
3. Must be buildable with the RST build system

## Enhanced Makefile Templates

### Library Template (makelib.optimized.linux)

Supports dynamic optimization flags:
```makefile
# Optimization flags applied via environment variables
ifdef OPTIMIZATION_FLAGS
CFLAGS += $(BASE_CFLAGS) $(OPTIMIZATION_FLAGS)
endif

# OpenMP support
ifdef OPENMP
CFLAGS += -fopenmp
LFLAGS += -fopenmp
endif

# AVX2 support  
ifdef AVX2
CFLAGS += -mavx2 -mfma -DAVX2_ENABLED
endif
```

### Binary Template (makebin.optimized.linux)

Enhanced with optimization support and performance testing targets.

## Command Line Interface

### Syntax
```
make.code.optimized [type] [options]
```

### Types
- `lib` - Build libraries only
- `bin` - Build binaries only  
- `dlm` - Build IDL DLMs only
- `hdr` - Build headers only
- (none) - Build everything

### Options
- `-o, --optimization LEVEL` - Set optimization level (none,opt1,opt2,opt3)
- `-p, --pattern PATTERN` - Build only modules matching pattern
- `--auto-optimize` - Auto-detect optimal optimization level
- `--list-optimizations` - Show available optimizations and exit
- `--hardware-info` - Show hardware capabilities and exit
- `-v, --verbose` - Verbose output
- `-h, --help` - Show help

### Examples

```bash
# Auto-detect and build with optimal settings
make.code.optimized --auto-optimize

# Build only grid modules with advanced optimization
make.code.optimized -o opt2 -p grid -v

# Show what optimized modules are available
make.code.optimized --list-optimizations

# Check hardware and get recommendations
make.code.optimized --hardware-info
```

## Integration with Existing Workflow

### Backward Compatibility
- Uses same configuration files as original `make.code`
- Falls back to standard modules when optimized versions unavailable
- Maintains same build order and dependencies
- Same environment variable requirements

### Existing Scripts
Replace `make.code` calls with `make.code.optimized`:

```bash
# Old way
make.code lib

# New way with optimization
make.code.optimized -o opt1 lib
```

## Configuration Files

### build_optimized.txt
Contains dynamic optimization detection rules and fallback configurations. The file is designed to be mostly documentation, with the system relying on automatic detection.

### Environment Variables
The system sets these environment variables for optimized builds:
- `OPENMP=1` - Enable OpenMP support
- `SIMD=1` - Enable SIMD instructions  
- `AVX2=1` - Enable AVX2 support
- `CUDA=1` - Enable CUDA support
- `OPTIMIZATION_FLAGS` - Compiler optimization flags

## Adding New Optimized Modules

### Naming Convention
Follow the `*_optimized*` naming pattern:
- `module.version_optimized.optversion`
- `module_optimized.version`
- `module.optimized.version`

### Directory Structure
```
codebase/superdarn/src.lib/tk/mymodule.1.0_optimized.1/
├── src/
│   ├── makefile          # Standard RST makefile
│   └── *.c              # Optimized source files
├── include/
│   └── *.h              # Header files
└── docs/
    └── README.md        # Optimization documentation
```

### Makefile Requirements
The optimized module's makefile should:
1. Include standard RST makefile templates
2. Support optimization environment variables
3. Use same OUTPUT name as original module
4. Follow RST build conventions

Example:
```makefile
include $(MAKECFG).$(SYSTEM)

INCLUDE=-I$(IPATH)/base -I$(IPATH)/superdarn
SRC=optimized_source.c
OBJS=optimized_source.o
OUTPUT=mymodule

# Support optimization flags
ifdef OPENMP
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

ifdef AVX2
CFLAGS += -mavx2 -mfma
endif

include $(MAKELIB).$(SYSTEM)
```

## Validation and Testing

### Validation Script
Use the provided validation script to test the system:

```bash
./validate_optimization_system.sh
```

This tests:
- Environment setup
- Hardware detection
- Dynamic module discovery
- Build system integration
- Specific optimized modules

### Manual Testing
```bash
# Test hardware detection
make.code.optimized --hardware-info

# Test module discovery
make.code.optimized --list-optimizations

# Test build with verbose output
make.code.optimized -o opt1 -v lib
```

## Troubleshooting

### Common Issues

#### No Optimized Modules Found
- Check that optimized modules follow naming conventions
- Verify modules have valid makefiles
- Use `--list-optimizations` to see what's detected

#### Build Failures with Optimization
- Try lower optimization level (opt1 instead of opt2)
- Check compiler support for optimization flags
- Use verbose mode (`-v`) to see detailed build information

#### Environment Variables Not Set
- Ensure RST environment is sourced
- Check `RSTPATH`, `BUILD`, `CODEBASE` variables
- Verify `make.code.optimized` is executable

### Debug Information
Enable verbose mode for detailed information:
```bash
make.code.optimized -v -o opt2 lib 2>&1 | tee build.log
```

## Performance Considerations

### Optimization Level Selection
- **Development**: Use `none` or `opt1` for faster compilation
- **Testing**: Use `opt1` or `opt2` for balanced performance
- **Production**: Use `opt2` for best performance/stability balance
- **Research**: Use `opt3` only after thorough testing

### Hardware Requirements
- **opt1**: Any multi-core system with OpenMP
- **opt2**: Modern CPU with AVX2 support
- **opt3**: High-end system with latest CPU features

### Build Time Impact
- `none`: Baseline build time
- `opt1`: ~10-20% increase
- `opt2`: ~20-40% increase  
- `opt3`: ~50-100% increase (due to LTO)

## Future Enhancements

### Planned Features
1. Profile-guided optimization support
2. Cross-compilation optimization detection
3. Module-specific optimization profiles
4. Build time optimization caching
5. Integration with CI/CD systems

### Contributing Optimized Modules
1. Follow naming conventions
2. Include performance benchmarks
3. Document optimization techniques used
4. Ensure backward compatibility
5. Test across multiple hardware configurations

## Support

For issues or questions:
1. Check this documentation
2. Run validation script
3. Use verbose mode for debugging
4. Check existing optimized modules as examples
5. Contact SuperDARN RST development team

---

**Version**: 1.0  
**Last Updated**: June 2025  
**Compatibility**: RST 4.0+, Linux/macOS/Windows
