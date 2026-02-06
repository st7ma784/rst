# First Data Processing

This tutorial walks through processing SuperDARN data from raw files to visualizations.

## Overview

You'll learn to:
1. Obtain SuperDARN data
2. Process raw ACF to fitted data
3. Create gridded data products
4. Generate convection maps
5. Create visualizations

**Time**: ~30 minutes

## Step 1: Obtain Data

### Download from FRDR

SuperDARN data is available from [FRDR](https://www.frdr-dfdr.ca/repo/collection/superdarn):

```bash
# Create working directory
mkdir -p ~/superdarn_tutorial && cd ~/superdarn_tutorial

# Download sample data (example - check FRDR for actual URLs)
# Data files are typically named: YYYYMMDD.HH.radar.rawacf.bz2
```

### Use Test Data

For this tutorial, use included test data:

```bash
cd /path/to/rst
ls test_data/
```

## Step 2: Raw to Fit Processing

### Understanding RAWACF and FITACF

- **RAWACF**: Raw auto-correlation functions from radar
- **FITACF**: Fitted parameters (velocity, power, spectral width)

### Process Data

```bash
# Basic fit processing
make_fit test_data/sample.rawacf > sample.fitacf

# With CUDA acceleration (automatic if available)
make_fit test_data/sample.rawacf > sample.fitacf

# Check output
dmapdump sample.fitacf | head -100
```

### Key Parameters

```bash
# Specify algorithm version
make_fit -fitacf-version 3.0 input.rawacf > output.fitacf

# Enable verbose output
make_fit -vb input.rawacf > output.fitacf
```

## Step 3: Create Gridded Data

### Understanding Grids

Gridded data combines multiple radar observations onto a regular spatial grid.

### Generate Grid

```bash
# Create grid from fit file
make_grid sample.fitacf > sample.grid

# With time averaging (2-minute intervals)
make_grid -i 120 sample.fitacf > sample.grid

# Check grid statistics
grid_info sample.grid
```

### Grid Options

| Option | Description | Example |
|--------|-------------|---------|
| `-i` | Time interval (seconds) | `-i 120` |
| `-old` | Use old format | `-old` |
| `-c` | Channel selection | `-c 0` |

## Step 4: Generate Convection Maps

### Map Processing

```bash
# Create potential map from grid
map_grd sample.grid > sample.map

# With specific parameters
map_grd -v sample.grid > sample.map
```

### Map Options

```bash
# Specify hemisphere
map_grd -sh sample.grid > sample.map  # Southern hemisphere

# Set model
map_grd -model PSR sample.grid > sample.map
```

## Step 5: Create Visualizations

### Grid Plots

```bash
# Basic grid plot
grid_plot -png grid_output.png sample.grid

# Customize appearance
grid_plot -png grid_output.png \
    -coast \
    -fov \
    -time 12:00 \
    sample.grid
```

### Map Plots

```bash
# Basic convection map
map_plot -png map_output.png sample.map

# With potential contours
map_plot -png map_output.png \
    -pot \
    -coast \
    sample.map
```

### Plot Options

| Option | Description |
|--------|-------------|
| `-png` | Output PNG file |
| `-ps` | Output PostScript |
| `-coast` | Draw coastlines |
| `-fov` | Show radar field of view |
| `-pot` | Show potential contours |
| `-vec` | Show velocity vectors |

## Complete Pipeline Script

```bash
#!/bin/bash
# complete_processing.sh

INPUT_FILE=$1
OUTPUT_DIR=${2:-./output}

mkdir -p $OUTPUT_DIR

echo "Processing: $INPUT_FILE"

# Step 1: Raw to Fit
echo "Creating FITACF..."
make_fit "$INPUT_FILE" > "$OUTPUT_DIR/data.fitacf"

# Step 2: Fit to Grid
echo "Creating grid..."
make_grid -i 120 "$OUTPUT_DIR/data.fitacf" > "$OUTPUT_DIR/data.grid"

# Step 3: Create Map
echo "Creating map..."
map_grd "$OUTPUT_DIR/data.grid" > "$OUTPUT_DIR/data.map"

# Step 4: Visualizations
echo "Creating plots..."
grid_plot -png "$OUTPUT_DIR/grid.png" -coast "$OUTPUT_DIR/data.grid"
map_plot -png "$OUTPUT_DIR/map.png" -pot -coast "$OUTPUT_DIR/data.map"

echo "Done! Output in $OUTPUT_DIR/"
```

Usage:
```bash
chmod +x complete_processing.sh
./complete_processing.sh test_data/sample.rawacf ./my_output
```

## Performance with CUDA

Compare CPU vs CUDA processing:

```bash
# Time CPU processing
time RST_DISABLE_CUDA=1 make_fit large_file.rawacf > /dev/null

# Time CUDA processing  
time make_fit large_file.rawacf > /dev/null
```

Typical speedups:
- **FITACF processing**: 8-16x faster
- **Grid operations**: 5-10x faster
- **Large datasets**: Best performance gains

## Next Steps

- [User Guide: Make Fit](../user_guide/make_fit.md) - Detailed fit documentation
- [User Guide: Make Grid](../user_guide/make_grid.md) - Grid processing options
- [Architecture Guide](../architecture/) - Understanding the algorithms

## Troubleshooting

### "No data in output"

Check input file format:
```bash
dmapdump input.rawacf | head
```

### "CUDA not available"

Verify CUDA installation:
```bash
nvidia-smi
nvcc --version
```

### Slow processing

Enable CUDA or check system resources:
```bash
# Check if using CUDA
RST_VERBOSE=1 make_fit input.rawacf
```
