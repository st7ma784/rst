# User Guide

Comprehensive guide for using RST to process SuperDARN data.

```{toctree}
:maxdepth: 2

citing
data
make_fit
make_grid
map_grid
field_plot
grid_plot
map_plot
time_plot
fov_plot
linux_install
mac_install
colors
despecking
```

## Overview

The RST User Guide covers:

- **Data formats** - Understanding SuperDARN data types
- **Processing commands** - `make_fit`, `make_grid`, `map_grd`
- **Visualization** - Creating plots and figures
- **Installation** - Platform-specific guides

## Quick Reference

### Processing Pipeline

```
RAWACF → make_fit → FITACF → make_grid → GRID → map_grd → MAP
```

### Common Commands

| Task | Command |
|------|---------|
| Process raw data | `make_fit input.rawacf > output.fitacf` |
| Create grid | `make_grid input.fitacf > output.grid` |
| Generate map | `map_grd input.grid > output.map` |
| Field plot | `field_plot -png out.png input.fitacf` |
| Grid plot | `grid_plot -png out.png input.grid` |
| Map plot | `map_plot -png out.png input.map` |

## Getting Started

1. [Install RST](../tutorials/installation.md)
2. [Quick Start Tutorial](../tutorials/quickstart.md)
3. [First Data Processing](../tutorials/first-processing.md)
