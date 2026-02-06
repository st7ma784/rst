# Command Reference

Complete reference for RST command-line tools.

## Processing Commands

### make_fit

Process raw ACF data to fitted parameters.

```bash
make_fit [options] input.rawacf > output.fitacf
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-vb` | Verbose output | Off |
| `-fitacf-version X` | Algorithm version | 3.0 |
| `-c N` | Channel number | 0 |
| `-old` | Use old format | Off |

**Examples:**

```bash
# Basic usage
make_fit data.rawacf > data.fitacf

# Verbose with specific version
make_fit -vb -fitacf-version 3.0 data.rawacf > data.fitacf

# Process specific channel
make_fit -c 1 data.rawacf > data.fitacf
```

---

### make_grid

Create gridded data from fit files.

```bash
make_grid [options] input.fitacf > output.grid
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-i N` | Time interval (seconds) | 120 |
| `-old` | Use old format | Off |
| `-c N` | Channel number | 0 |
| `-cn NAME` | Channel name | - |
| `-ex` | Enable extended output | Off |

**Examples:**

```bash
# Basic gridding
make_grid data.fitacf > data.grid

# 2-minute averaging
make_grid -i 120 data.fitacf > data.grid

# Old format input
make_grid -old data.fitacf > data.grid
```

---

### map_grd

Generate convection maps from gridded data.

```bash
map_grd [options] input.grid > output.map
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-sh` | Southern hemisphere | North |
| `-model NAME` | Statistical model | - |
| `-v` | Verbose output | Off |

**Examples:**

```bash
# Northern hemisphere
map_grd data.grid > data.map

# Southern hemisphere
map_grd -sh data.grid > data.map
```

---

## Visualization Commands

### grid_plot

Create visualizations from grid files.

```bash
grid_plot [options] input.grid
```

**Options:**

| Option | Description |
|--------|-------------|
| `-png FILE` | Output PNG file |
| `-ps FILE` | Output PostScript file |
| `-coast` | Draw coastlines |
| `-fov` | Show field of view |
| `-time HH:MM` | Specific time |
| `-vel` | Show velocity vectors |

**Examples:**

```bash
# Basic PNG output
grid_plot -png output.png data.grid

# With coastlines and FOV
grid_plot -png output.png -coast -fov data.grid

# Specific time
grid_plot -png output.png -time 12:00 data.grid
```

---

### map_plot

Visualize convection maps.

```bash
map_plot [options] input.map
```

**Options:**

| Option | Description |
|--------|-------------|
| `-png FILE` | Output PNG file |
| `-ps FILE` | Output PostScript file |
| `-pot` | Show potential contours |
| `-coast` | Draw coastlines |
| `-vec` | Show velocity vectors |
| `-hmb` | Show Heppner-Maynard boundary |

**Examples:**

```bash
# Basic map plot
map_plot -png output.png data.map

# With potential contours
map_plot -png output.png -pot -coast data.map
```

---

### field_plot

Create field-of-view plots.

```bash
field_plot [options] input.fitacf
```

**Options:**

| Option | Description |
|--------|-------------|
| `-png FILE` | Output PNG |
| `-coast` | Draw coastlines |
| `-param NAME` | Parameter to plot (vel, pwr, wdt) |

---

### time_plot

Create time series plots.

```bash
time_plot [options] input.fitacf
```

**Options:**

| Option | Description |
|--------|-------------|
| `-png FILE` | Output PNG |
| `-b N` | Beam number |
| `-r N` | Range gate |
| `-param NAME` | Parameter (vel, pwr, wdt) |

---

## Utility Commands

### dmapdump

Inspect data file contents.

```bash
dmapdump [options] input_file
```

**Examples:**

```bash
# View file contents
dmapdump data.fitacf | head -100

# Check file structure
dmapdump -l data.fitacf
```

---

### trim

Extract time range from files.

```bash
trim [options] input_file > output_file
```

**Options:**

| Option | Description |
|--------|-------------|
| `-st YYYYMMDD HH:MM` | Start time |
| `-et YYYYMMDD HH:MM` | End time |

---

### combine

Merge multiple files.

```bash
combine [options] file1 file2 ... > output
```

---

## Data Formats

### Input Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| RAWACF | `.rawacf` | Raw auto-correlation functions |
| FITACF | `.fitacf` | Fitted parameters |
| DAT | `.dat` | Legacy format |

### Output Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| FITACF | `.fitacf` | Fitted velocity, power, width |
| GRID | `.grid` | Spatial grid |
| MAP | `.map` | Convection map |
| PNG | `.png` | Image output |
| PS | `.ps` | PostScript output |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RSTPATH` | RST installation path | - |
| `RST_DISABLE_CUDA` | Disable GPU acceleration | 0 |
| `RST_VERBOSE` | Enable verbose output | 0 |
| `RST_CUDA_DEVICE` | Specify CUDA device | 0 |

**Example:**

```bash
# Disable CUDA
export RST_DISABLE_CUDA=1
make_fit data.rawacf > data.fitacf

# Use verbose mode
RST_VERBOSE=1 make_fit data.rawacf > data.fitacf
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Processing error |
| 5 | CUDA error |
