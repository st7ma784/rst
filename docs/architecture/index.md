# Architecture Documentation

Technical documentation covering RST's internal architecture, the CUDA migration, and design decisions.

```{toctree}
:maxdepth: 2

overview
original-design
cuda-implementation
data-structures
module-architecture
migration-patterns
```

## Documentation Overview

| Document | Audience | Description |
|----------|----------|-------------|
| [Overview](overview.md) | All | High-level architecture summary |
| [Original Design](original-design.md) | Developers | Legacy RST architecture |
| [CUDA Implementation](cuda-implementation.md) | Developers | GPU acceleration details |
| [Data Structures](data-structures.md) | Developers | Old vs new data structures |
| [Module Architecture](module-architecture.md) | Developers | Per-module technical details |
| [Migration Patterns](migration-patterns.md) | Contributors | How to migrate modules |

## Quick Architecture Summary

### Data Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  RAWACF  │────▶│  FITACF  │────▶│   GRID   │────▶│   MAP    │
│ (raw IQ) │     │(fitted)  │     │(gridded) │     │(convect) │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
     │                │                │                │
     ▼                ▼                ▼                ▼
  raw.1.22        fitacf_v3.0      grid.1.24      cnvmap.1.17
  (CUDA ✅)        (CUDA ✅)       (CUDA ✅)       (CUDA ✅)
```

### Key Transformations

| Old (CPU) | New (CUDA) | Benefit |
|-----------|------------|---------|
| Linked lists | Array + validity mask | GPU parallelization |
| Sequential loops | CUDA kernels | Massive parallelism |
| malloc/free | Unified memory | Automatic transfers |
| Single-threaded | Multi-stream | Overlap compute/transfer |

### Performance Gains

- **FITACF**: 8-16x speedup
- **Grid Operations**: 5-10x speedup  
- **ACF Processing**: 20-60x speedup
- **Overall Pipeline**: 5-30x speedup
