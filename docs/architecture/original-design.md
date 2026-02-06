# Original RST Design

This document describes the original RST architecture before CUDA acceleration, helping developers understand what was transformed and why.

## Historical Context

The Radar Software Toolkit (RST) was developed over decades to process SuperDARN radar data. The original design prioritized:

- **Correctness** - Scientifically accurate algorithms
- **Portability** - Run on various Unix systems  
- **Modularity** - Independent processing components
- **Flexibility** - Handle diverse radar configurations

## Original Architecture

### Processing Model

```
Sequential Processing (Original)
════════════════════════════════

┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Range 1 │───▶│ Range 2 │───▶│ Range 3 │───▶│  ...    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
    │              │              │
    ▼              ▼              ▼
  Process       Process        Process
  (serial)      (serial)       (serial)
```

### Core Data Structures

#### Linked Lists

The original RST heavily used linked lists for dynamic data:

```c
// Original linked list structure (fitacf)
typedef struct ACFData {
    int lag;
    float real_part;
    float imag_part;
    struct ACFData *next;   // Pointer to next element
} ACFData;

// Traversal pattern
ACFData *current = head;
while (current != NULL) {
    process_acf_point(current);
    current = current->next;  // Sequential access
}
```

**Why linked lists were used:**
- Dynamic sizing (unknown data length)
- Easy insertion/deletion during filtering
- Memory efficiency for sparse data
- Natural fit for sequential algorithms

**Problems for GPU:**
- Pointer chasing prevents parallelization
- Poor memory locality/coalescing
- Cannot easily partition for threads
- Unpredictable access patterns

#### llist Library

RST provided a generic linked list library (`rlist`):

```c
// Original llist operations
struct RlistData *rlist_make_empty_list(void);
int rlist_add_node(struct RlistData *list, void *data);
int rlist_remove_node(struct RlistData *list, void *data);
void *rlist_iterate(struct RlistData *list);
void rlist_free_list(struct RlistData *list);
```

### Module Structure

#### FITACF v3.0 (Original)

```
fitacf_v3.0/
├── src/
│   ├── fitacf.c           # Main fitting algorithm
│   ├── calc_phi_res.c     # Phase residual calculation
│   ├── do_fit.c           # Core fitting routines
│   ├── fit_noise.c        # Noise estimation
│   ├── lmfit.c            # Levenberg-Marquardt
│   └── noise_stat.c       # Noise statistics
├── include/
│   └── fitacf.h
└── makefile
```

#### Processing Flow

```c
// Original FITACF processing (simplified)
void fitacf_process(RawACF *raw) {
    // 1. Build linked list of ranges
    RangeList *ranges = create_range_list(raw);
    
    // 2. Filter bad ranges (modifies list)
    filter_bad_lags(ranges);      // O(n)
    
    // 3. Process each range sequentially
    RangeNode *r = ranges->head;
    while (r != NULL) {           // Sequential loop
        // Calculate ACF
        calc_acf(r->data);        // O(lags)
        
        // Fit model
        do_fit(r->data);          // O(iterations)
        
        r = r->next;
    }
    
    // 4. Statistical analysis
    calc_statistics(ranges);      // Sequential
}
```

### Memory Management

#### Allocation Pattern

```c
// Original: Many small allocations
ACFData *acf = malloc(sizeof(ACFData));
acf->next = malloc(sizeof(ACFData));
// ... repeated for each data point

// Later: Individual frees
while (current != NULL) {
    ACFData *next = current->next;
    free(current);
    current = next;
}
```

**Issues:**
- Memory fragmentation
- Allocation overhead
- Poor cache utilization
- Cannot use GPU memory pools

### Algorithm Characteristics

#### FITACF Fitting

```
Original Algorithm Complexity
────────────────────────────

For N ranges, L lags, I iterations:

1. Range loop:           O(N)
2. Per range:
   - Lag processing:     O(L)
   - Fitting:           O(I × L)
   - Noise estimation:   O(L)

Total: O(N × I × L)

With N=75, L=100, I=50:
= 375,000 sequential operations
```

#### Grid Processing

```c
// Original grid cell lookup
GridCell *find_cell(Grid *grid, float lat, float lon) {
    // Linear search through all cells
    GridCell *cell = grid->cells;
    while (cell != NULL) {
        if (cell_contains(cell, lat, lon)) {
            return cell;
        }
        cell = cell->next;
    }
    return NULL;  // O(n) worst case
}
```

### Build System

#### Original Makefile Structure

```makefile
# Top-level coordination
include $(MAKECFG)/makecfg
include $(MAKECFG)/rlib.cfgmk

# Module compilation
OBJS = fitacf.o do_fit.o calc_phi_res.o
LIB = libfitacf.a

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
```

### Performance Characteristics

#### Profiling Results (Original)

```
FITACF CPU Profile (typical):
─────────────────────────────
Function              Time %    Calls
─────────────────────────────────────
do_fit                  45%    75,000
calc_acf                25%    75,000  
noise_estimation        15%    75,000
llist_operations        10%   500,000
memory_allocation        5%   200,000
```

**Bottlenecks identified:**
1. **Sequential range processing** - No parallelism
2. **Linked list traversal** - Cache misses
3. **Repeated allocations** - Overhead
4. **Independent operations** - Could parallelize

### Why CUDA Migration?

#### Parallelism Opportunity

```
Original (Sequential)          CUDA (Parallel)
─────────────────────         ──────────────────

Range 1 ─▶ Result 1           Range 1 ─┐
    │                         Range 2 ─┼─▶ All Results
Range 2 ─▶ Result 2           Range 3 ─┤    (parallel)
    │                          ...    ─┤
Range 3 ─▶ Result 3           Range N ─┘
    │
  ...

Time: O(N)                    Time: O(1) ideally
```

#### Data Parallelism in SuperDARN

| Operation | Parallelizable? | Data Independence |
|-----------|-----------------|-------------------|
| ACF per range | ✅ Yes | Each range independent |
| Lag processing | ✅ Yes | Each lag independent |
| Statistical reduction | ✅ Yes | Parallel reduction |
| Linked list ops | ❌ No | Sequential by design |

## Summary of Limitations

### Original Design Constraints

1. **Serial Execution Model**
   - Single-threaded processing
   - No data parallelism exploited
   - CPU-bound compute

2. **Linked List Dependencies**
   - Pointer-based traversal
   - Cannot partition for threads
   - Poor memory patterns

3. **Memory Fragmentation**
   - Many small allocations
   - No memory pooling
   - CPU-GPU transfer overhead

4. **Algorithm Structure**
   - Loop-carried dependencies assumed
   - No SIMD optimization
   - Sequential mathematical operations

### Migration Requirements

To enable CUDA acceleration:

1. **Replace linked lists** → Arrays with validity masks
2. **Batch operations** → Process multiple ranges together
3. **Unified memory** → Manage CPU/GPU transfers
4. **Preserve APIs** → Maintain backward compatibility

See [Data Structures](data-structures.md) for the specific transformations applied.
