# Contributing to RST

Welcome! This guide explains how to contribute to the Radar Software Toolkit.

## Getting Started

### Prerequisites

1. **Fork the repository** on GitHub
2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/rst.git
   cd rst
   ```
3. **Set up upstream:**
   ```bash
   git remote add upstream https://github.com/SuperDARN/rst.git
   ```

### Development Environment

```bash
# Source environment
source .profile.bash

# Build the toolkit
cd build && make

# Run tests
./scripts/ecosystem_validation.sh
```

## Contribution Workflow

### 1. Create a Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-feature
```

### 2. Make Changes

- Write code following the style guide
- Add tests for new functionality
- Update documentation

### 3. Test Your Changes

```bash
# Run full test suite
./scripts/build_and_test_all.sh

# Run specific tests
cd module/tests && ./run_tests.sh

# Check for errors
./scripts/ecosystem_validation.sh
```

### 4. Commit Changes

```bash
# Stage changes
git add -A

# Commit with descriptive message
git commit -m "feat: add CUDA kernel for xyz processing

- Implement parallel processing for xyz
- Add unit tests
- Update documentation"
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/my-feature
```

Then open a Pull Request on GitHub.

---

## Code Style

### C Code

```c
// Use snake_case for functions and variables
int calculate_velocity(float *data, int count);

// Use UPPER_CASE for constants
#define MAX_RANGE_GATES 75

// Use descriptive names
float velocity_error;  // Good
float ve;              // Bad

// Braces on same line
if (condition) {
    do_something();
} else {
    do_other();
}

// Document functions
/**
 * Calculate velocity from ACF data.
 * 
 * @param data  Input ACF data array
 * @param count Number of elements
 * @return Calculated velocity in m/s
 */
float calculate_velocity(float *data, int count);
```

### CUDA Code

```cuda
// Kernel names end with _kernel
__global__ void process_acf_kernel(...);

// Use consistent thread indexing
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// Check bounds
if (idx >= n) return;

// Document shared memory usage
__shared__ float sdata[256];  // For reduction
```

### Python Code

Follow PEP 8:

```python
# Use snake_case for functions
def process_data(input_file: str) -> np.ndarray:
    """Process SuperDARN data file.
    
    Args:
        input_file: Path to input file
        
    Returns:
        Processed data array
    """
    pass

# Use type hints
def calculate_velocity(acf: np.ndarray, lag: int) -> float:
    pass
```

---

## Testing

### Writing Tests

#### C/CUDA Tests

```c
// test_module.c

#include "test_framework.h"
#include "module.h"

void test_basic_functionality(void) {
    // Setup
    float data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Execute
    float result = process(data, 10);
    
    // Verify
    ASSERT_FLOAT_EQUAL(result, 5.5, 0.001);
}

void test_edge_cases(void) {
    // Empty input
    float result = process(NULL, 0);
    ASSERT_EQUAL(result, 0);
}

int main() {
    RUN_TEST(test_basic_functionality);
    RUN_TEST(test_edge_cases);
    return TEST_RESULTS();
}
```

#### Python Tests

```python
# test_module.py

import pytest
import numpy as np
from superdarn_gpu import process_data

def test_basic_processing():
    """Test basic data processing."""
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    result = process_data(data)
    assert np.isclose(result, 3.0, rtol=0.001)

def test_cuda_cpu_equivalence():
    """Ensure CUDA and CPU produce same results."""
    data = np.random.randn(1000).astype(np.float32)
    
    cpu_result = process_data(data, use_cuda=False)
    cuda_result = process_data(data, use_cuda=True)
    
    np.testing.assert_allclose(cpu_result, cuda_result, rtol=1e-5)

@pytest.fixture
def sample_rawacf():
    """Load sample RAWACF file."""
    return load_file("test_data/sample.rawacf")
```

### Running Tests

```bash
# All tests
./scripts/build_and_test_all.sh

# Specific module
cd codebase/superdarn/src.lib/tk/fitacf_v3.0/tests
./run_tests.sh

# Python tests
cd pythonv2
pytest tests/ -v

# With coverage
pytest tests/ --cov=superdarn_gpu
```

---

## Documentation

### Code Documentation

```c
/**
 * @file fitacf.c
 * @brief FITACF processing implementation
 * 
 * This file implements the FITACF algorithm for fitting
 * auto-correlation function data to extract velocity,
 * power, and spectral width.
 */

/**
 * Process ACF data and extract fit parameters.
 * 
 * @param[in]  raw     Raw ACF input data
 * @param[out] fit     Output fit results
 * @param[in]  config  Processing configuration
 * 
 * @return 0 on success, error code on failure
 * 
 * @note Uses CUDA acceleration when available
 * @see cuda_fitacf_process()
 */
int fitacf_process(RawACF *raw, FitACF *fit, Config *config);
```

### User Documentation

Update relevant docs in `docs/`:

```markdown
# Feature Name

Brief description of the feature.

## Usage

```bash
command [options] input > output
```

## Options

| Option | Description |
|--------|-------------|
| `-x` | Description |

## Examples

```bash
# Basic example
command input.dat > output.dat
```
```

---

## Pull Request Guidelines

### PR Title Format

```
type: short description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Adding tests
- perf: Performance improvement
```

### PR Description Template

```markdown
## Description
Brief description of changes.

## Changes
- Change 1
- Change 2

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation updated

## Related Issues
Fixes #123
```

### Review Process

1. **Automated checks** run on PR
2. **Code review** by maintainer
3. **Testing** by community
4. **Merge** when approved

---

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/SuperDARN/rst/discussions)
- **Bugs**: [GitHub Issues](https://github.com/SuperDARN/rst/issues)
- **Email**: Contact DAWG members

---

## Recognition

Contributors are listed in [AUTHORS.md](../../AUTHORS.md). Thank you for contributing!
