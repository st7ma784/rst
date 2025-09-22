# SuperDARN Test Data

This directory contains synthetic SuperDARN test data for validating the CUDA implementation.

## File Formats

- `.bin` files: Binary format for efficient loading in C/CUDA programs
- `.txt` files: Human-readable text format for inspection
- `.h` files: C header files with static arrays for direct inclusion

## Data Sets

- `small`: 25 ranges, 17 lags - for quick testing
- `medium`: 75 ranges, 17 lags - typical SuperDARN size
- `large`: 150 ranges, 17 lags - stress testing

## Usage

Use these files to test CPU vs CUDA implementations and validate numerical accuracy.
The data includes realistic physics-based ACF patterns with appropriate noise levels.
