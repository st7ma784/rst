#!/usr/bin/env python3

import sys
sys.path.append('.')
from superdarn_optimization_testing_framework import SuperDARNOptimizationTester

# Test component discovery
t = SuperDARNOptimizationTester('c:/Users/st7ma/Documents/rst')
components = t.find_all_components()

print(f"Found {len(components['libraries'])} libraries:")
for name, paths in list(components['libraries'].items())[:5]:
    opt_status = "optimized" if paths['optimized'] else "original only"
    print(f"  {name}: {opt_status}")

print(f"\nFound {len(components['binaries'])} binaries:")
for name, paths in list(components['binaries'].items())[:5]:
    opt_status = "optimized" if paths['optimized'] else "original only"
    print(f"  {name}: {opt_status}")

# Check if our new optimized component is detected
if 'binplotlib.1.0' in components['libraries']:
    lib = components['libraries']['binplotlib.1.0']
    print(f"\nbinplotlib.1.0 optimization status:")
    print(f"  Original: {lib['original']}")
    print(f"  Optimized: {lib['optimized']}")
