#!/usr/bin/env python3

import sys
sys.path.append('.')
from superdarn_optimization_testing_framework import SuperDARNOptimizationTester

# Test fitacf component discovery
t = SuperDARNOptimizationTester('c:/Users/st7ma/Documents/rst')
components = t.find_all_components()

print('Checking for fitacf components:')
for name in sorted(components['libraries'].keys()):
    if 'fitacf' in name.lower():
        paths = components['libraries'][name]
        opt_status = 'optimized' if paths['optimized'] else 'original only'
        print(f'  {name}: {opt_status}')
        if paths['optimized']:
            print(f'    Original: {paths["original"]}')
            print(f'    Optimized: {paths["optimized"]}')

print(f'\nTotal libraries found: {len(components["libraries"])}')
print('All library names:')
for name in sorted(components['libraries'].keys()):
    print(f'  {name}')
