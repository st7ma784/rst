"""
Side-by-Side Comparison Test Framework

Provides utilities for comparing C/CUDA and Python implementations
with detailed performance and correctness metrics.
"""

from .framework import ComparisonTestFramework, ModuleComparison, TestResult
from .reporters import ComparisonReporter, JSONReporter, HTMLReporter
from .fixtures import generate_test_data, load_reference_data

__all__ = [
    'ComparisonTestFramework',
    'ModuleComparison', 
    'TestResult',
    'ComparisonReporter',
    'JSONReporter',
    'HTMLReporter',
    'generate_test_data',
    'load_reference_data'
]
