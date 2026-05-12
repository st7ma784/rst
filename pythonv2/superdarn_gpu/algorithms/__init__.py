"""
Core algorithms for SuperDARN processing with GPU acceleration
"""

import warnings

from .fitting import LeastSquaresFitter, PhaseUnwrapper
from .interpolation import SpatialInterpolator
from .lag_validation import LagValidator

try:
    from .statistics import StatisticalProcessor
except ImportError:
    StatisticalProcessor = None  # Optional module in this branch
    warnings.warn("superdarn_gpu.algorithms.statistics is unavailable")

__all__ = [
    'LeastSquaresFitter', 'PhaseUnwrapper',
    'SpatialInterpolator',
    'LagValidator',
]

if StatisticalProcessor is not None:
    __all__.append('StatisticalProcessor')