"""
I/O operations for SuperDARN data formats
"""

import warnings

from .readers import load, load_rawacf, load_fitacf, load_grid, load_map

try:
    from .writers import save, save_rawacf, save_fitacf, save_grid, save_map
except ImportError:
    warnings.warn("superdarn_gpu.io.writers is unavailable; save operations are disabled")

    def _writers_unavailable(*args, **kwargs):
        raise NotImplementedError("Writer module is not available in this build")

    save = _writers_unavailable
    save_rawacf = _writers_unavailable
    save_fitacf = _writers_unavailable
    save_grid = _writers_unavailable
    save_map = _writers_unavailable

try:
    from .streaming import DataStreamer, BatchLoader
except ImportError:
    warnings.warn("superdarn_gpu.io.streaming is unavailable; streaming helpers are disabled")

    class DataStreamer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Streaming module is not available in this build")

    class BatchLoader:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Streaming module is not available in this build")

__all__ = [
    'load', 'load_rawacf', 'load_fitacf', 'load_grid', 'load_map',
    'save', 'save_rawacf', 'save_fitacf', 'save_grid', 'save_map',
    'DataStreamer', 'BatchLoader'
]