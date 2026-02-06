"""
GPU-accelerated processing modules for SuperDARN data
"""

from .fitacf import FitACFProcessor, process_fitacf
from .acf import ACFProcessor, calculate_acf
from .grid import GridProcessor, create_grid
from .convmap import ConvMapProcessor, ConvMapConfig, ConvMapData
from .mapping import create_map, fit_spherical_harmonics, MapProcessor

# Make submodules available as module attributes
from . import fitacf
from . import grid
from . import mapping
from . import acf
from . import convmap

__all__ = [
    # Module references
    'fitacf', 'grid', 'mapping', 'acf', 'convmap',
    # FitACF
    'FitACFProcessor', 'process_fitacf',
    # ACF
    'ACFProcessor', 'calculate_acf', 
    # Grid
    'GridProcessor', 'create_grid',
    # ConvMap/Mapping  
    'ConvMapProcessor', 'ConvMapConfig', 'ConvMapData',
    'create_map', 'fit_spherical_harmonics', 'MapProcessor',
]