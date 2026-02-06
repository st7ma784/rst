"""
Mapping module for SuperDARN convection map processing

Re-exports convmap functionality for backward compatibility with the
'mapping' module name convention used elsewhere in the codebase.
"""

from .convmap import (
    ConvMapProcessor,
    ConvMapConfig,
    ConvMapData,
)

# Alias for backward compatibility
MapProcessor = ConvMapProcessor
MapConfig = ConvMapConfig
MapData = ConvMapData


def create_map(grid_data, **kwargs):
    """
    Create a convection map from gridded velocity data.
    
    Convenience function wrapping ConvMapProcessor.
    
    Parameters
    ----------
    grid_data : dict
        Gridded velocity data with keys:
        - latitudes: latitude values
        - longitudes: longitude values  
        - velocities: velocity magnitudes
        - azimuths: velocity azimuths
        - weights: optional data quality weights
    **kwargs
        Additional arguments passed to ConvMapConfig
        
    Returns
    -------
    ConvMapData
        Processing results including potential and velocity fields
    """
    config = ConvMapConfig(**kwargs)
    processor = ConvMapProcessor(config)
    return processor.fit(grid_data)


def fit_spherical_harmonics(grid_data, l_max=8, **kwargs):
    """
    Fit spherical harmonics to grid data.
    
    Parameters
    ----------
    grid_data : dict
        Grid data containing velocity measurements
    l_max : int, optional
        Maximum spherical harmonic order (default 8)
    **kwargs
        Additional ConvMapConfig parameters
        
    Returns
    -------
    ConvMapData
        Fitted spherical harmonic coefficients and derived fields
    """
    config = ConvMapConfig(l_max=l_max, **kwargs)
    processor = ConvMapProcessor(config)
    return processor.fit(grid_data)


__all__ = [
    'ConvMapProcessor', 'ConvMapConfig', 'ConvMapData',
    'MapProcessor', 'MapConfig', 'MapData',
    'create_map', 'fit_spherical_harmonics',
]
