"""
Data writers for SuperDARN file formats

Provides functions to save processed data in various formats including
DMAP, HDF5, and netCDF.
"""

from typing import Any, Union, Optional
from pathlib import Path
import numpy as np


def save(data: Any, filepath: Union[str, Path], format: Optional[str] = None) -> Path:
    """
    Save data to file with auto-detected format
    
    Parameters
    ----------
    data : Any
        Data object to save (RawACF, FitACF, GridData, ConvMapData)
    filepath : str or Path
        Output file path
    format : str, optional
        Force specific format ('dmap', 'hdf5', 'netcdf')
        
    Returns
    -------
    Path
        Path to saved file
    """
    filepath = Path(filepath)
    
    # Auto-detect format from extension
    if format is None:
        ext = filepath.suffix.lower()
        format_map = {
            '.rawacf': 'dmap',
            '.fitacf': 'dmap',
            '.grd': 'dmap',
            '.map': 'dmap',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.nc': 'netcdf',
            '.netcdf': 'netcdf'
        }
        format = format_map.get(ext, 'dmap')
    
    # Dispatch to appropriate writer
    if format == 'hdf5':
        return _save_hdf5(data, filepath)
    elif format == 'netcdf':
        return _save_netcdf(data, filepath)
    else:
        return _save_dmap(data, filepath)


def save_rawacf(data: Any, filepath: Union[str, Path]) -> Path:
    """Save RawACF data to file"""
    return save(data, filepath)


def save_fitacf(data: Any, filepath: Union[str, Path]) -> Path:
    """Save FitACF data to file"""
    return save(data, filepath)


def save_grid(data: Any, filepath: Union[str, Path]) -> Path:
    """Save grid data to file"""
    return save(data, filepath)


def save_map(data: Any, filepath: Union[str, Path]) -> Path:
    """Save convection map data to file"""
    return save(data, filepath)


def _save_dmap(data: Any, filepath: Path) -> Path:
    """Save data in DMAP format"""
    # Placeholder implementation
    # In production, this would use proper DMAP serialization
    import pickle
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    return filepath


def _save_hdf5(data: Any, filepath: Path) -> Path:
    """Save data in HDF5 format"""
    try:
        import h5py
        
        with h5py.File(filepath, 'w') as f:
            # Save data attributes
            for key, value in data.__dict__.items():
                if key.startswith('_'):
                    continue
                    
                if hasattr(value, '__array__'):
                    # NumPy/CuPy array
                    arr = np.array(value.get() if hasattr(value, 'get') else value)
                    f.create_dataset(key, data=arr)
                elif isinstance(value, (int, float, str, bool)):
                    f.attrs[key] = value
                    
    except ImportError:
        raise ImportError("h5py required for HDF5 output. Install with: pip install h5py")
    
    return filepath


def _save_netcdf(data: Any, filepath: Path) -> Path:
    """Save data in NetCDF format"""
    try:
        import netCDF4 as nc
        
        with nc.Dataset(filepath, 'w', format='NETCDF4') as f:
            # Save data attributes
            for key, value in data.__dict__.items():
                if key.startswith('_'):
                    continue
                    
                if hasattr(value, '__array__'):
                    arr = np.array(value.get() if hasattr(value, 'get') else value)
                    
                    # Create dimensions
                    for i, size in enumerate(arr.shape):
                        dim_name = f"{key}_dim{i}"
                        if dim_name not in f.dimensions:
                            f.createDimension(dim_name, size)
                    
                    # Create variable
                    dims = tuple(f"{key}_dim{i}" for i in range(arr.ndim))
                    var = f.createVariable(key, arr.dtype, dims)
                    var[:] = arr
                    
                elif isinstance(value, (int, float, str)):
                    f.setncattr(key, value)
                    
    except ImportError:
        raise ImportError("netCDF4 required for NetCDF output. Install with: pip install netCDF4")
    
    return filepath
