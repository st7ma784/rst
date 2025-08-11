"""
High-performance readers for SuperDARN data formats with GPU optimization
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Iterator
import warnings
from datetime import datetime
import struct

import numpy as np
import h5py
import netCDF4 as nc

from ..core.backends import get_array_module, ensure_array, get_backend, Backend
from ..core.datatypes import RawACF, FitACF, GridData, ConvectionMap, RadarParameters
from ..core.memory import check_memory_availability, estimate_memory_requirement

def load(filename: Union[str, Path], 
         data_type: Optional[str] = None,
         use_gpu: Optional[bool] = None,
         **kwargs) -> Union[RawACF, FitACF, GridData, ConvectionMap]:
    """
    Automatically detect and load SuperDARN data file
    
    Parameters
    ----------
    filename : str or Path
        Path to data file
    data_type : str, optional
        Force specific data type ('rawacf', 'fitacf', 'grid', 'map')
    use_gpu : bool, optional
        Load data directly to GPU memory
    **kwargs
        Additional arguments passed to specific loader
        
    Returns
    -------
    RadarData
        Loaded data structure
    """
    filename = Path(filename)
    
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Auto-detect format if not specified
    if data_type is None:
        data_type = _detect_file_format(filename)
    
    # Route to appropriate loader
    loaders = {
        'rawacf': load_rawacf,
        'fitacf': load_fitacf,
        'grid': load_grid,
        'map': load_map
    }
    
    if data_type not in loaders:
        raise ValueError(f"Unknown data type: {data_type}. "
                        f"Supported types: {list(loaders.keys())}")
    
    return loaders[data_type](filename, use_gpu=use_gpu, **kwargs)

def load_rawacf(filename: Union[str, Path], 
                use_gpu: Optional[bool] = None,
                time_range: Optional[tuple] = None,
                beam_filter: Optional[List[int]] = None) -> List[RawACF]:
    """
    Load RawACF data with GPU optimization
    
    Parameters
    ----------
    filename : str or Path
        Path to rawacf file
    use_gpu : bool, optional
        Load data directly to GPU memory
    time_range : tuple, optional
        (start_time, end_time) filter
    beam_filter : List[int], optional
        List of beam numbers to load
        
    Returns
    -------
    List[RawACF]
        List of RawACF records
    """
    filename = Path(filename)
    
    if filename.suffix.lower() in ['.h5', '.hdf5']:
        return _load_rawacf_hdf5(filename, use_gpu, time_range, beam_filter)
    elif filename.suffix.lower() in ['.nc', '.netcdf']:
        return _load_rawacf_netcdf(filename, use_gpu, time_range, beam_filter)
    else:
        # Try legacy DMAP format
        return _load_rawacf_dmap(filename, use_gpu, time_range, beam_filter)

def load_fitacf(filename: Union[str, Path],
                use_gpu: Optional[bool] = None,
                parameters: Optional[List[str]] = None) -> List[FitACF]:
    """
    Load FITACF data with selective parameter loading
    
    Parameters
    ----------
    filename : str or Path
        Path to fitacf file
    use_gpu : bool, optional
        Load data directly to GPU memory
    parameters : List[str], optional
        Specific parameters to load ('velocity', 'power', etc.)
        
    Returns
    -------
    List[FitACF]
        List of FITACF records
    """
    filename = Path(filename)
    
    if filename.suffix.lower() in ['.h5', '.hdf5']:
        return _load_fitacf_hdf5(filename, use_gpu, parameters)
    elif filename.suffix.lower() in ['.nc', '.netcdf']:
        return _load_fitacf_netcdf(filename, use_gpu, parameters)
    else:
        return _load_fitacf_dmap(filename, use_gpu, parameters)

def load_grid(filename: Union[str, Path],
              use_gpu: Optional[bool] = None) -> GridData:
    """
    Load gridded SuperDARN data
    
    Parameters
    ----------
    filename : str or Path
        Path to grid file
    use_gpu : bool, optional
        Load data directly to GPU memory
        
    Returns
    -------
    GridData
        Gridded data
    """
    filename = Path(filename)
    
    if filename.suffix.lower() in ['.h5', '.hdf5']:
        return _load_grid_hdf5(filename, use_gpu)
    elif filename.suffix.lower() in ['.nc', '.netcdf']:
        return _load_grid_netcdf(filename, use_gpu)
    else:
        return _load_grid_dmap(filename, use_gpu)

def load_map(filename: Union[str, Path],
             use_gpu: Optional[bool] = None) -> ConvectionMap:
    """
    Load convection map data
    
    Parameters
    ----------
    filename : str or Path
        Path to map file
    use_gpu : bool, optional
        Load data directly to GPU memory
        
    Returns
    -------
    ConvectionMap
        Convection map data
    """
    filename = Path(filename)
    
    if filename.suffix.lower() in ['.h5', '.hdf5']:
        return _load_map_hdf5(filename, use_gpu)
    elif filename.suffix.lower() in ['.nc', '.netcdf']:
        return _load_map_netcdf(filename, use_gpu)
    else:
        return _load_map_dmap(filename, use_gpu)

# Format detection
def _detect_file_format(filename: Path) -> str:
    """Detect SuperDARN file format from filename and content"""
    
    # Check filename patterns
    name_lower = filename.name.lower()
    
    if 'rawacf' in name_lower or name_lower.endswith('.rawacf'):
        return 'rawacf'
    elif 'fitacf' in name_lower or name_lower.endswith('.fitacf'):
        return 'fitacf'
    elif 'grid' in name_lower or name_lower.endswith('.grid'):
        return 'grid'
    elif 'map' in name_lower or name_lower.endswith('.map'):
        return 'map'
    
    # Try to detect from file content
    try:
        if filename.suffix.lower() in ['.h5', '.hdf5']:
            with h5py.File(filename, 'r') as f:
                if 'rawacf' in f.keys() or 'acf' in f.keys():
                    return 'rawacf'
                elif 'fitacf' in f.keys() or 'velocity' in f.keys():
                    return 'fitacf'
                elif 'grid' in f.keys() or 'lat' in f.keys():
                    return 'grid'
                elif 'map' in f.keys() or 'coeffs' in f.keys():
                    return 'map'
    except:
        pass
    
    # Default fallback
    warnings.warn(f"Could not detect format for {filename}, assuming rawacf")
    return 'rawacf'

# HDF5 loaders (modern format)
def _load_rawacf_hdf5(filename: Path, use_gpu: Optional[bool], 
                      time_range: Optional[tuple], beam_filter: Optional[List[int]]) -> List[RawACF]:
    """Load RawACF from HDF5 format"""
    
    xp = get_array_module()
    records = []
    
    with h5py.File(filename, 'r') as f:
        # Get number of records
        n_records = f.attrs.get('n_records', len(f.keys()))
        
        for i in range(n_records):
            record_key = f"record_{i:06d}"
            if record_key not in f:
                continue
            
            record_group = f[record_key]
            
            # Load radar parameters
            prm_group = record_group['parameters']
            
            # Filter by time if specified
            timestamp = datetime.fromisoformat(prm_group.attrs['timestamp'].decode())
            if time_range and not (time_range[0] <= timestamp <= time_range[1]):
                continue
            
            # Filter by beam if specified
            beam_num = prm_group.attrs['bmnum']
            if beam_filter and beam_num not in beam_filter:
                continue
            
            # Create RadarParameters
            prm = RadarParameters(
                station_id=prm_group.attrs['stid'],
                beam_number=beam_num,
                scan_flag=prm_group.attrs['scan'],
                channel=prm_group.attrs['channel'],
                cp_id=prm_group.attrs['cp'],
                nave=prm_group.attrs['nave'],
                lagfr=prm_group.attrs['lagfr'],
                smsep=prm_group.attrs['smsep'],
                txpow=prm_group.attrs['txpow'],
                atten=prm_group.attrs['atten'],
                noise_search=prm_group.attrs['noise.search'],
                noise_mean=prm_group.attrs['noise.mean'],
                tfreq=prm_group.attrs['tfreq'],
                nrang=prm_group.attrs['nrang'],
                frang=prm_group.attrs['frang'],
                rsep=prm_group.attrs['rsep'],
                xcf=prm_group.attrs['xcf'],
                mppul=prm_group.attrs['mppul'],
                mpinc=prm_group.attrs['mpinc'],
                mplgs=prm_group.attrs['mplgs'],
                txpl=prm_group.attrs['txpl'],
                intt_sc=prm_group.attrs['intt.sc'],
                intt_us=prm_group.attrs['intt.us'],
                timestamp=timestamp
            )
            
            # Create RawACF structure
            rawacf = RawACF(
                nrang=prm.nrang,
                mplgs=prm.mplgs,
                nave=prm.nave,
                use_gpu=use_gpu
            )
            rawacf.prm = prm
            
            # Load data arrays
            data_group = record_group['data']
            
            rawacf.acf = ensure_array(data_group['acf'][:])
            rawacf.xcf = ensure_array(data_group['xcf'][:])
            rawacf.power = ensure_array(data_group['pwr'][:])
            rawacf.noise = ensure_array(data_group['noise'][:])
            rawacf.qflg = ensure_array(data_group['qflg'][:])
            rawacf.gflg = ensure_array(data_group['gflg'][:])
            rawacf.slist = ensure_array(data_group['slist'][:])
            
            records.append(rawacf)
    
    return records

def _load_fitacf_hdf5(filename: Path, use_gpu: Optional[bool], 
                      parameters: Optional[List[str]]) -> List[FitACF]:
    """Load FITACF from HDF5 format"""
    
    records = []
    
    with h5py.File(filename, 'r') as f:
        n_records = f.attrs.get('n_records', len(f.keys()))
        
        for i in range(n_records):
            record_key = f"record_{i:06d}"
            if record_key not in f:
                continue
            
            record_group = f[record_key]
            prm_group = record_group['parameters']
            
            # Create RadarParameters (similar to rawacf)
            timestamp = datetime.fromisoformat(prm_group.attrs['timestamp'].decode())
            prm = RadarParameters(
                station_id=prm_group.attrs['stid'],
                beam_number=prm_group.attrs['bmnum'],
                scan_flag=prm_group.attrs['scan'],
                channel=prm_group.attrs['channel'],
                cp_id=prm_group.attrs['cp'],
                nave=prm_group.attrs['nave'],
                lagfr=prm_group.attrs['lagfr'],
                smsep=prm_group.attrs['smsep'],
                txpow=prm_group.attrs['txpow'],
                atten=prm_group.attrs['atten'],
                noise_search=prm_group.attrs['noise.search'],
                noise_mean=prm_group.attrs['noise.mean'],
                tfreq=prm_group.attrs['tfreq'],
                nrang=prm_group.attrs['nrang'],
                frang=prm_group.attrs['frang'],
                rsep=prm_group.attrs['rsep'],
                xcf=prm_group.attrs['xcf'],
                mppul=prm_group.attrs['mppul'],
                mpinc=prm_group.attrs['mpinc'],
                mplgs=prm_group.attrs['mplgs'],
                txpl=prm_group.attrs['txpl'],
                intt_sc=prm_group.attrs['intt.sc'],
                intt_us=prm_group.attrs['intt.us'],
                timestamp=timestamp
            )
            
            # Create FITACF structure
            fitacf = FitACF(nrang=prm.nrang, use_gpu=use_gpu)
            fitacf.prm = prm
            
            # Load fitted parameters
            data_group = record_group['data']
            
            # Load only requested parameters if specified
            param_map = {
                'velocity': 'v',
                'velocity_error': 'v_e', 
                'power': 'p_l',
                'power_error': 'p_l_e',
                'spectral_width': 'w_l',
                'spectral_width_error': 'w_l_e',
                'phase': 'phi0',
                'elevation': 'elv'
            }
            
            if parameters is None:
                parameters = list(param_map.keys())
            
            for param in parameters:
                if param in param_map and param_map[param] in data_group:
                    array_data = ensure_array(data_group[param_map[param]][:])
                    setattr(fitacf, param, array_data)
            
            # Always load flags and lists
            fitacf.qflg = ensure_array(data_group['qflg'][:])
            fitacf.gflg = ensure_array(data_group['gflg'][:]) 
            fitacf.slist = ensure_array(data_group['slist'][:])
            
            records.append(fitacf)
    
    return records

# Placeholder implementations for other formats
def _load_rawacf_dmap(filename: Path, use_gpu: Optional[bool], 
                      time_range: Optional[tuple], beam_filter: Optional[List[int]]) -> List[RawACF]:
    """Load from legacy DMAP format (to be implemented)"""
    raise NotImplementedError("Legacy DMAP format loading not yet implemented")

def _load_fitacf_dmap(filename: Path, use_gpu: Optional[bool], 
                      parameters: Optional[List[str]]) -> List[FitACF]:
    """Load FITACF from legacy DMAP format"""
    raise NotImplementedError("Legacy DMAP format loading not yet implemented")

def _load_grid_hdf5(filename: Path, use_gpu: Optional[bool]) -> GridData:
    """Load grid from HDF5 format"""
    raise NotImplementedError("Grid HDF5 loading not yet implemented")

def _load_map_hdf5(filename: Path, use_gpu: Optional[bool]) -> ConvectionMap:
    """Load convection map from HDF5 format"""
    raise NotImplementedError("Map HDF5 loading not yet implemented")

# NetCDF format loaders (placeholders)
def _load_rawacf_netcdf(filename: Path, use_gpu: Optional[bool], 
                        time_range: Optional[tuple], beam_filter: Optional[List[int]]) -> List[RawACF]:
    """Load RawACF from NetCDF format"""
    raise NotImplementedError("NetCDF format loading not yet implemented")

def _load_fitacf_netcdf(filename: Path, use_gpu: Optional[bool], 
                        parameters: Optional[List[str]]) -> List[FitACF]:
    """Load FITACF from NetCDF format"""
    raise NotImplementedError("NetCDF format loading not yet implemented")

def _load_grid_netcdf(filename: Path, use_gpu: Optional[bool]) -> GridData:
    """Load grid from NetCDF format"""
    raise NotImplementedError("Grid NetCDF loading not yet implemented")

def _load_map_netcdf(filename: Path, use_gpu: Optional[bool]) -> ConvectionMap:
    """Load map from NetCDF format"""
    raise NotImplementedError("Map NetCDF loading not yet implemented")

def _load_grid_dmap(filename: Path, use_gpu: Optional[bool]) -> GridData:
    """Load grid from DMAP format"""
    raise NotImplementedError("Grid DMAP loading not yet implemented")

def _load_map_dmap(filename: Path, use_gpu: Optional[bool]) -> ConvectionMap:
    """Load map from DMAP format"""
    raise NotImplementedError("Map DMAP loading not yet implemented")