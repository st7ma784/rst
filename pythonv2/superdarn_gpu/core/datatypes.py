"""
GPU-optimized data structures for SuperDARN processing
"""

from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from .backends import get_array_module, ensure_array, to_cpu, to_gpu

@dataclass
class RadarParameters:
    """Basic radar operation parameters"""
    station_id: int
    beam_number: int
    scan_flag: int
    channel: int
    cp_id: int
    nave: int
    lagfr: int
    smsep: int
    txpow: int
    atten: int
    noise_search: float
    noise_mean: float
    tfreq: int
    nrang: int
    frang: int
    rsep: int
    xcf: int
    mppul: int
    mpinc: int
    mplgs: int
    txpl: int
    intt_sc: int
    intt_us: int
    timestamp: datetime

class RadarData:
    """
    Base class for all SuperDARN data structures with GPU support
    """
    
    def __init__(self, use_gpu: Optional[bool] = None):
        """
        Initialize radar data structure
        
        Parameters
        ----------
        use_gpu : bool, optional
            Whether to use GPU memory. If None, uses current backend.
        """
        self.use_gpu = use_gpu
        self.xp = get_array_module()
        self._metadata = {}
    
    def to_gpu(self):
        """Transfer all arrays to GPU"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '__array__') and not attr_name.startswith('_'):
                setattr(self, attr_name, to_gpu(attr))
        return self
    
    def to_cpu(self):
        """Transfer all arrays to CPU"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '__array__') and not attr_name.startswith('_'):
                setattr(self, attr_name, to_cpu(attr))
        return self
    
    def synchronize(self):
        """Synchronize GPU operations"""
        from .backends import synchronize
        synchronize()

class RawACF(RadarData):
    """
    Raw auto-correlation function data with GPU optimization
    """
    
    def __init__(self, 
                 nrang: int,
                 mplgs: int,
                 nave: int = 1,
                 use_gpu: Optional[bool] = None):
        super().__init__(use_gpu)
        
        # Initialize arrays on appropriate backend
        self.nrang = nrang
        self.mplgs = mplgs
        self.nave = nave
        
        # Main data arrays - complex64 for memory efficiency
        self.acf = self.xp.zeros((nrang, mplgs), dtype=self.xp.complex64)
        self.xcf = self.xp.zeros((nrang, mplgs), dtype=self.xp.complex64)
        
        # Real-valued arrays
        self.power = self.xp.zeros(nrang, dtype=self.xp.float32)
        self.noise = self.xp.zeros(nrang, dtype=self.xp.float32)
        
        # Flags and quality indicators
        self.qflg = self.xp.zeros(nrang, dtype=self.xp.int8)
        self.gflg = self.xp.zeros(nrang, dtype=self.xp.int8)
        self.slist = self.xp.zeros(nrang, dtype=self.xp.int16)
        
        # Lag information
        self.lag_power = self.xp.zeros((nrang, mplgs), dtype=self.xp.float32)
        self.lag_phase = self.xp.zeros((nrang, mplgs), dtype=self.xp.float32)
        
        # Radar parameters
        self.prm = None  # Will be set from RadarParameters

class FitACF(RadarData):
    """
    Fitted ACF parameters with GPU optimization
    """
    
    def __init__(self, 
                 nrang: int,
                 use_gpu: Optional[bool] = None):
        super().__init__(use_gpu)
        
        self.nrang = nrang
        
        # Fitted parameters (primary science products)
        self.velocity = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        self.velocity_error = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        self.power = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        self.power_error = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        self.spectral_width = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        self.spectral_width_error = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        
        # Phase information
        self.phase = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        self.elevation = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        
        # Quality flags
        self.qflg = self.xp.zeros(nrang, dtype=self.xp.int8)  # Quality flag
        self.gflg = self.xp.zeros(nrang, dtype=self.xp.int8)  # Ground scatter flag
        self.slist = self.xp.zeros(nrang, dtype=self.xp.int16)  # Range gate list
        
        # Fitting diagnostics
        self.chi2 = self.xp.full(nrang, self.xp.nan, dtype=self.xp.float32)
        self.nlag_fit = self.xp.zeros(nrang, dtype=self.xp.int8)
        
        # Radar parameters
        self.prm = None

class GridData(RadarData):
    """
    Gridded SuperDARN data with spatial interpolation
    """
    
    def __init__(self,
                 nlat: int = 181,  # -90 to +90 degrees
                 nlon: int = 361,  # 0 to 360 degrees  
                 use_gpu: Optional[bool] = None):
        super().__init__(use_gpu)
        
        self.nlat = nlat
        self.nlon = nlon
        
        # Grid coordinates
        self.lat = self.xp.linspace(-90, 90, nlat, dtype=self.xp.float32)
        self.lon = self.xp.linspace(0, 360, nlon, dtype=self.xp.float32)
        
        # Gridded data arrays
        self.velocity = self.xp.full((nlat, nlon), self.xp.nan, dtype=self.xp.float32)
        self.velocity_error = self.xp.full((nlat, nlon), self.xp.nan, dtype=self.xp.float32)
        self.power = self.xp.full((nlat, nlon), self.xp.nan, dtype=self.xp.float32)
        self.spectral_width = self.xp.full((nlat, nlon), self.xp.nan, dtype=self.xp.float32)
        
        # Vector components (magnetic coordinates)
        self.vel_north = self.xp.full((nlat, nlon), self.xp.nan, dtype=self.xp.float32)
        self.vel_east = self.xp.full((nlat, nlon), self.xp.nan, dtype=self.xp.float32)
        
        # Quality and source information
        self.num_points = self.xp.zeros((nlat, nlon), dtype=self.xp.int16)
        self.source_flag = self.xp.zeros((nlat, nlon), dtype=self.xp.int8)
        
        # Grid metadata
        self.start_time = None
        self.end_time = None
        self.grid_type = "regular"  # regular, adaptive, etc.

class ConvectionMap(RadarData):
    """
    Global convection map derived from gridded data
    """
    
    def __init__(self,
                 lmax: int = 8,  # Maximum spherical harmonic degree
                 mmax: Optional[int] = None,  # Maximum order (default: lmax)
                 nlat: int = 181,
                 nlon: int = 361,
                 use_gpu: Optional[bool] = None):
        super().__init__(use_gpu)
        
        self.lmax = lmax
        self.mmax = mmax or lmax
        self.nlat = nlat
        self.nlon = nlon
        
        # Spherical harmonic coefficients
        n_coeffs = (lmax + 1) * (lmax + 2) // 2  # Number of coefficients
        self.coeffs = self.xp.zeros(n_coeffs, dtype=self.xp.complex64)
        self.coeffs_error = self.xp.zeros(n_coeffs, dtype=self.xp.float32)
        
        # Fitted electric potential and field
        self.potential = self.xp.zeros((nlat, nlon), dtype=self.xp.float32)
        self.electric_field = self.xp.zeros((nlat, nlon, 2), dtype=self.xp.float32)  # E_north, E_east
        
        # Derived convection velocity
        self.velocity = self.xp.zeros((nlat, nlon, 2), dtype=self.xp.float32)  # v_north, v_east
        self.velocity_magnitude = self.xp.zeros((nlat, nlon), dtype=self.xp.float32)
        
        # Model parameters
        self.imf_bx = 0.0  # Interplanetary magnetic field
        self.imf_by = 0.0
        self.imf_bz = 0.0
        self.solar_wind_velocity = 400.0
        
        # Fitting statistics
        self.chi2 = 0.0
        self.rms_error = 0.0
        self.num_vectors = 0
        
        # Time information
        self.start_time = None
        self.end_time = None

# Utility functions for data conversion

def rawacf_to_fitacf(rawacf: RawACF, algorithm: str = "v3") -> FitACF:
    """
    Convert RawACF to FitACF using specified algorithm
    
    Parameters
    ----------
    rawacf : RawACF
        Input raw ACF data
    algorithm : str
        Fitting algorithm to use
        
    Returns
    -------
    FitACF
        Fitted parameters
    """
    # This will be implemented in processing modules
    pass

def fitacf_to_grid(fitacf_list: List[FitACF], 
                   grid_resolution: float = 1.0) -> GridData:
    """
    Grid multiple FITACF records onto regular spatial grid
    
    Parameters
    ----------
    fitacf_list : list of FitACF
        Input fitted data
    grid_resolution : float
        Grid resolution in degrees
        
    Returns  
    -------
    GridData
        Gridded data
    """
    # This will be implemented in processing modules
    pass

def grid_to_map(grid: GridData, 
                model_order: int = 8) -> ConvectionMap:
    """
    Generate convection map from gridded data
    
    Parameters
    ----------
    grid : GridData
        Input gridded data
    model_order : int
        Maximum spherical harmonic order
        
    Returns
    -------
    ConvectionMap
        Global convection map
    """
    # This will be implemented in processing modules
    pass