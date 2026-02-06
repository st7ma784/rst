"""
Test data fixtures and generators for comparison tests

Provides utilities for generating synthetic test data with known
characteristics for validating algorithm implementations.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path


class DictAsObject:
    """
    Wrapper that allows dict access as attributes.
    Used to simulate dataclass-like objects from dicts.
    """
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictAsObject(value))
            else:
                setattr(self, key, value)
        self._data = data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def __repr__(self):
        return f"DictAsObject({self._data})"


def generate_test_data(module_name: str, 
                       size: str = 'medium',
                       seed: int = 42) -> Dict[str, Any]:
    """
    Generate test data for a specific module
    
    Parameters
    ----------
    module_name : str
        Module name ('acf', 'fitacf', 'grid', 'convmap', etc.)
    size : str
        Data size ('small', 'medium', 'large')
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Test data appropriate for the module
    """
    np.random.seed(seed)
    
    generators = {
        'acf': _generate_acf_test_data,
        'fitacf': _generate_fitacf_test_data,
        'grid': _generate_grid_test_data,
        'convmap': _generate_convmap_test_data,
        'raw': _generate_raw_test_data,
        'fit': _generate_fit_test_data,
    }
    
    if module_name not in generators:
        raise ValueError(f"Unknown module: {module_name}")
        
    return generators[module_name](size)


def _get_size_params(size: str) -> Dict[str, int]:
    """Get size parameters based on size string"""
    sizes = {
        'small': {'nrang': 25, 'mplgs': 18, 'nave': 10, 'nbeams': 4},
        'medium': {'nrang': 75, 'mplgs': 18, 'nave': 50, 'nbeams': 16},
        'large': {'nrang': 100, 'mplgs': 23, 'nave': 100, 'nbeams': 24}
    }
    return sizes.get(size, sizes['medium'])


def _generate_radar_params(size_params: Dict[str, int]) -> DictAsObject:
    """Generate radar parameters as object with attributes"""
    params_dict = {
        'station_id': 65,
        'beam_number': 7,
        'scan_flag': 1,
        'channel': 1,
        'cp_id': 153,
        'nave': size_params['nave'],
        'lagfr': 4800,
        'smsep': 300,
        'txpow': 9000,
        'atten': 0,
        'noise_search': 2.5,
        'noise_mean': 2.3,
        'tfreq': 10500,
        'nrang': size_params['nrang'],
        'frang': 180,
        'rsep': 45,
        'xcf': 1,
        'mppul': 8,
        'mpinc': 1500,
        'mplgs': size_params['mplgs'],
        'txpl': 100,
        'intt_sc': 3,
        'intt_us': 0,
        'timestamp': datetime.now().isoformat()
    }
    return DictAsObject(params_dict)


def _generate_acf_test_data(size: str) -> Dict[str, Any]:
    """Generate ACF test data with synthetic I/Q samples"""
    params = _get_size_params(size)
    prm = _generate_radar_params(params)
    
    nrang = params['nrang']
    nave = params['nave']
    mplgs = params['mplgs']
    
    # Generate synthetic I/Q samples
    nsamp = nrang * nave * 2
    samples = np.zeros(nsamp, dtype=np.complex64)
    
    # Add realistic signal for some range gates
    for range_idx in range(nrang):
        if 10 <= range_idx <= nrang - 10:
            # Ionospheric scatter with Doppler shift
            velocity = 400.0 * np.sin(2 * np.pi * range_idx / nrang)
            frequency = velocity / (3e8 / prm['tfreq'] / 1e3)
            amplitude = 100.0 * np.exp(-range_idx * 0.02)
            
            for avg in range(nave):
                idx = avg * nrang * 2 + range_idx * 2
                phase = 2 * np.pi * frequency * avg * 0.001
                
                samples[idx] = amplitude * np.cos(phase) + np.random.normal(0, 5)
                samples[idx + 1] = amplitude * np.sin(phase) + np.random.normal(0, 5)
        else:
            # Noise-only
            for avg in range(nave):
                idx = avg * nrang * 2 + range_idx * 2
                samples[idx] = np.random.normal(0, 10)
                samples[idx + 1] = np.random.normal(0, 10)
    
    # Generate lag table
    pulse_table = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153]
    lag_table = np.array(pulse_table[:mplgs]) * prm['mpinc']
    
    return {
        'samples': samples,
        'prm': prm,
        'lag_table': lag_table,
        '_expected': {
            'signal_ranges': list(range(10, nrang - 10)),
            'noise_ranges': list(range(0, 10)) + list(range(nrang - 10, nrang))
        }
    }


def _generate_fitacf_test_data(size: str) -> Dict[str, Any]:
    """Generate FITACF test data with synthetic ACF"""
    params = _get_size_params(size)
    prm = _generate_radar_params(params)
    
    nrang = params['nrang']
    mplgs = params['mplgs']
    
    # Generate synthetic ACF with Lorentzian decay
    acf = np.zeros((nrang, mplgs), dtype=np.complex64)
    power = np.zeros(nrang, dtype=np.float32)
    noise = np.ones(nrang, dtype=np.float32) * 10.0
    
    true_params = {}
    
    for r in range(nrang):
        if 15 <= r <= nrang - 15:
            # Known parameters
            true_velocity = 300.0 * np.sin(2 * np.pi * r / nrang)
            true_width = 100.0 + r * 1.5
            true_power = 500.0 * np.exp(-r * 0.03)
            
            true_params[r] = {
                'velocity': true_velocity,
                'spectral_width': true_width,
                'power': true_power
            }
            
            # Generate Lorentzian ACF
            for lag in range(mplgs):
                lag_time = lag * prm['mpinc'] * 1e-6
                
                decay = np.exp(-true_width * lag_time / 100.0)
                phase = true_velocity * lag_time / 200.0
                
                acf_real = true_power * decay * np.cos(phase)
                acf_imag = true_power * decay * np.sin(phase)
                
                # Add noise
                noise_level = true_power * 0.03
                acf_real += np.random.normal(0, noise_level)
                acf_imag += np.random.normal(0, noise_level)
                
                acf[r, lag] = complex(acf_real, acf_imag)
            
            power[r] = np.real(acf[r, 0])
        else:
            # Noise only
            for lag in range(mplgs):
                acf[r, lag] = complex(np.random.normal(0, 5), np.random.normal(0, 5))
            power[r] = abs(acf[r, 0])
    
    return {
        'acf': acf,
        'power': power,
        'noise': noise,
        'prm': prm,
        'slist': np.arange(nrang),
        '_expected': true_params
    }


def _generate_grid_test_data(size: str) -> Dict[str, Any]:
    """Generate grid test data with scattered velocity vectors"""
    params = _get_size_params(size)
    
    # Grid parameters
    lat_min, lat_max = 50.0, 80.0
    lon_min, lon_max = -120.0, -60.0
    
    # Generate scattered vectors
    n_vectors = {
        'small': 200,
        'medium': 1000,
        'large': 5000
    }.get(size, 1000)
    
    vectors = []
    for i in range(n_vectors):
        lat = np.random.uniform(lat_min, lat_max)
        lon = np.random.uniform(lon_min, lon_max)
        
        # Velocity depends on position (simulate convection pattern)
        vel_n = 200.0 * np.sin(np.radians(lat - 65.0)) + np.random.normal(0, 30)
        vel_e = 150.0 * np.cos(np.radians(lon + 90.0)) + np.random.normal(0, 30)
        
        velocity = np.sqrt(vel_n**2 + vel_e**2)
        azimuth = np.degrees(np.arctan2(vel_e, vel_n))
        
        vectors.append({
            'lat': lat,
            'lon': lon,
            'velocity': velocity,
            'azimuth': azimuth,
            'velocity_error': np.random.uniform(10, 50),
            'power': np.random.uniform(10, 100),
            'spectral_width': np.random.uniform(50, 300),
            'station_id': np.random.choice([65, 66, 67, 68]),
            'beam': np.random.randint(0, 16),
            'range_gate': np.random.randint(10, 75)
        })
    
    return {
        'vectors': vectors,
        'grid_config': {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_resolution': 1.0,
            'lon_resolution': 2.0
        },
        'timestamp': datetime.now().isoformat()
    }


def _generate_convmap_test_data(size: str) -> Dict[str, Any]:
    """Generate convection map test data from gridded velocities"""
    # First generate grid data
    grid_data = _generate_grid_test_data(size)
    
    # Additional parameters for convection mapping
    n_coefficients = {
        'small': 8,
        'medium': 16,
        'large': 24
    }.get(size, 16)
    
    return {
        'grid_data': grid_data,
        'model': 'statistical',
        'order': n_coefficients,
        'imf_by': np.random.uniform(-5, 5),
        'imf_bz': np.random.uniform(-10, 0),
        'hemisphere': 'north',
        'timestamp': datetime.now().isoformat()
    }


def _generate_raw_test_data(size: str) -> Dict[str, Any]:
    """Generate raw radar data"""
    params = _get_size_params(size)
    prm = _generate_radar_params(params)
    
    nrang = params['nrang']
    mplgs = params['mplgs']
    
    # Generate raw power data
    pwr0 = np.zeros(nrang, dtype=np.float32)
    acfd = np.zeros((nrang, mplgs, 2), dtype=np.float32)
    xcfd = np.zeros((nrang, mplgs, 2), dtype=np.float32)
    
    for r in range(nrang):
        if 10 <= r <= nrang - 10:
            pwr0[r] = 1000.0 * np.exp(-r * 0.02) + np.random.normal(0, 50)
            
            for lag in range(mplgs):
                decay = np.exp(-lag * 0.1)
                acfd[r, lag, 0] = pwr0[r] * decay + np.random.normal(0, 10)
                acfd[r, lag, 1] = pwr0[r] * decay * 0.1 + np.random.normal(0, 10)
                xcfd[r, lag, 0] = acfd[r, lag, 0] * 0.8
                xcfd[r, lag, 1] = acfd[r, lag, 1] * 0.8
        else:
            pwr0[r] = np.abs(np.random.normal(0, 20))
    
    return {
        'prm': prm,
        'pwr0': pwr0,
        'acfd': acfd,
        'xcfd': xcfd
    }


def _generate_fit_test_data(size: str) -> Dict[str, Any]:
    """Generate fitted data for further processing"""
    params = _get_size_params(size)
    prm = _generate_radar_params(params)
    
    nrang = params['nrang']
    
    # Generate fitted parameters
    velocity = np.zeros(nrang, dtype=np.float32)
    power = np.zeros(nrang, dtype=np.float32)
    width = np.zeros(nrang, dtype=np.float32)
    qflg = np.zeros(nrang, dtype=np.int32)
    gflg = np.zeros(nrang, dtype=np.int32)
    
    for r in range(nrang):
        if 10 <= r <= nrang - 10:
            velocity[r] = 300.0 * np.sin(2 * np.pi * r / nrang) + np.random.normal(0, 20)
            power[r] = 30.0 * np.exp(-r * 0.02) + np.random.normal(0, 2)
            width[r] = 150.0 + r * 1.0 + np.random.normal(0, 10)
            qflg[r] = 1
            gflg[r] = 1 if np.random.random() < 0.1 else 0  # 10% ground scatter
    
    return {
        'prm': prm,
        'velocity': velocity,
        'power': power,
        'spectral_width': width,
        'qflg': qflg,
        'gflg': gflg
    }


def load_reference_data(module_name: str,
                       data_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Load reference test data from files
    
    Parameters
    ----------
    module_name : str
        Module name
    data_dir : Path, optional
        Directory containing reference data
        
    Returns
    -------
    dict or None
        Reference data if available
    """
    if data_dir is None:
        # Try to find test_data directory
        current = Path(__file__).resolve()
        for parent in current.parents:
            test_data = parent / 'test_data'
            if test_data.exists():
                data_dir = test_data
                break
    
    if data_dir is None:
        return None
    
    # Look for module-specific reference data
    patterns = [
        f'{module_name}_reference.npz',
        f'{module_name}_test.npz',
        f'{module_name}.dat'
    ]
    
    for pattern in patterns:
        data_file = data_dir / pattern
        if data_file.exists():
            if data_file.suffix == '.npz':
                return dict(np.load(data_file, allow_pickle=True))
    
    return None


def save_reference_data(data: Dict[str, Any],
                       module_name: str,
                       data_dir: Path) -> Path:
    """
    Save test data as reference for future comparisons
    
    Parameters
    ----------
    data : dict
        Test data to save
    module_name : str
        Module name
    data_dir : Path
        Output directory
        
    Returns
    -------
    Path
        Path to saved file
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / f'{module_name}_reference.npz'
    
    # Convert data for saving
    save_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            save_data[key] = value
        elif isinstance(value, (list, dict)):
            save_data[key] = np.array([value], dtype=object)
        else:
            save_data[key] = np.array([value])
    
    np.savez(output_file, **save_data)
    return output_file
