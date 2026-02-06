"""
GPU-accelerated convection map processing for SuperDARN data

This module implements spherical harmonic fitting and statistical models
to generate global ionospheric convection maps from gridded radar data.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from ..core.backends import get_array_module, get_backend, Backend, synchronize
from ..core.datatypes import GridData
from ..core.pipeline import Stage
from ..core.memory import MemoryMonitor

class ConvectionModel(Enum):
    """Available convection models"""
    STATISTICAL = "statistical"
    EMPIRICAL = "empirical"
    CS10 = "cs10"
    TS18 = "ts18"

class CoordinateSystem(Enum):
    """Coordinate systems"""
    GEOGRAPHIC = "geographic"
    AACGM = "aacgm"  # Altitude Adjusted Corrected Geomagnetic

@dataclass
class ConvMapConfig:
    """Configuration for convection map processing"""
    # Model parameters
    model: ConvectionModel = ConvectionModel.STATISTICAL
    order: int = 8  # Spherical harmonic order
    
    # Coordinate system
    coordinate_system: CoordinateSystem = CoordinateSystem.AACGM
    
    # Boundary parameters
    latmin: float = 60.0  # Minimum latitude (degrees)
    lat_shift: float = 0.0  # Latitude shift for boundary
    
    # IMF parameters
    imf_delay: float = 20.0  # IMF delay from bow shock (minutes)
    
    # Processing options
    error_weighting: bool = True  # Weight by velocity errors
    chi_square_limit: float = 3.0  # Chi-square cutoff
    velocity_limit: float = 2000.0  # Maximum velocity (m/s)
    
    # Model contribution
    model_weight: float = 0.5  # Weight for model vs data (0-1)
    
    # GPU optimization
    use_gpu_fitting: bool = True
    batch_size: int = 1024

@dataclass
class ConvMapData:
    """Convection map output data structure"""
    # Spherical harmonic coefficients
    coefficients: np.ndarray  # Shape: (2, order+1, order+1) - cos/sin
    
    # Potential map
    potential: np.ndarray  # Shape: (nlat, nlon)
    
    # Velocity components
    velocity_north: np.ndarray
    velocity_east: np.ndarray
    velocity_magnitude: np.ndarray
    
    # Grid coordinates
    latitudes: np.ndarray
    longitudes: np.ndarray
    
    # Statistics
    chi_square: float
    chi_square_normalized: float
    n_data_points: int
    n_model_points: int
    
    # Boundary
    hm_boundary: np.ndarray  # Heppner-Maynard boundary
    
    # Metadata
    imf_by: float
    imf_bz: float
    timestamp: str
    model_used: str

class ConvMapProcessor(Stage):
    """
    GPU-accelerated convection map processor for SuperDARN data
    
    Implements spherical harmonic fitting of gridded velocity vectors
    to produce global ionospheric convection patterns.
    """
    
    def __init__(self, config: Optional[ConvMapConfig] = None, **kwargs):
        super().__init__(name="ConvMap Processor", **kwargs)
        
        self.config = config or ConvMapConfig()
        self.xp = get_array_module()
        
        # Initialize GPU kernels if available
        if get_backend() == Backend.CUPY and self.config.use_gpu_fitting:
            self._init_gpu_kernels()
        
        # Precompute associated Legendre polynomials
        self._init_legendre_cache()
        
        # Processing statistics
        self.stats = {
            'vectors_used': 0,
            'vectors_rejected': 0,
            'fitting_iterations': 0,
            'processing_time': 0.0,
            'chi_square': 0.0
        }
    
    def _init_gpu_kernels(self):
        """Initialize CUDA kernels for spherical harmonic fitting"""
        try:
            import cupy as cp
            
            # Spherical harmonic evaluation kernel
            self.sh_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void evaluate_spherical_harmonics(
                const float* theta,      // colatitude in radians
                const float* phi,        // longitude in radians
                float* ylm_cos,          // output: cos(m*phi) * P_l^m(cos(theta))
                float* ylm_sin,          // output: sin(m*phi) * P_l^m(cos(theta))
                const float* plm_cache,  // precomputed associated Legendre
                int n_points,
                int lmax
            ) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= n_points) return;
                
                float t = theta[idx];
                float p = phi[idx];
                float cost = cosf(t);
                
                int coeff_idx = 0;
                for (int l = 0; l <= lmax; l++) {
                    for (int m = 0; m <= l; m++) {
                        // Get precomputed P_l^m
                        float plm = plm_cache[idx * ((lmax+1)*(lmax+2)/2) + l*(l+1)/2 + m];
                        
                        ylm_cos[idx * ((lmax+1)*(lmax+2)/2) + coeff_idx] = plm * cosf(m * p);
                        ylm_sin[idx * ((lmax+1)*(lmax+2)/2) + coeff_idx] = plm * sinf(m * p);
                        coeff_idx++;
                    }
                }
            }
            ''', 'evaluate_spherical_harmonics')
            
            # Potential gradient kernel for velocity calculation
            self.gradient_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void calculate_velocity_from_potential(
                const float* potential,  // potential grid
                float* v_north,          // output: northward velocity
                float* v_east,           // output: eastward velocity
                const float* lat,        // latitude array
                const float* lon,        // longitude array
                float dlat,              // latitude spacing
                float dlon,              // longitude spacing
                int nlat,
                int nlon
            ) {
                int i = blockDim.x * blockIdx.x + threadIdx.x;  // lat index
                int j = blockDim.y * blockIdx.y + threadIdx.y;  // lon index
                
                if (i >= nlat || j >= nlon) return;
                
                int idx = i * nlon + j;
                
                // Earth radius (km) at ionospheric height
                float R = 6371.0f + 300.0f;
                float lat_rad = lat[i] * 3.14159265f / 180.0f;
                
                // d(potential)/d(theta) for v_east
                // d(potential)/d(phi) for v_north (with negative sign)
                
                float dphi_dlat, dphi_dlon;
                
                // Central differences with boundary handling
                if (i > 0 && i < nlat - 1) {
                    dphi_dlat = (potential[(i+1)*nlon + j] - potential[(i-1)*nlon + j]) / (2.0f * dlat);
                } else if (i == 0) {
                    dphi_dlat = (potential[(i+1)*nlon + j] - potential[i*nlon + j]) / dlat;
                } else {
                    dphi_dlat = (potential[i*nlon + j] - potential[(i-1)*nlon + j]) / dlat;
                }
                
                // Longitude wrapping
                int j_minus = (j - 1 + nlon) % nlon;
                int j_plus = (j + 1) % nlon;
                dphi_dlon = (potential[i*nlon + j_plus] - potential[i*nlon + j_minus]) / (2.0f * dlon);
                
                // Convert potential gradient to velocity (m/s)
                // v = -grad(phi) / B, simplified for ionosphere
                float B = 50000e-9f;  // Approximate B field (Tesla)
                
                v_north[idx] = dphi_dlon / (R * cosf(lat_rad) * B) * 1000.0f;
                v_east[idx] = -dphi_dlat / (R * B) * 1000.0f;
            }
            ''', 'calculate_velocity_from_potential')
            
            print("ConvMap GPU kernels initialized successfully")
            
        except Exception as e:
            warnings.warn(f"Could not initialize ConvMap GPU kernels: {e}")
            self.sh_kernel = None
            self.gradient_kernel = None
    
    def _init_legendre_cache(self):
        """Initialize cache for associated Legendre polynomials"""
        # Will be computed for specific latitudes during processing
        self._plm_cache = None
        self._cache_lats = None
    
    def validate_input(self, grid_data: Union[GridData, Dict[str, Any]]) -> bool:
        """Validate input grid data"""
        if grid_data is None:
            return False
            
        if isinstance(grid_data, dict):
            required_keys = ['vectors']
            if not all(key in grid_data for key in required_keys):
                return False
            if not grid_data['vectors'] or len(grid_data['vectors']) == 0:
                return False
        elif isinstance(grid_data, GridData):
            if grid_data.n_vectors == 0:
                return False
        else:
            return False
            
        return True
    
    def get_memory_estimate(self, grid_data: Union[GridData, Dict[str, Any]]) -> int:
        """Estimate memory requirements"""
        if isinstance(grid_data, dict):
            n_vectors = len(grid_data.get('vectors', []))
        else:
            n_vectors = grid_data.n_vectors
            
        order = self.config.order
        n_coeffs = (order + 1) * (order + 2) // 2
        
        # Grid size for output
        nlat = int((90 - self.config.latmin) / 1.0) + 1
        nlon = 360
        
        # Spherical harmonic basis functions
        basis_size = n_vectors * n_coeffs * 2 * 4  # float32
        
        # Output grids
        grid_size = nlat * nlon * 4 * 4  # 4 arrays (potential, v_n, v_e, v_mag)
        
        # Working arrays
        working_size = n_vectors * 4 * 10  # Multiple working arrays
        
        return int((basis_size + grid_size + working_size) * 1.5)
    
    def process(self, grid_data: Union[GridData, Dict[str, Any]], 
                imf_by: float = 0.0,
                imf_bz: float = -5.0) -> ConvMapData:
        """
        Process gridded data to generate convection map
        
        Parameters
        ----------
        grid_data : GridData or dict
            Gridded velocity vectors
        imf_by : float
            IMF By component (nT)
        imf_bz : float
            IMF Bz component (nT)
            
        Returns
        -------
        ConvMapData
            Convection map with potential and velocity fields
        """
        import time
        start_time = time.time()
        
        xp = self.xp
        
        # Extract vectors
        if isinstance(grid_data, dict):
            vectors = grid_data['vectors']
        else:
            vectors = grid_data.vectors
        
        # Convert to arrays
        lats = xp.array([v['lat'] for v in vectors], dtype=xp.float32)
        lons = xp.array([v['lon'] for v in vectors], dtype=xp.float32)
        vels = xp.array([v['velocity'] for v in vectors], dtype=xp.float32)
        azis = xp.array([v['azimuth'] for v in vectors], dtype=xp.float32)
        errs = xp.array([v.get('velocity_error', 50.0) for v in vectors], dtype=xp.float32)
        
        # Filter vectors
        mask = (lats >= self.config.latmin) & (vels < self.config.velocity_limit) & (errs > 0)
        lats = lats[mask]
        lons = lons[mask]
        vels = vels[mask]
        azis = azis[mask]
        errs = errs[mask]
        
        n_vectors = int(len(lats))
        self.stats['vectors_used'] = n_vectors
        self.stats['vectors_rejected'] = len(vectors) - n_vectors
        
        # Convert to velocity components
        azis_rad = xp.radians(azis)
        v_north = vels * xp.cos(azis_rad)
        v_east = vels * xp.sin(azis_rad)
        
        # Calculate weights
        if self.config.error_weighting:
            weights = 1.0 / (errs ** 2)
        else:
            weights = xp.ones_like(errs)
        
        # Build spherical harmonic basis
        order = self.config.order
        n_coeffs = (order + 1) * (order + 2)  # Total coefficients (cos + sin)
        
        # Convert to colatitude and longitude in radians
        theta = xp.radians(90.0 - lats)  # Colatitude
        phi = xp.radians(lons)
        
        # Compute basis functions
        basis_cos, basis_sin = self._compute_spherical_harmonics(theta, phi, order)
        
        # Combine into design matrix
        # For velocity fitting, we need derivatives of spherical harmonics
        basis_matrix = self._build_velocity_design_matrix(
            theta, phi, basis_cos, basis_sin, order, lats
        )
        
        # Stack velocity components
        v_data = xp.concatenate([v_north, v_east])
        w_data = xp.concatenate([weights, weights])
        
        # Add model contribution
        if self.config.model_weight > 0:
            model_v_north, model_v_east = self._get_model_velocities(
                lats, lons, imf_by, imf_bz
            )
            n_model = len(model_v_north)
            
            # Append model data with lower weight
            model_weight = self.config.model_weight / (1.0 - self.config.model_weight + 1e-10)
            v_data = xp.concatenate([v_data, model_v_north, model_v_east])
            w_data = xp.concatenate([w_data, 
                                     xp.ones(n_model) * model_weight,
                                     xp.ones(n_model) * model_weight])
            
            # Extend basis matrix for model points
            model_theta = xp.radians(90.0 - lats)
            model_phi = xp.radians(lons)
            model_basis_cos, model_basis_sin = self._compute_spherical_harmonics(
                model_theta, model_phi, order
            )
            model_basis = self._build_velocity_design_matrix(
                model_theta, model_phi, model_basis_cos, model_basis_sin, order, lats
            )
            basis_matrix = xp.vstack([basis_matrix, model_basis])
        
        # Weighted least squares fit
        W = xp.diag(w_data)
        
        # Normal equations: (A^T W A) x = A^T W b
        AtwA = basis_matrix.T @ W @ basis_matrix
        Atwb = basis_matrix.T @ W @ v_data
        
        # Add regularization
        reg = xp.eye(n_coeffs) * 1e-6
        
        # Solve
        try:
            coefficients = xp.linalg.solve(AtwA + reg, Atwb)
        except Exception:
            # Fallback to least squares
            coefficients, residuals, rank, s = xp.linalg.lstsq(basis_matrix, v_data, rcond=None)
        
        # Calculate chi-square
        residuals = v_data - basis_matrix @ coefficients
        chi_sq = float(xp.sum(w_data * residuals**2))
        chi_sq_norm = chi_sq / (len(v_data) - n_coeffs)
        
        self.stats['chi_square'] = chi_sq_norm
        
        # Generate output grid
        nlat = int((90 - self.config.latmin) / 1.0) + 1
        nlon = 360
        
        out_lats = xp.linspace(self.config.latmin, 90.0, nlat)
        out_lons = xp.linspace(0, 359, nlon)
        
        # Compute potential on grid
        potential = self._compute_potential_grid(
            out_lats, out_lons, coefficients, order
        )
        
        # Compute velocities from potential gradient
        v_n_grid, v_e_grid = self._compute_velocity_grid(
            potential, out_lats, out_lons
        )
        
        v_mag_grid = xp.sqrt(v_n_grid**2 + v_e_grid**2)
        
        # Compute HM boundary
        hm_boundary = self._compute_hm_boundary(imf_by, imf_bz)
        
        # Convert to numpy if using GPU
        if hasattr(potential, 'get'):
            potential = potential.get()
            v_n_grid = v_n_grid.get()
            v_e_grid = v_e_grid.get()
            v_mag_grid = v_mag_grid.get()
            out_lats = out_lats.get()
            out_lons = out_lons.get()
            coefficients = coefficients.get()
        
        # Reshape coefficients
        coeff_matrix = self._reshape_coefficients(coefficients, order)
        
        self.stats['processing_time'] = time.time() - start_time
        
        return ConvMapData(
            coefficients=coeff_matrix,
            potential=potential,
            velocity_north=v_n_grid,
            velocity_east=v_e_grid,
            velocity_magnitude=v_mag_grid,
            latitudes=out_lats,
            longitudes=out_lons,
            chi_square=chi_sq,
            chi_square_normalized=chi_sq_norm,
            n_data_points=n_vectors,
            n_model_points=0,
            hm_boundary=hm_boundary,
            imf_by=imf_by,
            imf_bz=imf_bz,
            timestamp=grid_data.get('timestamp', '') if isinstance(grid_data, dict) else '',
            model_used=self.config.model.value
        )
    
    def _compute_spherical_harmonics(self, theta: np.ndarray, phi: np.ndarray, 
                                      order: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spherical harmonic basis functions
        
        Returns both cosine and sine components for real spherical harmonics.
        """
        xp = self.xp
        n_points = len(theta)
        n_coeffs = (order + 1) * (order + 2) // 2
        
        basis_cos = xp.zeros((n_points, n_coeffs), dtype=xp.float32)
        basis_sin = xp.zeros((n_points, n_coeffs), dtype=xp.float32)
        
        cos_theta = xp.cos(theta)
        
        idx = 0
        for l in range(order + 1):
            for m in range(l + 1):
                # Compute associated Legendre polynomial P_l^m(cos(theta))
                plm = self._associated_legendre(l, m, cos_theta)
                
                # Normalization factor
                norm = xp.sqrt((2*l + 1) / (4 * xp.pi) * 
                              math.factorial(l - m) / math.factorial(l + m))
                
                basis_cos[:, idx] = norm * plm * xp.cos(m * phi)
                basis_sin[:, idx] = norm * plm * xp.sin(m * phi)
                
                idx += 1
        
        return basis_cos, basis_sin
    
    def _associated_legendre(self, l: int, m: int, x: np.ndarray) -> np.ndarray:
        """Compute associated Legendre polynomial P_l^m(x)"""
        xp = self.xp
        
        # Start with P_m^m
        pmm = xp.ones_like(x)
        if m > 0:
            somx2 = xp.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm = -pmm * fact * somx2
                fact += 2.0
        
        if l == m:
            return pmm
        
        # Compute P_{m+1}^m
        pmmp1 = x * (2 * m + 1) * pmm
        
        if l == m + 1:
            return pmmp1
        
        # Use recurrence for l > m + 1
        pll = xp.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll
        
        return pll
    
    def _build_velocity_design_matrix(self, theta, phi, basis_cos, basis_sin, 
                                       order, lats):
        """Build design matrix relating coefficients to velocity components"""
        xp = self.xp
        n_points = len(theta)
        n_coeffs = (order + 1) * (order + 2)  # Combined cos and sin
        
        # For velocity, we need theta and phi derivatives
        # Simplified: use direct relationship between potential SH and velocity
        
        # Combine cos and sin basis into single matrix
        half_coeffs = n_coeffs // 2
        
        # Stack for v_north and v_east
        A_north = xp.hstack([basis_cos, basis_sin])
        A_east = xp.hstack([basis_sin, -basis_cos])  # Note the negative
        
        # Scale by latitude factor for proper geometry
        lat_rad = xp.radians(lats)
        scale = 1.0 / xp.cos(lat_rad)
        scale = xp.clip(scale, 0.1, 10.0)  # Avoid singularity at poles
        
        A_east = A_east * scale[:, None]
        
        # Stack into full matrix
        A = xp.vstack([A_north, A_east])
        
        return A
    
    def _get_model_velocities(self, lats, lons, imf_by, imf_bz):
        """Get statistical model velocities for given positions"""
        xp = self.xp
        
        # Simplified statistical model based on IMF conditions
        # In practice, this would use full CS10 or TS18 models
        
        n_points = len(lats)
        
        # Basic two-cell convection pattern
        # Dawn-dusk asymmetry from IMF By
        by_effect = imf_by / 10.0
        
        # Strength from IMF Bz
        strength = 400.0 * (1.0 + abs(imf_bz) / 5.0)
        
        # Convection pattern
        v_north = xp.zeros(n_points, dtype=xp.float32)
        v_east = xp.zeros(n_points, dtype=xp.float32)
        
        for i in range(n_points):
            lat, lon = float(lats[i]), float(lons[i])
            
            # Distance from pole
            colat = 90.0 - lat
            
            if colat < 30:
                # Inside convection zone
                # Antisunward flow over pole
                mlt = (lon / 15.0) % 24  # Approximate MLT
                
                # Basic convection pattern
                if 6 < mlt < 18:
                    # Dayside
                    v_north[i] = -strength * (colat / 30.0)
                else:
                    # Nightside  
                    v_north[i] = strength * (colat / 30.0)
                
                # Return flow
                v_east[i] = strength * 0.5 * xp.sin(xp.radians(lon)) * (1 + by_effect)
        
        return v_north, v_east
    
    def _compute_potential_grid(self, lats, lons, coefficients, order):
        """Compute electric potential on output grid"""
        xp = self.xp
        
        nlat = len(lats)
        nlon = len(lons)
        
        potential = xp.zeros((nlat, nlon), dtype=xp.float32)
        
        # Convert to grid of theta, phi
        lon_grid, lat_grid = xp.meshgrid(lons, lats)
        theta_grid = xp.radians(90.0 - lat_grid).flatten()
        phi_grid = xp.radians(lon_grid).flatten()
        
        # Compute spherical harmonics on grid
        basis_cos, basis_sin = self._compute_spherical_harmonics(
            theta_grid, phi_grid, order
        )
        
        # Combine and multiply by coefficients
        n_half = len(coefficients) // 2
        pot_flat = (basis_cos @ coefficients[:n_half] + 
                   basis_sin @ coefficients[n_half:])
        
        potential = pot_flat.reshape((nlat, nlon))
        
        return potential
    
    def _compute_velocity_grid(self, potential, lats, lons):
        """Compute velocity from potential gradient"""
        xp = self.xp
        
        nlat, nlon = potential.shape
        dlat = float(lats[1] - lats[0]) if nlat > 1 else 1.0
        dlon = float(lons[1] - lons[0]) if nlon > 1 else 1.0
        
        # Convert dlat/dlon to meters
        R = 6371e3 + 300e3  # Earth radius + ionospheric height
        dlat_m = dlat * xp.pi / 180.0 * R
        
        # Gradient
        dpot_dlat = xp.gradient(potential, dlat_m, axis=0)
        
        # Longitude gradient needs latitude correction
        lat_grid = xp.tile(lats[:, None], (1, nlon))
        cos_lat = xp.cos(xp.radians(lat_grid))
        dlon_m = dlon * xp.pi / 180.0 * R * cos_lat
        
        dpot_dlon = xp.zeros_like(potential)
        for i in range(nlat):
            dpot_dlon[i, :] = xp.gradient(potential[i, :], float(dlon_m[i, 0]))
        
        # Velocity from E × B drift
        # v = E × B / B² ≈ -∇Φ × B̂ / B
        B = 50000e-9  # Approximate B field (T)
        
        v_east = dpot_dlat / B
        v_north = -dpot_dlon / B
        
        return v_north, v_east
    
    def _compute_hm_boundary(self, imf_by, imf_bz):
        """Compute Heppner-Maynard convection boundary"""
        xp = self.xp
        
        # Simplified HM boundary calculation
        # Boundary latitude depends on IMF conditions
        
        base_lat = 65.0
        bz_effect = -imf_bz * 0.5 if imf_bz < 0 else -imf_bz * 0.2
        
        boundary_lat = base_lat + bz_effect
        boundary_lat = max(55.0, min(75.0, boundary_lat))
        
        # Slight MLT variation
        mlts = xp.linspace(0, 24, 49)[:-1]  # 0.5 hour resolution
        boundary = xp.ones(48) * boundary_lat
        
        # Dayside compression
        for i, mlt in enumerate(mlts):
            if 9 < float(mlt) < 15:
                boundary[i] += 2.0  # Higher latitude on dayside
            if 21 < float(mlt) or float(mlt) < 3:
                boundary[i] -= 3.0  # Lower latitude on nightside
        
        if hasattr(boundary, 'get'):
            boundary = boundary.get()
            
        return boundary
    
    def _reshape_coefficients(self, coefficients, order):
        """Reshape flat coefficient array into matrix form"""
        n_half = len(coefficients) // 2
        
        cos_coeffs = np.zeros((order + 1, order + 1))
        sin_coeffs = np.zeros((order + 1, order + 1))
        
        idx = 0
        for l in range(order + 1):
            for m in range(l + 1):
                if idx < n_half:
                    cos_coeffs[l, m] = coefficients[idx]
                    sin_coeffs[l, m] = coefficients[n_half + idx] if n_half + idx < len(coefficients) else 0
                idx += 1
        
        return np.stack([cos_coeffs, sin_coeffs])
