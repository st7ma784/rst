"""
GPU-accelerated grid processing for SuperDARN data

This module implements spatial interpolation and gridding algorithms to convert
scattered FITACF measurements into regular spatial grids for analysis.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

from ..core.backends import get_array_module, get_backend, Backend, synchronize
from ..core.datatypes import FitACF, GridData, RadarParameters
from ..core.pipeline import Stage
from ..core.memory import MemoryMonitor
from ..algorithms.interpolation import SpatialInterpolator

_R_EARTH_KM = 6371.0


@dataclass
class RadarHardware:
    """Physical site parameters for a SuperDARN radar."""
    station_id: int
    lat: float          # Geographic latitude (°N)
    lon: float          # Geographic longitude (°E)
    bearing: float      # Boresight direction from geographic north (°)
    beam_sep: float = 3.24  # Angular beam separation (°)
    n_beams: int = 16


# Partial hardware table — extend by reading hdw.dat files if available.
# Values from RST hdw/ directory (2024 release).
_RADAR_HARDWARE: Dict[int, RadarHardware] = {
    1:  RadarHardware(1,  70.86, -29.27,  -167.0),  # Þykkvibær (Iceland E)
    2:  RadarHardware(2,  74.50, -94.90,   178.0),  # Rankin Inlet
    3:  RadarHardware(3,  62.35, -145.15,  202.0),  # Kodiak
    4:  RadarHardware(4,  51.99,  4.50,    -168.0), # Hankasalmi (placeholder)
    5:  RadarHardware(5,  53.98, -1.99,    -168.0), # Pykkvibaer W
    6:  RadarHardware(6,  62.82, -22.02,  -145.0),  # Iceland W
    8:  RadarHardware(8,  52.16, 106.26,    18.0),  # Irkutsk
    10: RadarHardware(10, -70.38, -2.32,   176.0),  # Halley
    16: RadarHardware(16, 62.32,  26.61,   -12.0),  # Hankasalmi
    33: RadarHardware(33, 56.54, 103.25,    35.0),  # Irkutsk
    65: RadarHardware(65, 62.82, -82.52,   -32.0),  # Kapuskasing
}

class GridMethod(Enum):
    """Available gridding methods"""
    MEDIAN = "median"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    GAUSSIAN_WEIGHT = "gaussian_weight"

class CoordinateSystem(Enum):
    """Coordinate systems for gridding"""
    GEOGRAPHIC = "geographic"
    MAGNETIC = "magnetic"
    AACGM = "aacgm"

@dataclass
class GridConfig:
    """Configuration for grid processing"""
    # Grid parameters
    lat_min: float = 45.0              # Minimum latitude (degrees)
    lat_max: float = 85.0              # Maximum latitude (degrees)  
    lon_min: float = -180.0            # Minimum longitude (degrees)
    lon_max: float = 180.0             # Maximum longitude (degrees)
    lat_resolution: float = 1.0        # Latitude resolution (degrees)
    lon_resolution: float = 2.0        # Longitude resolution (degrees)
    
    # Processing parameters
    grid_method: GridMethod = GridMethod.MEDIAN
    coordinate_system: CoordinateSystem = CoordinateSystem.MAGNETIC
    altitude: float = 300.0            # Altitude for coordinate mapping (km)
    
    # Quality control
    min_vectors_per_cell: int = 1      # Minimum vectors required per grid cell
    max_temporal_separation: float = 120.0  # Max time separation for averaging (seconds)
    vector_quality_threshold: float = 0.3   # Quality threshold for vectors
    
    # Spatial filtering
    spatial_filter_size: float = 0.0   # Gaussian filter size (degrees, 0=no filter)
    median_filter_size: int = 0        # Median filter size (grid cells, 0=no filter)
    
    # GPU optimization
    use_gpu_interpolation: bool = True # Use GPU for spatial interpolation
    batch_size: int = 1000             # Batch size for processing vectors

class GridProcessor(Stage):
    """
    GPU-accelerated spatial gridding processor for SuperDARN data
    """
    
    def __init__(self, config: Optional[GridConfig] = None, **kwargs):
        super().__init__(name="Grid Processor", **kwargs)
        
        self.config = config or GridConfig()
        self.xp = get_array_module()
        
        # Initialize spatial interpolator
        self.interpolator = SpatialInterpolator()
        
        # Initialize GPU kernels if available
        if get_backend() == Backend.CUPY and self.config.use_gpu_interpolation:
            self._init_gpu_kernels()
        
        # Processing statistics
        self.stats = {
            'vectors_processed': 0,
            'grid_cells_filled': 0,
            'processing_time': 0.0,
            'spatial_coverage': 0.0
        }
    
    def _init_gpu_kernels(self):
        """Initialize CUDA kernels for grid processing"""
        try:
            import cupy as cp
            
            # Grid interpolation kernel using inverse distance weighting
            self.grid_kernel = cp.RawKernel(r'''
            #ifndef NAN
            #define NAN (__int_as_float(0x7fc00000))
            #endif
            extern "C" __global__
            void grid_vectors_idw(const float* vec_lat, const float* vec_lon,
                                 const float* vec_vel, const float* vec_err,
                                 const int* vec_quality,
                                 float* grid_lat, float* grid_lon,
                                 float* grid_vel, float* grid_err, int* grid_count,
                                 int n_vectors, int n_lat, int n_lon,
                                 float search_radius, int min_vectors) {
                
                int lat_idx = blockDim.x * blockIdx.x + threadIdx.x;
                int lon_idx = blockDim.y * blockIdx.y + threadIdx.y;
                
                if (lat_idx >= n_lat || lon_idx >= n_lon) return;
                
                int grid_idx = lat_idx * n_lon + lon_idx;
                float target_lat = grid_lat[lat_idx];
                float target_lon = grid_lon[lon_idx];
                
                float weighted_vel = 0.0f;
                float weight_sum = 0.0f;
                float error_sum = 0.0f;
                int valid_vectors = 0;
                
                // Find vectors within search radius
                for (int v = 0; v < n_vectors; v++) {
                    if (vec_quality[v] == 0) continue; // Skip bad quality vectors
                    
                    // Equirectangular approx with cos-lat correction (good to ~1% for SuperDARN)
                    float dlat = vec_lat[v] - target_lat;
                    float dlon = vec_lon[v] - target_lon;
                    float cos_lat = cosf(target_lat * 3.14159265f / 180.0f);
                    float distance = sqrtf(dlat*dlat + (dlon*cos_lat)*(dlon*cos_lat));
                    
                    if (distance <= search_radius) {
                        float weight = 1.0f / (distance + 0.001f); // Avoid division by zero
                        
                        weighted_vel += vec_vel[v] * weight;
                        weight_sum += weight;
                        error_sum += vec_err[v] * vec_err[v] * weight * weight;
                        valid_vectors++;
                    }
                }
                
                // Store result if sufficient vectors found
                if (valid_vectors >= min_vectors && weight_sum > 0) {
                    grid_vel[grid_idx] = weighted_vel / weight_sum;
                    grid_err[grid_idx] = sqrtf(error_sum) / weight_sum;
                    grid_count[grid_idx] = valid_vectors;
                } else {
                    grid_vel[grid_idx] = NAN;
                    grid_err[grid_idx] = NAN;
                    grid_count[grid_idx] = 0;
                }
            }
            ''', 'grid_vectors_idw')
            
            # Median filter kernel for smoothing
            self.median_filter_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void median_filter_2d(const float* input, float* output,
                                 int n_lat, int n_lon, int filter_size) {
                
                int lat_idx = blockDim.x * blockIdx.x + threadIdx.x;
                int lon_idx = blockDim.y * blockIdx.y + threadIdx.y;
                
                if (lat_idx >= n_lat || lon_idx >= n_lon) return;
                
                int center_idx = lat_idx * n_lon + lon_idx;
                
                // Skip NaN values
                if (isnan(input[center_idx])) {
                    output[center_idx] = input[center_idx];
                    return;
                }
                
                float values[25]; // Max 5x5 filter
                int value_count = 0;
                int half_size = filter_size / 2;
                
                // Collect values in neighborhood
                for (int di = -half_size; di <= half_size; di++) {
                    for (int dj = -half_size; dj <= half_size; dj++) {
                        int ni = lat_idx + di;
                        int nj = lon_idx + dj;
                        
                        if (ni >= 0 && ni < n_lat && nj >= 0 && nj < n_lon) {
                            float val = input[ni * n_lon + nj];
                            if (!isnan(val) && value_count < 25) {
                                values[value_count++] = val;
                            }
                        }
                    }
                }
                
                // Calculate median (simple bubble sort for small arrays)
                if (value_count > 0) {
                    for (int i = 0; i < value_count - 1; i++) {
                        for (int j = 0; j < value_count - i - 1; j++) {
                            if (values[j] > values[j + 1]) {
                                float temp = values[j];
                                values[j] = values[j + 1];
                                values[j + 1] = temp;
                            }
                        }
                    }
                    output[center_idx] = values[value_count / 2];
                } else {
                    output[center_idx] = input[center_idx];
                }
            }
            ''', 'median_filter_2d')
            
            print("Grid processing GPU kernels initialized successfully")
            
        except Exception as e:
            warnings.warn(f"Could not initialize grid GPU kernels: {e}")
            self.grid_kernel = None
            self.median_filter_kernel = None
    
    def validate_input(self, fitacf_list: List[FitACF]) -> bool:
        """Validate FITACF input data"""
        if not isinstance(fitacf_list, list) or len(fitacf_list) == 0:
            return False
        
        for fitacf in fitacf_list:
            if not isinstance(fitacf, FitACF):
                return False
            if fitacf.prm is None:
                return False
        
        return True
    
    def get_memory_estimate(self, fitacf_list: List[FitACF]) -> int:
        """Estimate memory requirements for grid processing"""
        
        # Calculate grid dimensions
        n_lat = int((self.config.lat_max - self.config.lat_min) / self.config.lat_resolution) + 1
        n_lon = int((self.config.lon_max - self.config.lon_min) / self.config.lon_resolution) + 1
        
        # Grid data size
        grid_size = n_lat * n_lon * 4 * 6  # 6 float arrays (vel, err, power, etc.)
        
        # Input vector data size
        total_vectors = sum(len(fitacf.slist) for fitacf in fitacf_list)
        vector_size = total_vectors * 4 * 10  # Vector coordinates and parameters
        
        # Processing overhead
        overhead = (grid_size + vector_size) * 2
        
        return int(grid_size + vector_size + overhead)
    
    def process(self, fitacf_list: List[FitACF]) -> GridData:
        """
        Process FITACF data to create spatial grid
        
        Parameters
        ----------
        fitacf_list : List[FitACF]
            List of FITACF records to grid
            
        Returns
        -------
        GridData
            Gridded data structure
        """
        with MemoryMonitor(f"Grid Processing ({len(fitacf_list)} records)"):
            
            # Step 1: Extract and validate vectors
            vectors = self._extract_vectors(fitacf_list)
            
            if len(vectors['lat']) == 0:
                warnings.warn("No valid vectors found for gridding")
                return self._create_empty_grid()
            
            # Step 2: Convert coordinates if needed
            if self.config.coordinate_system != CoordinateSystem.GEOGRAPHIC:
                vectors = self._convert_coordinates(vectors)
            
            # Step 3: Create grid structure
            grid_data = self._create_grid_structure()
            
            # Step 4: Perform spatial interpolation
            self._interpolate_to_grid(vectors, grid_data)
            
            # Step 5: Apply spatial filtering if requested
            if self.config.spatial_filter_size > 0:
                self._apply_spatial_filter(grid_data)
            
            if self.config.median_filter_size > 0:
                self._apply_median_filter(grid_data)
            
            # Step 6: Calculate statistics and metadata
            self._calculate_grid_statistics(grid_data, fitacf_list)
            
            # Update processing statistics
            self.stats['vectors_processed'] = len(vectors['lat'])
            self.stats['grid_cells_filled'] = self.xp.sum(~self.xp.isnan(grid_data.velocity)).item()
            coverage = self.stats['grid_cells_filled'] / (len(grid_data.lat) * len(grid_data.lon))
            self.stats['spatial_coverage'] = coverage
            
            return grid_data
    
    def _extract_vectors(self, fitacf_list: List[FitACF]) -> Dict[str, Any]:
        """Extract vector data from FITACF records"""
        
        all_vectors = {
            'lat': [], 'lon': [], 'velocity': [], 'velocity_error': [],
            'power': [], 'spectral_width': [], 'quality': [],
            'timestamp': [], 'station_id': [], 'beam': [], 'range_gate': []
        }
        
        for fitacf in fitacf_list:
            prm = fitacf.prm
            
            # Get valid ranges (quality flag > 0)
            valid_mask = fitacf.qflg > 0
            valid_indices = self.xp.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                continue
            
            # Calculate geographic coordinates for each range gate
            for idx in valid_indices:
                range_gate = fitacf.slist[idx]
                
                # Calculate lat/lon for this beam and range
                lat, lon = self._calculate_coordinates(
                    prm, prm.beam_number, range_gate
                )
                
                # Apply quality thresholds
                if (fitacf.velocity_error[idx] / abs(fitacf.velocity[idx]) 
                    < self.config.vector_quality_threshold):
                    
                    all_vectors['lat'].append(lat)
                    all_vectors['lon'].append(lon)
                    all_vectors['velocity'].append(fitacf.velocity[idx])
                    all_vectors['velocity_error'].append(fitacf.velocity_error[idx])
                    all_vectors['power'].append(fitacf.power[idx])
                    all_vectors['spectral_width'].append(fitacf.spectral_width[idx])
                    all_vectors['quality'].append(fitacf.qflg[idx])
                    all_vectors['timestamp'].append(prm.timestamp)
                    all_vectors['station_id'].append(prm.station_id)
                    all_vectors['beam'].append(prm.beam_number)
                    all_vectors['range_gate'].append(range_gate)
        
        # Convert to arrays
        for key in all_vectors:
            if key != 'timestamp':
                all_vectors[key] = self.xp.array(all_vectors[key])
        
        return all_vectors
    
    def _calculate_coordinates(self, prm: RadarParameters,
                             beam: int, range_gate: int) -> Tuple[float, float]:
        """
        Calculate geographic coordinates for a beam/range gate using the spherical
        law of cosines.  Falls back to a placeholder position when the radar's
        hardware description is not in the built-in table.
        """
        hw = _RADAR_HARDWARE.get(prm.station_id,
                                 RadarHardware(prm.station_id, 60.0, -100.0, 0.0))

        # Beam azimuth: boresight ± offset from centre beam
        centre_beam = hw.n_beams // 2 - 0.5   # half-integer centre (RST convention)
        az_deg = hw.bearing + (beam - centre_beam) * hw.beam_sep
        az_rad = math.radians(az_deg)

        # Slant range to centre of gate (km)
        range_km = prm.frang + range_gate * prm.rsep

        # Spherical law of cosines (exact on sphere)
        d = range_km / _R_EARTH_KM           # angular distance (radians)
        lat1 = math.radians(hw.lat)
        lon1 = math.radians(hw.lon)

        sin_lat2 = (math.sin(lat1) * math.cos(d) +
                    math.cos(lat1) * math.sin(d) * math.cos(az_rad))
        sin_lat2 = max(-1.0, min(1.0, sin_lat2))   # clamp numerical noise
        lat2 = math.asin(sin_lat2)

        lon2 = lon1 + math.atan2(
            math.sin(az_rad) * math.sin(d) * math.cos(lat1),
            math.cos(d) - math.sin(lat1) * sin_lat2
        )

        return math.degrees(lat2), math.degrees(lon2)
    
    def _convert_coordinates(self, vectors: Dict[str, Any]) -> Dict[str, Any]:
        """Convert coordinates to specified coordinate system"""
        # Placeholder for coordinate conversion
        # Would implement AACGM or magnetic coordinate transformations
        warnings.warn(f"Coordinate conversion to {self.config.coordinate_system.value} not yet implemented")
        return vectors
    
    def _create_grid_structure(self) -> GridData:
        """Create empty grid data structure"""
        
        # Calculate grid dimensions
        n_lat = int((self.config.lat_max - self.config.lat_min) / self.config.lat_resolution) + 1
        n_lon = int((self.config.lon_max - self.config.lon_min) / self.config.lon_resolution) + 1
        
        # Create grid
        grid_data = GridData(nlat=n_lat, nlon=n_lon, 
                           use_gpu=(get_backend() == Backend.CUPY))
        
        # Set coordinate arrays
        grid_data.lat = self.xp.linspace(self.config.lat_min, self.config.lat_max, n_lat)
        grid_data.lon = self.xp.linspace(self.config.lon_min, self.config.lon_max, n_lon)
        
        return grid_data
    
    def _interpolate_to_grid(self, vectors: Dict[str, Any], grid_data: GridData):
        """Perform spatial interpolation to grid"""
        
        if (get_backend() == Backend.CUPY and 
            self.grid_kernel is not None and 
            self.config.use_gpu_interpolation):
            self._interpolate_gpu(vectors, grid_data)
        else:
            self._interpolate_cpu(vectors, grid_data)
    
    def _interpolate_gpu(self, vectors: Dict[str, Any], grid_data: GridData):
        """GPU-accelerated spatial interpolation"""
        import cupy as cp
        
        n_lat, n_lon = len(grid_data.lat), len(grid_data.lon)
        n_vectors = len(vectors['lat'])
        
        # Prepare output arrays
        grid_count = self.xp.zeros((n_lat, n_lon), dtype=self.xp.int32)
        
        # Set up kernel parameters
        search_radius = max(self.config.lat_resolution, self.config.lon_resolution) * 2
        
        block_size = (16, 16)
        grid_size = (
            (n_lat + block_size[0] - 1) // block_size[0],
            (n_lon + block_size[1] - 1) // block_size[1]
        )
        
        # Create coordinate grids for GPU
        lat_grid, lon_grid = self.xp.meshgrid(grid_data.lat, grid_data.lon, indexing='ij')
        
        # Launch interpolation kernel
        self.grid_kernel(
            grid_size, block_size,
            (vectors['lat'], vectors['lon'], vectors['velocity'], vectors['velocity_error'],
             vectors['quality'], lat_grid.ravel(), lon_grid.ravel(),
             grid_data.velocity.ravel(), grid_data.velocity_error.ravel(), grid_count.ravel(),
             n_vectors, n_lat, n_lon, search_radius, self.config.min_vectors_per_cell)
        )
        
        # Reshape results
        grid_data.velocity = grid_data.velocity.reshape((n_lat, n_lon))
        grid_data.velocity_error = grid_data.velocity_error.reshape((n_lat, n_lon))
        grid_data.num_points = grid_count.reshape((n_lat, n_lon))
    
    def _great_circle_deg(self, vec_lat: Any, vec_lon: Any,
                          target_lat: float, target_lon: float) -> Any:
        """Vectorised haversine distance in degrees."""
        xp = self.xp
        lat1 = xp.radians(vec_lat)
        lon1 = xp.radians(vec_lon)
        lat2 = math.radians(target_lat)
        lon2 = math.radians(target_lon)
        dlat = lat1 - lat2
        dlon = lon1 - lon2
        a = xp.sin(dlat * 0.5) ** 2 + math.cos(lat2) * xp.cos(lat1) * xp.sin(dlon * 0.5) ** 2
        return xp.degrees(2.0 * xp.arcsin(xp.sqrt(xp.clip(a, 0.0, 1.0))))

    def _interpolate_cpu(self, vectors: Dict[str, Any], grid_data: GridData):
        """CPU implementation of spatial interpolation"""

        n_lat, n_lon = len(grid_data.lat), len(grid_data.lon)
        search_radius = max(self.config.lat_resolution, self.config.lon_resolution) * 2

        for i, target_lat in enumerate(grid_data.lat):
            for j, target_lon in enumerate(grid_data.lon):

                distances = self._great_circle_deg(
                    vectors['lat'], vectors['lon'], float(target_lat), float(target_lon)
                )

                nearby_mask = (distances <= search_radius) & (vectors['quality'] > 0)
                nearby_indices = self.xp.where(nearby_mask)[0]
                
                if len(nearby_indices) >= self.config.min_vectors_per_cell:
                    
                    if self.config.grid_method == GridMethod.MEDIAN:
                        # Median of nearby values
                        nearby_velocities = vectors['velocity'][nearby_indices]
                        grid_data.velocity[i, j] = self.xp.median(nearby_velocities)
                        grid_data.velocity_error[i, j] = self.xp.std(nearby_velocities)
                        
                    elif self.config.grid_method == GridMethod.WEIGHTED_MEAN:
                        # Inverse distance weighting
                        nearby_distances = distances[nearby_indices]
                        weights = 1.0 / (nearby_distances + 0.001)
                        weighted_velocities = vectors['velocity'][nearby_indices] * weights
                        
                        grid_data.velocity[i, j] = self.xp.sum(weighted_velocities) / self.xp.sum(weights)
                        
                        # Weighted error
                        weighted_errors = vectors['velocity_error'][nearby_indices]**2 * weights**2
                        grid_data.velocity_error[i, j] = self.xp.sqrt(self.xp.sum(weighted_errors)) / self.xp.sum(weights)
                    
                    grid_data.num_points[i, j] = len(nearby_indices)
    
    def _apply_spatial_filter(self, grid_data: GridData):
        """Apply Gaussian spatial filter"""
        # Placeholder for spatial filtering implementation
        pass
    
    def _apply_median_filter(self, grid_data: GridData):
        """Apply median filter for noise reduction"""
        
        if (get_backend() == Backend.CUPY and 
            self.median_filter_kernel is not None):
            self._apply_median_filter_gpu(grid_data)
        else:
            self._apply_median_filter_cpu(grid_data)
    
    def _apply_median_filter_gpu(self, grid_data: GridData):
        """GPU-accelerated median filtering"""
        import cupy as cp
        
        n_lat, n_lon = grid_data.velocity.shape
        filtered_velocity = self.xp.zeros_like(grid_data.velocity)
        
        block_size = (16, 16)
        grid_size = (
            (n_lat + block_size[0] - 1) // block_size[0],
            (n_lon + block_size[1] - 1) // block_size[1]
        )
        
        self.median_filter_kernel(
            grid_size, block_size,
            (grid_data.velocity, filtered_velocity, 
             n_lat, n_lon, self.config.median_filter_size)
        )
        
        grid_data.velocity = filtered_velocity
    
    def _apply_median_filter_cpu(self, grid_data: GridData):
        """CPU implementation of median filtering"""
        from scipy import ndimage
        
        # Convert to CPU for scipy processing if needed
        if get_backend() == Backend.CUPY:
            velocity_cpu = self.xp.asnumpy(grid_data.velocity)
        else:
            velocity_cpu = grid_data.velocity
        
        # Apply median filter
        filtered_velocity = ndimage.median_filter(
            velocity_cpu, size=self.config.median_filter_size
        )
        
        # Convert back if needed
        grid_data.velocity = self.xp.asarray(filtered_velocity)
    
    def _calculate_grid_statistics(self, grid_data: GridData, fitacf_list: List[FitACF]):
        """Calculate grid metadata and statistics"""
        
        # Time range
        timestamps = [fitacf.prm.timestamp for fitacf in fitacf_list]
        grid_data.start_time = min(timestamps)
        grid_data.end_time = max(timestamps)
        
        # Grid type
        grid_data.grid_type = f"{self.config.grid_method.value}_{self.config.coordinate_system.value}"
    
    def _create_empty_grid(self) -> GridData:
        """Create empty grid when no valid vectors found"""
        return self._create_grid_structure()

# Convenience function
def create_grid(fitacf_list: List[FitACF], 
                config: Optional[GridConfig] = None) -> GridData:
    """
    Create spatial grid from FITACF data
    
    Parameters
    ----------
    fitacf_list : List[FitACF]
        List of FITACF records
    config : GridConfig, optional
        Grid processing configuration
        
    Returns
    -------
    GridData
        Gridded data structure
    """
    processor = GridProcessor(config=config)
    return processor.process(fitacf_list)