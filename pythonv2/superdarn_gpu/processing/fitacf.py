"""
GPU-accelerated FITACF processing with CUPy

This module provides high-performance FITACF processing using GPU acceleration,
porting the sophisticated algorithms from the C CUDA implementation.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.backends import get_array_module, get_backend, Backend, synchronize
from ..core.datatypes import RawACF, FitACF
from ..core.pipeline import Stage
from ..core.memory import MemoryMonitor, memory_pool
from ..algorithms.fitting import LeastSquaresFitter, PhaseUnwrapper

class FitACFAlgorithm(Enum):
    """Available FITACF algorithms"""
    V2_5 = "v2.5"  # Legacy algorithm
    V3_0 = "v3.0"  # Modern algorithm with improvements

@dataclass 
class FitACFConfig:
    """Configuration for FITACF processing"""
    algorithm: FitACFAlgorithm = FitACFAlgorithm.V3_0
    
    # Fitting parameters
    min_power_threshold: float = 3.0  # Minimum power threshold (dB)
    max_velocity: float = 2000.0      # Maximum velocity (m/s)
    max_spectral_width: float = 1000.0 # Maximum spectral width (m/s)
    
    # Quality control
    min_lags_for_fit: int = 3         # Minimum lags required for fitting
    max_phase_unwrap_error: float = 0.1 # Phase unwrapping tolerance
    ground_scatter_threshold: float = 0.3 # Ground scatter detection
    
    # GPU optimization
    batch_size: int = 1024            # Process multiple ranges simultaneously
    use_shared_memory: bool = True    # Use GPU shared memory
    async_processing: bool = True     # Asynchronous processing
    
    # Advanced options
    enable_xcf: bool = True           # Process cross-correlation function
    elevation_correction: bool = True # Apply elevation angle correction
    tdiff_correction: bool = True     # Multi-channel time difference correction

class FitACFProcessor(Stage):
    """
    GPU-accelerated FITACF processor using CUPy
    """
    
    def __init__(self, config: Optional[FitACFConfig] = None, **kwargs):
        super().__init__(name="FITACF Processor", **kwargs)
        
        self.config = config or FitACFConfig()
        self.xp = get_array_module()
        
        # Initialize GPU kernels if available
        if get_backend() == Backend.CUPY:
            self._init_gpu_kernels()
        
        # Initialize fitting components
        self.fitter = LeastSquaresFitter()
        self.phase_unwrapper = PhaseUnwrapper()
        
        # Processing statistics
        self.stats = {
            'processed_ranges': 0,
            'fitted_ranges': 0,
            'processing_time': 0.0,
            'gpu_utilization': 0.0
        }
    
    def _init_gpu_kernels(self):
        """Initialize custom CUDA kernels for FITACF processing"""
        try:
            import cupy as cp
            
            # Power calculation kernel
            self.power_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void calculate_power_lag0(const float2* acf, float* power, 
                                     int nrang, int mplgs) {
                int range_idx = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (range_idx < nrang) {
                    // Power is real part of lag-0 ACF
                    power[range_idx] = acf[range_idx * mplgs].x;
                }
            }
            ''', 'calculate_power_lag0')
            
            # Phase calculation kernel
            self.phase_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void calculate_phase(const float2* acf, float* phase, 
                                int nrang, int mplgs) {
                int range_idx = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (range_idx < nrang && mplgs > 1) {
                    // Phase from lag-1 ACF
                    float2 lag1 = acf[range_idx * mplgs + 1];
                    phase[range_idx] = atan2f(lag1.y, lag1.x);
                }
            }
            ''', 'calculate_phase')
            
            print("FITACF GPU kernels initialized successfully")
            
        except Exception as e:
            warnings.warn(f"Could not initialize GPU kernels: {e}")
            self.power_kernel = None
            self.phase_kernel = None
    
    def validate_input(self, rawacf: RawACF) -> bool:
        """Validate RawACF input data"""
        if not isinstance(rawacf, RawACF):
            return False
        
        if rawacf.nrang <= 0 or rawacf.mplgs <= 0:
            return False
        
        if rawacf.acf is None or rawacf.power is None:
            return False
        
        return True
    
    def get_memory_estimate(self, rawacf: RawACF) -> int:
        """Estimate memory requirements for FITACF processing"""
        nrang = rawacf.nrang
        mplgs = rawacf.mplgs
        
        # Input data size
        input_size = nrang * mplgs * 8  # Complex64 ACF data
        input_size += nrang * 4 * 6     # Power, noise, flags, etc.
        
        # Output data size  
        output_size = nrang * 4 * 12    # All fitted parameters
        
        # Intermediate arrays (fitting matrices, etc.)
        intermediate_size = nrang * mplgs * 16  # Working arrays
        
        # Total with safety margin
        return int((input_size + output_size + intermediate_size) * 1.5)
    
    def process(self, rawacf: RawACF) -> FitACF:
        """
        Process RawACF data to produce FITACF parameters
        
        Parameters
        ----------
        rawacf : RawACF
            Input raw ACF data
            
        Returns
        -------
        FitACF
            Fitted parameters
        """
        with MemoryMonitor(f"FITACF Processing (nrang={rawacf.nrang})"):
            
            # Create output structure
            fitacf = FitACF(nrang=rawacf.nrang, use_gpu=(get_backend() == Backend.CUPY))
            fitacf.prm = rawacf.prm
            
            # Copy range list and initial flags
            fitacf.slist = self.xp.copy(rawacf.slist)
            fitacf.qflg = self.xp.zeros_like(rawacf.qflg)
            fitacf.gflg = self.xp.zeros_like(rawacf.gflg)
            
            # Step 1: Calculate basic power from lag-0
            self._calculate_power(rawacf, fitacf)
            
            # Step 2: Apply power threshold and quality control
            valid_ranges = self._apply_power_threshold(rawacf, fitacf)
            
            # Step 3: Calculate phases from ACF
            self._calculate_phases(rawacf, fitacf, valid_ranges)
            
            # Step 4: Perform least-squares fitting for velocity and width
            self._fit_acf_parameters(rawacf, fitacf, valid_ranges)
            
            # Step 5: Ground scatter detection
            if self.config.ground_scatter_threshold > 0:
                self._detect_ground_scatter(rawacf, fitacf)
            
            # Step 6: Elevation angle calculation (if enabled)
            if self.config.elevation_correction and self.config.enable_xcf:
                self._calculate_elevation(rawacf, fitacf, valid_ranges)
            
            # Update statistics
            self.stats['processed_ranges'] += rawacf.nrang
            self.stats['fitted_ranges'] += self.xp.sum(fitacf.qflg > 0).item()
            
            return fitacf
    
    def _calculate_power(self, rawacf: RawACF, fitacf: FitACF):
        """Calculate power from lag-0 ACF"""
        if get_backend() == Backend.CUPY and self.power_kernel is not None:
            # Use custom GPU kernel
            import cupy as cp
            
            block_size = min(256, rawacf.nrang)
            grid_size = (rawacf.nrang + block_size - 1) // block_size
            
            self.power_kernel(
                (grid_size,), (block_size,),
                (rawacf.acf, fitacf.power, rawacf.nrang, rawacf.mplgs)
            )
        else:
            # CPU/fallback implementation  
            fitacf.power = self.xp.real(rawacf.acf[:, 0])
    
    def _apply_power_threshold(self, rawacf: RawACF, fitacf: FitACF) -> Any:
        """Apply power threshold and mark valid ranges"""
        
        # Convert power to dB
        power_db = 10 * self.xp.log10(fitacf.power + 1e-10)  # Avoid log(0)
        
        # Find ranges above threshold
        valid_mask = power_db >= self.config.min_power_threshold
        
        # Set quality flags
        fitacf.qflg[valid_mask] = 1
        
        return valid_mask
    
    def _calculate_phases(self, rawacf: RawACF, fitacf: FitACF, valid_ranges: Any):
        """Calculate phase information from ACF"""
        if rawacf.mplgs < 2:
            return
        
        if get_backend() == Backend.CUPY and self.phase_kernel is not None:
            # Use GPU kernel
            import cupy as cp
            
            block_size = min(256, rawacf.nrang)
            grid_size = (rawacf.nrang + block_size - 1) // block_size
            
            self.phase_kernel(
                (grid_size,), (block_size,),
                (rawacf.acf, fitacf.phase, rawacf.nrang, rawacf.mplgs)
            )
        else:
            # CPU implementation
            if rawacf.mplgs > 1:
                fitacf.phase = self.xp.angle(rawacf.acf[:, 1])
        
        # Apply phase unwrapping for valid ranges
        if self.xp.any(valid_ranges):
            valid_indices = self.xp.where(valid_ranges)[0]
            if len(valid_indices) > 0:
                fitacf.phase[valid_indices] = self.phase_unwrapper.unwrap(
                    fitacf.phase[valid_indices]
                )
    
    def _fit_acf_parameters(self, rawacf: RawACF, fitacf: FitACF, valid_ranges: Any):
        """
        Perform least-squares fitting to extract velocity and spectral width
        """
        if not self.xp.any(valid_ranges):
            return
        
        valid_indices = self.xp.where(valid_ranges)[0]
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            return
        
        # Process in batches for memory efficiency
        batch_size = min(self.config.batch_size, n_valid)
        
        for i in range(0, n_valid, batch_size):
            batch_end = min(i + batch_size, n_valid)
            batch_indices = valid_indices[i:batch_end]
            
            # Extract ACF data for batch
            batch_acf = rawacf.acf[batch_indices, :]
            batch_power = rawacf.power[batch_indices]
            
            # Perform batch fitting
            results = self.fitter.fit_lorentzian_batch(
                batch_acf, batch_power, 
                max_velocity=self.config.max_velocity,
                max_width=self.config.max_spectral_width
            )
            
            # Store results
            fitacf.velocity[batch_indices] = results['velocity']
            fitacf.velocity_error[batch_indices] = results['velocity_error']
            fitacf.spectral_width[batch_indices] = results['spectral_width']
            fitacf.spectral_width_error[batch_indices] = results['spectral_width_error']
            fitacf.power[batch_indices] = results['power']
            fitacf.power_error[batch_indices] = results['power_error']
    
    def _detect_ground_scatter(self, rawacf: RawACF, fitacf: FitACF):
        """Detect ground scatter using spectral characteristics"""
        
        # Ground scatter typically has low spectral width and specific power characteristics
        valid_mask = fitacf.qflg > 0
        
        if not self.xp.any(valid_mask):
            return
        
        # Ground scatter criteria
        low_width = fitacf.spectral_width < (self.config.ground_scatter_threshold * 
                                           self.config.max_spectral_width)
        high_power = fitacf.power > self.xp.median(fitacf.power[valid_mask])
        
        # Mark ground scatter
        ground_mask = valid_mask & low_width & high_power
        fitacf.gflg[ground_mask] = 1
    
    def _calculate_elevation(self, rawacf: RawACF, fitacf: FitACF, valid_ranges: Any):
        """Calculate elevation angle from cross-correlation function"""
        if rawacf.xcf is None or not self.config.enable_xcf:
            return
        
        valid_indices = self.xp.where(valid_ranges)[0]
        
        if len(valid_indices) == 0:
            return
        
        # Calculate elevation from XCF phase
        # This is a simplified implementation - full algorithm is more complex
        xcf_phase = self.xp.angle(rawacf.xcf[valid_indices, 0])
        
        # Convert phase to elevation (radar-specific conversion)
        # This would need actual radar hardware parameters
        fitacf.elevation[valid_indices] = xcf_phase * (180.0 / self.xp.pi)

# Convenience function for direct processing
def process_fitacf(rawacf: Union[RawACF, List[RawACF]], 
                   config: Optional[FitACFConfig] = None) -> Union[FitACF, List[FitACF]]:
    """
    Process RawACF data to FITACF using GPU acceleration
    
    Parameters
    ----------
    rawacf : RawACF or List[RawACF]
        Input raw ACF data
    config : FitACFConfig, optional
        Processing configuration
        
    Returns
    -------
    FitACF or List[FitACF]
        Fitted parameters
    """
    processor = FitACFProcessor(config=config)
    
    if isinstance(rawacf, list):
        return [processor.process(record) for record in rawacf]
    else:
        return processor.process(rawacf)

# GPU kernel implementations for advanced processing
def _create_advanced_kernels():
    """Create advanced CUDA kernels for FITACF processing"""
    
    if get_backend() != Backend.CUPY:
        return {}
    
    try:
        import cupy as cp
        
        # Lorentzian fitting kernel
        lorentzian_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void fit_lorentzian(const float2* acf, const float* power,
                           float* velocity, float* width, float* power_fitted,
                           int nrang, int mplgs, float max_vel, float max_width) {
            
            int range_idx = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (range_idx >= nrang) return;
            
            // Simplified Lorentzian fitting using least squares
            // Full implementation would include complete fitting algorithm
            
            float sum_power = 0.0f;
            float weighted_phase = 0.0f;
            float phase_variance = 0.0f;
            int valid_lags = 0;
            
            for (int lag = 0; lag < mplgs; lag++) {
                float2 acf_val = acf[range_idx * mplgs + lag];
                float lag_power = acf_val.x * acf_val.x + acf_val.y * acf_val.y;
                
                if (lag_power > 0.001f) {  // Valid lag
                    sum_power += lag_power;
                    
                    if (lag > 0) {
                        float phase = atan2f(acf_val.y, acf_val.x);
                        weighted_phase += phase * lag_power;
                        phase_variance += phase * phase * lag_power;
                    }
                    valid_lags++;
                }
            }
            
            if (valid_lags < 3) {
                // Not enough valid lags
                velocity[range_idx] = 0.0f;
                width[range_idx] = 0.0f;
                power_fitted[range_idx] = 0.0f;
                return;
            }
            
            // Calculate fitted parameters
            weighted_phase /= sum_power;
            phase_variance = phase_variance / sum_power - weighted_phase * weighted_phase;
            
            velocity[range_idx] = fminf(fmaxf(weighted_phase * 100.0f, -max_vel), max_vel);
            width[range_idx] = fminf(sqrtf(fmaxf(phase_variance, 0.0f)) * 50.0f, max_width);
            power_fitted[range_idx] = sum_power;
        }
        ''', 'fit_lorentzian')
        
        return {'lorentzian_kernel': lorentzian_kernel}
        
    except Exception as e:
        warnings.warn(f"Could not create advanced CUDA kernels: {e}")
        return {}

# Module initialization - create kernels if GPU available
_advanced_kernels = _create_advanced_kernels()