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
from ..algorithms.lag_validation import LagValidator

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
        self.fitter        = LeastSquaresFitter()
        self.phase_unwrapper = PhaseUnwrapper()
        self.lag_validator = LagValidator()
        
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
        
        # Phase unwrapping must operate across lags within each range, not across ranges.
        # fitacf.phase holds only the lag-1 angle; full multi-lag unwrapping happens
        # inside _fit_single_range via xp.unwrap on the valid_phases sequence.
    
    def _fit_acf_parameters(self, rawacf: RawACF, fitacf: FitACF, valid_ranges: Any):
        """
        Two-pass weighted LS with LagValidator (vectorized over full scan).

        1. Estimate noise floor from lowest-power ranges (RST ACF_cutoff_pwr)
        2. Compute alpha_2 for all (nrang, mplgs) — vectorized
        3. Cumsum-based validity mask — propagates cutoff per range in one op
        4. Bendat-Piersol sigma — vectorized
        5. Batch fit (valid_mask + bp_sigma → two-pass + quadratic width)
        """
        xp = self.xp
        if not xp.any(valid_ranges):
            return

        prm       = rawacf.prm
        mpinc_us  = getattr(prm, 'mpinc',  1500) if prm is not None else 1500
        tfreq_khz = getattr(prm, 'tfreq', 12000) if prm is not None else 12000
        nave      = getattr(prm, 'nave',     20) if prm is not None else 20
        lag_dt    = mpinc_us * 1e-6
        tfreq_hz  = tfreq_khz * 1000.0

        # ── noise floor (RST ACF_cutoff_pwr approach) ──────────────────────
        pwr0_all  = xp.abs(rawacf.acf[:, 0])
        noise_pwr = self.lag_validator.noise_floor_from_acf(rawacf.acf, pwr0_all)

        # ── validity mask + B-P sigma over FULL (nrang, mplgs) ─────────────
        alpha_2    = self.lag_validator.compute_alpha_2(rawacf.acf, pwr0_all, nave)
        valid_mask = self.lag_validator.compute_lag_validity_mask(
            rawacf.acf, pwr0_all, alpha_2, nave, noise_pwr
        )
        bp_sigma   = self.lag_validator.compute_bendat_piersol_sigma(
            rawacf.acf, pwr0_all, alpha_2, nave
        )

        valid_indices = xp.where(valid_ranges)[0]
        if len(valid_indices) == 0:
            return

        batch_size = min(self.config.batch_size, len(valid_indices))
        for i in range(0, len(valid_indices), batch_size):
            idx   = valid_indices[i:i + batch_size]
            b_acf = rawacf.acf[idx, :]
            b_pwr = pwr0_all[idx]
            b_vm  = valid_mask[idx, :]
            b_sig = bp_sigma[idx, :]

            results = self.fitter.fit_lorentzian_batch(
                b_acf, b_pwr,
                lag_time_step_sec=lag_dt,
                tfreq_hz=tfreq_hz,
                max_velocity=self.config.max_velocity,
                max_width=self.config.max_spectral_width,
                valid_mask=b_vm,
                bp_sigma=b_sig,
                noise_pwr=noise_pwr,
            )

            fitacf.velocity[idx]                   = results['velocity']
            fitacf.velocity_error[idx]             = results['velocity_error']
            fitacf.spectral_width[idx]             = results['spectral_width']
            fitacf.spectral_width_error[idx]       = results['spectral_width_error']
            fitacf.spectral_width_sigma[idx]       = results['spectral_width_sigma']
            fitacf.spectral_width_sigma_error[idx] = results['spectral_width_sigma_error']
            fitacf.power[idx]                      = results['power']
            fitacf.power_error[idx]                = results['power_error']
            fitacf.nlag_fit[idx]                   = self.lag_validator.count_valid_lags(b_vm)

            # Quality gate: demote fits where error > 80 % of signal or width ≤ 0
            bad_fit = (
                (fitacf.velocity_error[idx] > xp.abs(fitacf.velocity[idx]) * 0.8)
                | (fitacf.spectral_width[idx] <= 0)
            )
            fitacf.qflg[idx] = xp.where(bad_fit, xp.int8(0), fitacf.qflg[idx])
    
    def _detect_ground_scatter(self, rawacf: RawACF, fitacf: FitACF):
        """
        RST ground-scatter criterion from fitacf.2.5/src/ground_scatter.c.

        gflg = 1  if  |v| < GS_VMAX - (GS_VMAX / GS_WMAX) * |w_l|
        where GS_VMAX = 30 m/s, GS_WMAX = 90 m/s.

        This is the line in V/W space that separates ground scatter (low velocity,
        narrow width) from ionospheric scatter.
        """
        GS_VMAX = 30.0   # m/s
        GS_WMAX = 90.0   # m/s

        xp = self.xp
        valid = fitacf.qflg > 0
        if not xp.any(valid):
            return

        v_abs = xp.abs(fitacf.velocity)
        w_abs = xp.abs(fitacf.spectral_width)
        threshold = GS_VMAX - (GS_VMAX / GS_WMAX) * w_abs
        fitacf.gflg[valid & (v_abs < threshold)] = 1
    
    def _calculate_elevation(self, rawacf: RawACF, fitacf: FitACF, valid_ranges: Any):
        """
        Elevation angle from XCF interferometer phase — RST elevation.1.0 formula.

        sin(θ) = (φ_obs - Δχ_cable) / (k · d · cos(φ_beam))

        where:
          φ_obs      = lag-0 XCF phase (rad)
          Δχ_cable   = -2π · f · tdiff · 1e-6  (cable-delay correction, RST elevation.c)
          k          = 2π · f / c               (wavenumber)
          d          = antenna separation (m)
          cos(φ_beam)= cosine of beam azimuth off boresight
                       φ_beam = (beam - 7.5) * beam_sep_deg  (standard 16-beam radar)

        Refs: RST codebase/superdarn/src.lib/tk/elevation.1.0/src/elevation.c
        """
        if rawacf.xcf is None or not self.config.enable_xcf:
            fitacf.elevation[:] = float('nan')
            return

        valid_indices = self.xp.where(valid_ranges)[0]
        if len(valid_indices) == 0:
            return

        import math
        prm        = rawacf.prm
        tfreq_hz   = float(getattr(prm, 'tfreq',      12000)) * 1000.0   # kHz → Hz
        ant_sep    = float(getattr(prm, 'antenna_sep', 100.0))            # metres
        tdiff_us   = float(getattr(prm, 'tdiff',       0.0))              # µs
        beam_num   = float(getattr(prm, 'beam_number', 7))
        beam_sep   = float(getattr(prm, 'beam_sep',    3.24))             # degrees

        # Cable-delay correction: Δχ = -2π·f·tdiff·1e-6  (RST: elevation.c)
        dchi_cable = -2.0 * math.pi * tfreq_hz * tdiff_us * 1e-6

        # Beam-direction cosine (16-beam standard radar, centre at beam 7.5)
        phi_beam   = math.radians((beam_num - 7.5) * beam_sep)
        cos_phi    = max(math.cos(phi_beam), 0.1)   # clamp to avoid division near 90°

        k      = 2.0 * math.pi * tfreq_hz / 3e8
        denom  = k * ant_sep * cos_phi

        phi_obs = self.xp.angle(rawacf.xcf[valid_indices, 0])
        psi     = phi_obs - dchi_cable                              # cable-corrected phase

        if denom > 0.0:
            sin_elv = self.xp.clip(psi / denom, -1.0, 1.0)
            fitacf.elevation[valid_indices] = self.xp.degrees(self.xp.arcsin(sin_elv))
        else:
            fitacf.elevation[valid_indices] = float('nan')

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