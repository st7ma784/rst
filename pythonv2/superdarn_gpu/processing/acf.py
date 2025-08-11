"""
GPU-accelerated Auto-Correlation Function (ACF) calculation

This module implements the core ACF calculation algorithms from SuperDARN,
converting raw I/Q samples to correlation functions for further processing.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.backends import get_array_module, get_backend, Backend, synchronize
from ..core.datatypes import RawACF, RadarParameters
from ..core.pipeline import Stage
from ..core.memory import MemoryMonitor

@dataclass
class ACFConfig:
    """Configuration for ACF processing"""
    # Bad sample handling
    bad_sample_threshold: float = 1e6  # Threshold for bad samples
    dc_offset_removal: bool = True     # Remove DC offset
    
    # Processing options
    xcf_processing: bool = True        # Calculate cross-correlation function
    nave_integration: int = 1          # Number of averages to integrate
    
    # Quality control
    range_gate_filtering: bool = True  # Filter bad range gates
    noise_level_estimation: bool = True # Estimate noise levels
    
    # GPU optimization
    use_gpu_kernels: bool = True       # Use custom CUDA kernels
    batch_size: int = 512              # Batch size for parallel processing

class ACFProcessor(Stage):
    """
    GPU-accelerated ACF processor for SuperDARN raw I/Q data
    """
    
    def __init__(self, config: Optional[ACFConfig] = None, **kwargs):
        super().__init__(name="ACF Processor", **kwargs)
        
        self.config = config or ACFConfig()
        self.xp = get_array_module()
        
        # Initialize GPU kernels if available
        if get_backend() == Backend.CUPY and self.config.use_gpu_kernels:
            self._init_gpu_kernels()
        
        # Processing statistics
        self.stats = {
            'samples_processed': 0,
            'bad_samples_detected': 0,
            'ranges_processed': 0,
            'processing_time': 0.0
        }
    
    def _init_gpu_kernels(self):
        """Initialize custom CUDA kernels for ACF calculation"""
        try:
            import cupy as cp
            
            # ACF calculation kernel - processes multiple lags simultaneously
            self.acf_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void calculate_acf_kernel(const float2* samples, float2* acf_out,
                                    const int* lag_table, int nrang, int nsamp,
                                    int mplgs, int nave, int rngoff) {
                
                int range_idx = blockDim.x * blockIdx.x + threadIdx.x;
                int lag_idx = blockDim.y * blockIdx.y + threadIdx.y;
                
                if (range_idx >= nrang || lag_idx >= mplgs) return;
                
                int output_idx = range_idx * mplgs + lag_idx;
                int lag_offset = lag_table[lag_idx];
                
                float2 sum = {0.0f, 0.0f};
                int valid_samples = 0;
                
                // Calculate ACF for this range and lag
                for (int avg = 0; avg < nave; avg++) {
                    int base_idx = avg * nsamp + range_idx * rngoff;
                    
                    if (base_idx + lag_offset < nsamp) {
                        float2 sample1 = samples[base_idx];
                        float2 sample2 = samples[base_idx + lag_offset];
                        
                        // Complex correlation: sample1 * conjugate(sample2)
                        float real_part = sample1.x * sample2.x + sample1.y * sample2.y;
                        float imag_part = sample1.y * sample2.x - sample1.x * sample2.y;
                        
                        sum.x += real_part;
                        sum.y += imag_part;
                        valid_samples++;
                    }
                }
                
                // Average and store result
                if (valid_samples > 0) {
                    acf_out[output_idx].x = sum.x / valid_samples;
                    acf_out[output_idx].y = sum.y / valid_samples;
                } else {
                    acf_out[output_idx].x = 0.0f;
                    acf_out[output_idx].y = 0.0f;
                }
            }
            ''', 'calculate_acf_kernel')
            
            # XCF calculation kernel for interferometer data
            self.xcf_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void calculate_xcf_kernel(const float2* main_samples, const float2* int_samples,
                                    float2* xcf_out, const int* lag_table,
                                    int nrang, int nsamp, int mplgs, int nave, int rngoff) {
                
                int range_idx = blockDim.x * blockIdx.x + threadIdx.x;
                int lag_idx = blockDim.y * blockIdx.y + threadIdx.y;
                
                if (range_idx >= nrang || lag_idx >= mplgs) return;
                
                int output_idx = range_idx * mplgs + lag_idx;
                int lag_offset = lag_table[lag_idx];
                
                float2 sum = {0.0f, 0.0f};
                int valid_samples = 0;
                
                // Calculate XCF between main and interferometer
                for (int avg = 0; avg < nave; avg++) {
                    int base_idx = avg * nsamp + range_idx * rngoff;
                    
                    if (base_idx + lag_offset < nsamp) {
                        float2 main_sample = main_samples[base_idx];
                        float2 int_sample = int_samples[base_idx + lag_offset];
                        
                        // Cross-correlation: main * conjugate(interferometer)
                        float real_part = main_sample.x * int_sample.x + main_sample.y * int_sample.y;
                        float imag_part = main_sample.y * int_sample.x - main_sample.x * int_sample.y;
                        
                        sum.x += real_part;
                        sum.y += imag_part;
                        valid_samples++;
                    }
                }
                
                if (valid_samples > 0) {
                    xcf_out[output_idx].x = sum.x / valid_samples;
                    xcf_out[output_idx].y = sum.y / valid_samples;
                } else {
                    xcf_out[output_idx].x = 0.0f;
                    xcf_out[output_idx].y = 0.0f;
                }
            }
            ''', 'calculate_xcf_kernel')
            
            print("ACF GPU kernels initialized successfully")
            
        except Exception as e:
            warnings.warn(f"Could not initialize ACF GPU kernels: {e}")
            self.acf_kernel = None
            self.xcf_kernel = None
    
    def validate_input(self, iq_data: Dict[str, Any]) -> bool:
        """Validate I/Q input data"""
        required_keys = ['samples', 'prm', 'lag_table']
        
        if not all(key in iq_data for key in required_keys):
            return False
        
        if iq_data['samples'] is None or len(iq_data['samples']) == 0:
            return False
        
        return True
    
    def get_memory_estimate(self, iq_data: Dict[str, Any]) -> int:
        """Estimate memory requirements for ACF processing"""
        prm = iq_data['prm']
        nsamp = len(iq_data['samples'])
        
        # Input samples
        input_size = nsamp * 8  # Complex64 samples
        
        # Output ACF/XCF data  
        output_size = prm.nrang * prm.mplgs * 8  # Complex ACF
        if self.config.xcf_processing:
            output_size *= 2  # Double for XCF
        
        # Intermediate processing arrays
        intermediate_size = input_size * 2  # Working arrays
        
        return int((input_size + output_size + intermediate_size) * 1.5)
    
    def process(self, iq_data: Dict[str, Any]) -> RawACF:
        """
        Process I/Q samples to calculate ACF and XCF
        
        Parameters
        ----------
        iq_data : Dict
            Dictionary containing:
            - 'samples': Complex I/Q samples array
            - 'prm': RadarParameters object
            - 'lag_table': Lag table for correlation calculation
            - 'int_samples': Interferometer samples (optional)
            
        Returns
        -------
        RawACF
            Calculated auto-correlation functions
        """
        with MemoryMonitor(f"ACF Processing"):
            
            # Extract input data
            samples = self.xp.asarray(iq_data['samples'], dtype=self.xp.complex64)
            prm = iq_data['prm']
            lag_table = self.xp.asarray(iq_data['lag_table'], dtype=self.xp.int32)
            int_samples = iq_data.get('int_samples')
            
            # Create output structure
            rawacf = RawACF(
                nrang=prm.nrang,
                mplgs=prm.mplgs,
                nave=prm.nave,
                use_gpu=(get_backend() == Backend.CUPY)
            )
            rawacf.prm = prm
            
            # Step 1: Quality control and bad sample detection
            if self.config.bad_sample_threshold > 0:
                samples = self._remove_bad_samples(samples)
            
            # Step 2: DC offset removal
            if self.config.dc_offset_removal:
                samples = self._remove_dc_offset(samples)
            
            # Step 3: Calculate ACF using GPU or CPU
            rawacf.acf = self._calculate_acf(samples, lag_table, prm)
            
            # Step 4: Calculate XCF if interferometer data available
            if self.config.xcf_processing and int_samples is not None:
                int_samples = self.xp.asarray(int_samples, dtype=self.xp.complex64)
                if self.config.dc_offset_removal:
                    int_samples = self._remove_dc_offset(int_samples)
                rawacf.xcf = self._calculate_xcf(samples, int_samples, lag_table, prm)
            
            # Step 5: Calculate power and noise levels
            self._calculate_power_and_noise(rawacf)
            
            # Step 6: Set up range lists and flags
            rawacf.slist = self.xp.arange(prm.nrang, dtype=self.xp.int16)
            rawacf.qflg = self.xp.ones(prm.nrang, dtype=self.xp.int8)
            rawacf.gflg = self.xp.zeros(prm.nrang, dtype=self.xp.int8)
            
            # Update statistics
            self.stats['samples_processed'] += len(samples)
            self.stats['ranges_processed'] += prm.nrang
            
            return rawacf
    
    def _remove_bad_samples(self, samples: Any) -> Any:
        """Remove samples exceeding threshold (likely saturated)"""
        magnitude = self.xp.abs(samples)
        bad_mask = magnitude > self.config.bad_sample_threshold
        
        # Replace bad samples with noise
        if self.xp.any(bad_mask):
            noise_level = self.xp.std(samples[~bad_mask])
            noise_samples = (self.xp.random.randn(*samples.shape).astype(self.xp.complex64) * 
                           noise_level * 0.5)
            samples = self.xp.where(bad_mask, noise_samples, samples)
            
            self.stats['bad_samples_detected'] += self.xp.sum(bad_mask).item()
        
        return samples
    
    def _remove_dc_offset(self, samples: Any) -> Any:
        """Remove DC offset from I/Q samples"""
        dc_offset = self.xp.mean(samples)
        return samples - dc_offset
    
    def _calculate_acf(self, samples: Any, lag_table: Any, prm: RadarParameters) -> Any:
        """Calculate auto-correlation function"""
        
        if (get_backend() == Backend.CUPY and 
            self.acf_kernel is not None and 
            self.config.use_gpu_kernels):
            return self._calculate_acf_gpu(samples, lag_table, prm)
        else:
            return self._calculate_acf_cpu(samples, lag_table, prm)
    
    def _calculate_acf_gpu(self, samples: Any, lag_table: Any, prm: RadarParameters) -> Any:
        """GPU-accelerated ACF calculation"""
        import cupy as cp
        
        # Prepare output array
        acf_out = self.xp.zeros((prm.nrang, prm.mplgs), dtype=self.xp.complex64)
        
        # Set up kernel launch parameters
        block_size = (16, 16)  # 16x16 threads per block
        grid_size = (
            (prm.nrang + block_size[0] - 1) // block_size[0],
            (prm.mplgs + block_size[1] - 1) // block_size[1]
        )
        
        # Launch kernel
        self.acf_kernel(
            grid_size, block_size,
            (samples, acf_out, lag_table, 
             prm.nrang, len(samples), prm.mplgs, prm.nave, 2)  # rngoff=2 for I/Q
        )
        
        return acf_out
    
    def _calculate_acf_cpu(self, samples: Any, lag_table: Any, prm: RadarParameters) -> Any:
        """CPU implementation of ACF calculation"""
        
        acf_out = self.xp.zeros((prm.nrang, prm.mplgs), dtype=self.xp.complex64)
        
        # Reshape samples for easier processing
        nsamp_per_avg = len(samples) // prm.nave
        
        for rang in range(prm.nrang):
            for lag_idx in range(prm.mplgs):
                lag_offset = lag_table[lag_idx]
                
                acf_sum = 0.0j
                valid_samples = 0
                
                for avg in range(prm.nave):
                    base_idx = avg * nsamp_per_avg + rang * 2  # 2 samples per range (I/Q)
                    
                    if base_idx + lag_offset < len(samples):
                        sample1 = samples[base_idx]
                        sample2 = samples[base_idx + lag_offset]
                        
                        # ACF: sample1 * conjugate(sample2)
                        acf_sum += sample1 * self.xp.conj(sample2)
                        valid_samples += 1
                
                if valid_samples > 0:
                    acf_out[rang, lag_idx] = acf_sum / valid_samples
        
        return acf_out
    
    def _calculate_xcf(self, main_samples: Any, int_samples: Any, 
                      lag_table: Any, prm: RadarParameters) -> Any:
        """Calculate cross-correlation function between main and interferometer"""
        
        if (get_backend() == Backend.CUPY and 
            self.xcf_kernel is not None and 
            self.config.use_gpu_kernels):
            return self._calculate_xcf_gpu(main_samples, int_samples, lag_table, prm)
        else:
            return self._calculate_xcf_cpu(main_samples, int_samples, lag_table, prm)
    
    def _calculate_xcf_gpu(self, main_samples: Any, int_samples: Any, 
                          lag_table: Any, prm: RadarParameters) -> Any:
        """GPU-accelerated XCF calculation"""
        import cupy as cp
        
        xcf_out = self.xp.zeros((prm.nrang, prm.mplgs), dtype=self.xp.complex64)
        
        block_size = (16, 16)
        grid_size = (
            (prm.nrang + block_size[0] - 1) // block_size[0],
            (prm.mplgs + block_size[1] - 1) // block_size[1]
        )
        
        self.xcf_kernel(
            grid_size, block_size,
            (main_samples, int_samples, xcf_out, lag_table,
             prm.nrang, len(main_samples), prm.mplgs, prm.nave, 2)
        )
        
        return xcf_out
    
    def _calculate_xcf_cpu(self, main_samples: Any, int_samples: Any, 
                          lag_table: Any, prm: RadarParameters) -> Any:
        """CPU implementation of XCF calculation"""
        
        xcf_out = self.xp.zeros((prm.nrang, prm.mplgs), dtype=self.xp.complex64)
        nsamp_per_avg = len(main_samples) // prm.nave
        
        for rang in range(prm.nrang):
            for lag_idx in range(prm.mplgs):
                lag_offset = lag_table[lag_idx]
                
                xcf_sum = 0.0j
                valid_samples = 0
                
                for avg in range(prm.nave):
                    base_idx = avg * nsamp_per_avg + rang * 2
                    
                    if base_idx + lag_offset < len(main_samples):
                        main_sample = main_samples[base_idx]
                        int_sample = int_samples[base_idx + lag_offset]
                        
                        # XCF: main * conjugate(interferometer)
                        xcf_sum += main_sample * self.xp.conj(int_sample)
                        valid_samples += 1
                
                if valid_samples > 0:
                    xcf_out[rang, lag_idx] = xcf_sum / valid_samples
        
        return xcf_out
    
    def _calculate_power_and_noise(self, rawacf: RawACF):
        """Calculate power levels and noise estimates"""
        
        # Power from lag-0 ACF (real part)
        rawacf.power = self.xp.real(rawacf.acf[:, 0])
        
        # Estimate noise level from higher lags or low-power ranges
        if self.config.noise_level_estimation:
            # Simple noise estimation - use median of low-power ranges
            power_sorted = self.xp.sort(rawacf.power)
            noise_estimate = self.xp.median(power_sorted[:len(power_sorted)//4])  # Bottom 25%
            rawacf.noise[:] = noise_estimate
        else:
            rawacf.noise[:] = 0.0

# Convenience function for direct processing
def calculate_acf(iq_data: Dict[str, Any], 
                  config: Optional[ACFConfig] = None) -> RawACF:
    """
    Calculate ACF from I/Q samples using GPU acceleration
    
    Parameters
    ----------
    iq_data : Dict
        Dictionary with I/Q samples and parameters
    config : ACFConfig, optional
        Processing configuration
        
    Returns
    -------
    RawACF
        Calculated auto-correlation functions
    """
    processor = ACFProcessor(config=config)
    return processor.process(iq_data)

def create_lag_table(pulse_table: List[int], mpinc: int, mplgs: int) -> np.ndarray:
    """
    Create lag table for ACF calculation
    
    Parameters
    ----------
    pulse_table : List[int]
        Pulse sequence timing
    mpinc : int
        Multi-pulse increment (μs)
    mplgs : int
        Number of lags
        
    Returns
    -------
    np.ndarray
        Lag table for correlation calculation
    """
    lag_table = []
    
    for lag in range(mplgs):
        if lag < len(pulse_table):
            # Convert time delay to sample offset
            time_delay = pulse_table[lag] * mpinc  # μs
            sample_offset = int(time_delay * 2)  # Assuming 2 samples per μs
            lag_table.append(sample_offset)
        else:
            lag_table.append(0)
    
    return np.array(lag_table, dtype=np.int32)