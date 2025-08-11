#!/usr/bin/env python3
"""
Basic FITACF Processing Example with SuperDARN GPU

This example demonstrates how to process SuperDARN raw ACF data
to produce fitted parameters using GPU acceleration.
"""

import numpy as np
import superdarn_gpu as sd
from superdarn_gpu.core.datatypes import RawACF, RadarParameters
from superdarn_gpu.processing.fitacf import FitACFProcessor, FitACFConfig
from datetime import datetime
import time

def create_synthetic_rawacf():
    """
    Create synthetic RawACF data for demonstration
    """
    # Radar parameters
    nrang = 75    # Number of range gates  
    mplgs = 18    # Number of lags
    nave = 50     # Number of averages
    
    prm = RadarParameters(
        station_id=65,           # Saskatoon radar
        beam_number=7,
        scan_flag=1,
        channel=1,
        cp_id=153,
        nave=nave,
        lagfr=4800,
        smsep=300,
        txpow=9000,
        atten=0,
        noise_search=2.5,
        noise_mean=2.3,
        tfreq=10500,
        nrang=nrang,
        frang=180,
        rsep=45,
        xcf=1,
        mppul=8,
        mpinc=1500,
        mplgs=mplgs,
        txpl=100,
        intt_sc=3,
        intt_us=0,
        timestamp=datetime.now()
    )
    
    # Create RawACF structure
    rawacf = RawACF(nrang=nrang, mplgs=mplgs, nave=nave)
    rawacf.prm = prm
    
    # Generate synthetic ACF data
    # This simulates realistic SuperDARN ACF with Lorentzian decay
    for r in range(nrang):
        # Simulate ionospheric scatter with some ranges
        if r > 15 and r < 55 and np.random.random() > 0.3:
            
            # Realistic parameters
            velocity = np.random.uniform(-800, 800)  # m/s
            spectral_width = np.random.uniform(50, 300)  # m/s  
            power = np.random.uniform(100, 10000)  # Power
            
            # Create Lorentzian ACF
            for lag in range(mplgs):
                # Lorentzian model: ACF(τ) = P * exp(-W*τ + i*φ*τ)
                tau = lag * prm.mpinc * 1e-6  # Time delay
                
                # Decay due to spectral width
                decay = np.exp(-spectral_width * tau / 200.0)  # Scaling factor
                
                # Phase progression due to Doppler velocity
                phase = velocity * tau / 100.0  # Scaling factor
                
                # Complex ACF value
                acf_real = power * decay * np.cos(phase)
                acf_imag = power * decay * np.sin(phase)
                
                # Add noise
                noise_level = 10.0
                acf_real += np.random.normal(0, noise_level)
                acf_imag += np.random.normal(0, noise_level)
                
                rawacf.acf[r, lag] = acf_real + 1j * acf_imag
            
            # Set power (real part of lag-0)
            rawacf.power[r] = np.real(rawacf.acf[r, 0])
            
        else:
            # Noise-only ranges
            for lag in range(mplgs):
                noise = np.random.normal(0, 5.0, 2)  # Real and imaginary noise
                rawacf.acf[r, lag] = noise[0] + 1j * noise[1]
            rawacf.power[r] = abs(rawacf.acf[r, 0])
    
    # Set noise levels
    rawacf.noise[:] = 5.0
    
    # Set range list (consecutive ranges)
    rawacf.slist[:] = np.arange(nrang)
    
    return rawacf

def benchmark_processing():
    """
    Benchmark CPU vs GPU processing performance
    """
    print("Creating synthetic data...")
    rawacf = create_synthetic_rawacf()
    
    # Configure processing
    config = FitACFConfig(
        algorithm=sd.processing.fitacf.FitACFAlgorithm.V3_0,
        min_power_threshold=3.0,
        batch_size=512
    )
    
    # Test CPU processing
    print("\n=== CPU Processing ===")
    with sd.core.backends.BackendContext('numpy'):
        cpu_processor = FitACFProcessor(config=config)
        
        start_time = time.time()
        cpu_result = cpu_processor.process(rawacf)
        cpu_time = time.time() - start_time
        
        print(f"CPU processing time: {cpu_time:.3f} seconds")
        print(f"Fitted ranges: {np.sum(cpu_result.qflg > 0)}/{rawacf.nrang}")
    
    # Test GPU processing (if available)
    if sd.GPU_AVAILABLE:
        print("\n=== GPU Processing ===")
        with sd.core.backends.BackendContext('cupy'):
            gpu_processor = FitACFProcessor(config=config)
            
            start_time = time.time()
            gpu_result = gpu_processor.process(rawacf)
            gpu_time = time.time() - start_time
            
            print(f"GPU processing time: {gpu_time:.3f} seconds")
            print(f"Fitted ranges: {np.sum(sd.core.backends.to_cpu(gpu_result.qflg) > 0)}/{rawacf.nrang}")
            
            if cpu_time > gpu_time:
                speedup = cpu_time / gpu_time
                print(f"GPU speedup: {speedup:.1f}x")
        
        # Verify results are similar
        print("\n=== Verification ===")
        cpu_vel = cpu_result.velocity[cpu_result.qflg > 0]
        gpu_vel = sd.core.backends.to_cpu(gpu_result.velocity[sd.core.backends.to_cpu(gpu_result.qflg) > 0])
        
        if len(cpu_vel) > 0 and len(gpu_vel) > 0:
            correlation = np.corrcoef(cpu_vel, gpu_vel)[0, 1]
            print(f"Velocity correlation (CPU vs GPU): {correlation:.3f}")
    else:
        print("\n=== GPU Not Available ===")
        print("Install CuPy and ensure CUDA is available for GPU acceleration")

def analyze_results(fitacf):
    """
    Analyze and display FITACF results
    """
    print("\n=== FITACF Results Analysis ===")
    
    # Convert to CPU arrays for analysis
    velocity = sd.core.backends.to_cpu(fitacf.velocity)
    power = sd.core.backends.to_cpu(fitacf.power)
    spectral_width = sd.core.backends.to_cpu(fitacf.spectral_width)
    qflg = sd.core.backends.to_cpu(fitacf.qflg)
    gflg = sd.core.backends.to_cpu(fitacf.gflg)
    
    # Valid data statistics
    valid_mask = qflg > 0
    n_valid = np.sum(valid_mask)
    n_ground = np.sum(gflg > 0)
    
    print(f"Total ranges: {len(qflg)}")
    print(f"Valid ranges: {n_valid}")
    print(f"Ground scatter ranges: {n_ground}")
    print(f"Ionospheric ranges: {n_valid - n_ground}")
    
    if n_valid > 0:
        valid_vel = velocity[valid_mask]
        valid_power = power[valid_mask]
        valid_width = spectral_width[valid_mask]
        
        print(f"\nVelocity statistics:")
        print(f"  Range: [{np.min(valid_vel):.1f}, {np.max(valid_vel):.1f}] m/s")
        print(f"  Mean: {np.mean(valid_vel):.1f} m/s")
        print(f"  Std: {np.std(valid_vel):.1f} m/s")
        
        print(f"\nPower statistics:")
        print(f"  Range: [{np.min(valid_power):.1f}, {np.max(valid_power):.1f}]")
        print(f"  Mean: {np.mean(valid_power):.1f}")
        
        print(f"\nSpectral width statistics:")
        print(f"  Range: [{np.min(valid_width):.1f}, {np.max(valid_width):.1f}] m/s")
        print(f"  Mean: {np.mean(valid_width):.1f} m/s")

def main():
    """Main example function"""
    print("SuperDARN GPU FITACF Processing Example")
    print("=" * 50)
    
    print(f"Backend: {sd.get_backend()}")
    print(f"GPU Available: {sd.GPU_AVAILABLE}")
    
    # Run benchmark
    benchmark_processing()
    
    # Create and process single example
    print("\n" + "=" * 50)
    print("Processing Single Example")
    
    rawacf = create_synthetic_rawacf()
    
    # Process with default settings
    fitacf_result = sd.processing.process_fitacf(rawacf)
    
    # Analyze results
    analyze_results(fitacf_result)
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()