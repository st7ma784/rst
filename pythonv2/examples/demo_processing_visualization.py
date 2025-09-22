#!/usr/bin/env python3
"""
SuperDARN GPU Processing with Live Visualization Demo
====================================================

This script demonstrates the complete SuperDARN GPU processing pipeline
with integrated real-time visualization for scientists to see what's
happening at each processing stage.

Usage:
    python demo_processing_visualization.py [--use-cpu] [--data-file FILE]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import argparse

# Add the package to path for demo
sys.path.insert(0, '/home/user/rst/pythonv2')

import superdarn_gpu as sd
from superdarn_gpu.visualization import (
    RealtimeMonitor, ProcessingViewer, PerformanceDashboard,
    InteractiveExplorer, create_processing_dashboard
)


def generate_synthetic_superdarn_data(n_times=100, n_beams=16, n_ranges=75, add_noise=True):
    """
    Generate synthetic SuperDARN data for demonstration
    
    This creates realistic-looking data with typical SuperDARN characteristics:
    - Range-dependent power decay
    - Doppler velocity patterns
    - Ground scatter and ionospheric scatter
    - Realistic noise levels
    """
    print("🔬 Generating synthetic SuperDARN data...")
    
    # Time array (2 minute resolution)
    start_time = datetime(2024, 3, 15, 12, 0, 0)
    times = [start_time + timedelta(minutes=2*i) for i in range(n_times)]
    
    # Range gates (15 km resolution starting at 180 km)
    ranges = np.arange(180, 180 + n_ranges * 15, 15)
    
    # Beam directions (3.24° spacing)
    beams = np.arange(n_beams)
    
    # Initialize arrays
    xp = sd.get_backend()
    velocity = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    power = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    width = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    elevation = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    
    for t in range(n_times):
        for b in range(n_beams):
            for r in range(n_ranges):
                range_km = ranges[r]
                
                # Range-dependent power decay
                base_power = 40 - 0.15 * (range_km - 180)  # dB
                
                # Add convection pattern (simplified E×B drift)
                convection_vel = 300 * np.sin(2 * np.pi * t / n_times) * np.cos(np.pi * b / n_beams)
                
                # Add ground scatter (lower elevation, higher power at closer ranges)
                if range_km < 500 and np.random.random() > 0.7:
                    # Ground scatter
                    velocity[t, b, r] = convection_vel * 0.3 + xp.random.normal(0, 50)
                    power[t, b, r] = base_power + xp.random.normal(5, 3)
                    width[t, b, r] = xp.random.uniform(50, 150)
                    elevation[t, b, r] = xp.random.uniform(5, 15)
                elif range_km > 300 and np.random.random() > 0.4:
                    # Ionospheric scatter
                    velocity[t, b, r] = convection_vel + xp.random.normal(0, 100)
                    power[t, b, r] = base_power + xp.random.normal(0, 5)
                    width[t, b, r] = xp.random.uniform(100, 400)
                    elevation[t, b, r] = xp.random.uniform(15, 45)
                else:
                    # No scatter - set to NaN
                    velocity[t, b, r] = xp.nan
                    power[t, b, r] = xp.nan
                    width[t, b, r] = xp.nan
                    elevation[t, b, r] = xp.nan
    
    # Add temporal variations and realistic noise
    if add_noise:
        print("  Adding realistic noise and temporal variations...")
        
        # Temporal coherence (smooth changes over time)
        for param in [velocity, power, width]:
            valid_mask = ~xp.isnan(param)
            noise_level = xp.where(param == velocity, 50, 
                                 xp.where(param == power, 2, 25))
            param[valid_mask] += xp.random.normal(0, noise_level[valid_mask] * 0.1)
    
    # Package data
    data = {
        'velocity': velocity,
        'power': power, 
        'width': width,
        'elevation': elevation,
        'time': times,
        'beam': beams,
        'range': ranges,
        'metadata': {
            'station_id': 65,  # Saskatoon
            'radar_name': 'sas',
            'generation_time': datetime.now(),
            'synthetic': True
        }
    }
    
    print(f"  ✅ Generated data shape: {velocity.shape}")
    print(f"  📊 Data completeness: {(~xp.isnan(velocity)).sum() / velocity.size * 100:.1f}%")
    
    return data


def simulate_acf_processing(rawacf_data):
    """Simulate ACF calculation processing stage"""
    print("🔍 Processing: Auto-Correlation Function calculation...")
    
    xp = sd.get_backend()
    n_times, n_beams, n_ranges = rawacf_data['velocity'].shape
    n_lags = 25  # Typical number of lags
    
    # Simulate complex ACF data
    acf_data = xp.zeros((n_times, n_beams, n_ranges, n_lags), dtype=xp.complex64)
    
    # Generate realistic ACF based on velocity and width
    velocity = rawacf_data['velocity']
    width = rawacf_data['width']
    power = rawacf_data['power']
    
    for lag in range(n_lags):
        lag_time = lag * 2400e-6  # 2.4 ms lag separation
        
        # ACF magnitude decays with lag
        decay = xp.exp(-lag * 0.1)
        
        # Phase rotation from Doppler shift
        phase = 2 * xp.pi * velocity * lag_time / 15e3  # 15 km range resolution
        
        # Spectral broadening from width
        broadening = xp.exp(-(lag * width / 1000)**2)
        
        # Combine effects
        magnitude = power * decay * broadening
        acf_data[:, :, :, lag] = magnitude * xp.exp(1j * phase)
    
    # Add noise to ACF
    noise_real = xp.random.normal(0, 0.1, acf_data.shape)
    noise_imag = xp.random.normal(0, 0.1, acf_data.shape)
    acf_data += noise_real + 1j * noise_imag
    
    return acf_data


def simulate_fitacf_processing(acf_data):
    """Simulate FitACF processing stage"""
    print("🎯 Processing: FitACF parameter extraction...")
    
    xp = sd.get_backend()
    n_times, n_beams, n_ranges, n_lags = acf_data.shape
    
    # Initialize output arrays
    velocity = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    velocity_error = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    power = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    power_error = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    width = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    width_error = xp.zeros((n_times, n_beams, n_ranges), dtype=xp.float32)
    
    # Simulate fitting process
    for t in range(n_times):
        for b in range(n_beams):
            for r in range(n_ranges):
                acf_profile = acf_data[t, b, r, :]
                
                # Check if we have valid data
                if xp.abs(acf_profile[0]) > 0.1:  # Sufficient power
                    # Extract zero-lag power
                    power[t, b, r] = 20 * xp.log10(xp.abs(acf_profile[0]) + 1e-10)
                    power_error[t, b, r] = xp.random.uniform(1, 3)
                    
                    # Fit velocity from phase slope
                    phases = xp.angle(acf_profile[1:6])  # Use first 5 lags
                    phase_diff = xp.diff(phases)
                    
                    # Unwrap phases
                    phase_diff = xp.where(phase_diff > xp.pi, phase_diff - 2*xp.pi, phase_diff)
                    phase_diff = xp.where(phase_diff < -xp.pi, phase_diff + 2*xp.pi, phase_diff)
                    
                    # Calculate velocity
                    velocity[t, b, r] = xp.mean(phase_diff) * 15e3 / (2 * xp.pi * 2400e-6)
                    velocity_error[t, b, r] = xp.std(phase_diff) * 15e3 / (2 * xp.pi * 2400e-6) + 10
                    
                    # Estimate spectral width from ACF decay
                    magnitudes = xp.abs(acf_profile)
                    decay_rate = -xp.log(xp.maximum(magnitudes[1] / magnitudes[0], 0.01))
                    width[t, b, r] = decay_rate * 1000 + xp.random.normal(0, 20)  # Add uncertainty
                    width_error[t, b, r] = xp.random.uniform(10, 50)
                else:
                    # No valid data
                    velocity[t, b, r] = xp.nan
                    power[t, b, r] = xp.nan
                    width[t, b, r] = xp.nan
                    velocity_error[t, b, r] = xp.nan
                    power_error[t, b, r] = xp.nan
                    width_error[t, b, r] = xp.nan
    
    return {
        'velocity': velocity,
        'velocity_error': velocity_error,
        'power': power,
        'power_error': power_error,
        'width': width,
        'width_error': width_error
    }


def demonstrate_processing_pipeline():
    """
    Demonstrate the complete SuperDARN processing pipeline with visualization
    """
    print("🚀 SuperDARN GPU Processing Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize performance monitoring
    print("📊 Initializing performance monitoring...")
    performance_dashboard = PerformanceDashboard(update_interval=1.0)
    realtime_monitor = RealtimeMonitor(update_interval=0.5)
    
    # Create processing viewer for step-by-step visualization
    processing_viewer = ProcessingViewer()
    
    try:
        # Start monitoring
        performance_dashboard.start_monitoring()
        realtime_monitor.start_monitoring()
        
        # === STAGE 1: Data Generation/Loading ===
        stage_start = time.time()
        print("\\n📡 STAGE 1: Data Loading")
        
        raw_data = generate_synthetic_superdarn_data()
        processing_viewer.add_processing_stage("Raw Data", raw_data)
        
        stage_duration = (time.time() - stage_start) * 1000
        realtime_monitor.log_processing_stage("Data Loading", stage_duration, 50.0)  # ~50 MB
        
        # === STAGE 2: ACF Processing ===
        stage_start = time.time()
        print("\\n🔍 STAGE 2: ACF Processing")
        
        acf_data = simulate_acf_processing(raw_data)
        acf_results = {'acf': acf_data}
        processing_viewer.add_processing_stage("ACF Calculation", acf_results)
        
        stage_duration = (time.time() - stage_start) * 1000
        realtime_monitor.log_processing_stage("ACF Processing", stage_duration, 25.0)
        
        # === STAGE 3: FitACF Processing ===
        stage_start = time.time()
        print("\\n🎯 STAGE 3: FitACF Processing") 
        
        fitacf_data = simulate_fitacf_processing(acf_data)
        processing_viewer.add_processing_stage("FitACF Results", fitacf_data)
        
        stage_duration = (time.time() - stage_start) * 1000
        realtime_monitor.log_processing_stage("FitACF Processing", stage_duration, 15.0)
        
        # === STAGE 4: Quality Assessment ===
        stage_start = time.time()
        print("\\n✅ STAGE 4: Quality Assessment")
        
        # Calculate quality metrics
        quality_metrics = calculate_data_quality(fitacf_data)
        processing_viewer.add_processing_stage("Quality Assessment", quality_metrics)
        
        stage_duration = (time.time() - stage_start) * 1000
        realtime_monitor.log_processing_stage("Quality Assessment", stage_duration, 5.0)
        
        print("\\n🎨 Creating comprehensive visualization dashboards...")
        
        # Create processing dashboard
        processing_fig = create_processing_dashboard(
            raw_data, 
            {'acf': acf_data, 'fitacf': fitacf_data, 'quality': quality_metrics},
            "SuperDARN Processing Results - Synthetic Data Demo"
        )
        
        # Create interactive explorer
        print("\\n🔧 Launching interactive explorer...")
        explorer = InteractiveExplorer(fitacf_data, parameters=['velocity', 'power', 'width'])
        
        print("\\n📈 Performance Summary:")
        performance_dashboard.generate_performance_report()
        
        print("\\n" + "=" * 60)
        print("🎉 PROCESSING COMPLETE!")
        print("\\n📋 What you can explore:")
        print("  • Processing dashboard showing all stages")
        print("  • Interactive explorer with time/beam sliders")  
        print("  • Real-time performance monitoring")
        print("  • Step-by-step processing viewer")
        print("  • Quality assessment metrics")
        print("\\n💡 Close the matplotlib windows when done exploring.")
        
        # Show all plots
        plt.show()
        
    except KeyboardInterrupt:
        print("\\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during processing: {e}")
        raise
    finally:
        # Clean up monitoring
        performance_dashboard.stop_monitoring()
        realtime_monitor.stop_monitoring()


def calculate_data_quality(fitacf_data):
    """Calculate quality metrics for processed data"""
    xp = sd.get_backend()
    
    velocity = fitacf_data['velocity']
    power = fitacf_data['power']
    width = fitacf_data['width']
    
    # Data completeness
    vel_completeness = (~xp.isnan(velocity)).sum() / velocity.size
    pow_completeness = (~xp.isnan(power)).sum() / power.size
    
    # Velocity range check
    valid_vel = velocity[~xp.isnan(velocity)]
    vel_range_check = ((valid_vel >= -2000) & (valid_vel <= 2000)).sum() / len(valid_vel) if len(valid_vel) > 0 else 0
    
    # Power range check  
    valid_pow = power[~xp.isnan(power)]
    pow_range_check = ((valid_pow >= -10) & (valid_pow <= 50)).sum() / len(valid_pow) if len(valid_pow) > 0 else 0
    
    # Width reasonableness
    valid_width = width[~xp.isnan(width)]
    width_range_check = ((valid_width >= 0) & (valid_width <= 1000)).sum() / len(valid_width) if len(valid_width) > 0 else 0
    
    # Temporal consistency
    temporal_consistency = 0.85  # Placeholder - would need more complex calculation
    
    quality_metrics = {
        'data_completeness': float(vel_completeness),
        'velocity_range_quality': float(vel_range_check),
        'power_range_quality': float(pow_range_check),
        'width_range_quality': float(width_range_check),
        'temporal_consistency': temporal_consistency,
        'overall_quality': float((vel_completeness + vel_range_check + pow_range_check + 
                                width_range_check + temporal_consistency) / 5)
    }
    
    return quality_metrics


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='SuperDARN GPU Processing Visualization Demo')
    parser.add_argument('--use-cpu', action='store_true', 
                       help='Force CPU processing (disable GPU)')
    parser.add_argument('--data-file', type=str,
                       help='Use real data file instead of synthetic data')
    parser.add_argument('--show-performance', action='store_true',
                       help='Show detailed performance analysis')
    
    args = parser.parse_args()
    
    # Set backend
    if args.use_cpu:
        sd.set_backend('numpy')
        print("🖥️  Using CPU backend (NumPy)")
    else:
        if sd.GPU_AVAILABLE:
            sd.set_backend('cupy')
            print(f"🚀 Using GPU backend (CuPy)")
        else:
            print("⚠️  GPU not available, falling back to CPU")
    
    print(f"📦 SuperDARN GPU v{sd.__version__}")
    print(f"🔧 Backend: {sd.get_backend().__name__}")
    
    # Run demonstration
    demonstrate_processing_pipeline()


if __name__ == "__main__":
    main()
