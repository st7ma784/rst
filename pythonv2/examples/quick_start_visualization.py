#!/usr/bin/env python3
"""
Quick Start Example for SuperDARN GPU Visualization

This script demonstrates the key visualization capabilities in a simple,
easy-to-run format. Perfect for scientists who want to quickly see what
the visualization system can do.

Usage:
    python quick_start_visualization.py
    python quick_start_visualization.py --gpu-demo    # If you have GPU
    python quick_start_visualization.py --save-plots  # Save plots to files
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, '..')

try:
    import superdarn_gpu as sd
    from superdarn_gpu.visualization import (
        plot_range_time, plot_fan, plot_acf,
        create_processing_dashboard,
        InteractiveExplorer, PerformanceDashboard
    )
    print("✅ SuperDARN GPU visualization system loaded successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the examples/ directory")
    sys.exit(1)

def create_sample_data():
    """Create sample SuperDARN data for demonstration"""
    print("🔬 Creating sample SuperDARN data...")
    
    # Use NumPy for compatibility (CuPy if available)
    xp = sd.get_backend()
    
    # Standard SuperDARN dimensions
    n_times = 30    # 1 hour at 2-minute intervals
    n_beams = 16    # Standard beam pattern
    n_ranges = 50   # Range gates to ~900 km
    
    # Initialize data arrays
    velocity = xp.full((n_times, n_beams, n_ranges), xp.nan, dtype=xp.float32)
    power = xp.full((n_times, n_beams, n_ranges), xp.nan, dtype=xp.float32)
    width = xp.full((n_times, n_beams, n_ranges), xp.nan, dtype=xp.float32)
    
    # Create realistic ionospheric convection pattern
    ranges = xp.arange(180, 180 + n_ranges * 15, 15)
    
    for t in range(n_times):
        for b in range(n_beams):
            for r in range(n_ranges):
                range_km = ranges[r]
                
                # Simulate E/F region backscatter
                if range_km < 500:  # E region
                    scatter_prob = 0.3
                    base_velocity = 200
                else:  # F region
                    scatter_prob = 0.6 * xp.exp(-(range_km - 600)**2 / (2 * 200**2))
                    base_velocity = 500
                
                if xp.random.random() < scatter_prob:
                    # Convection pattern with diurnal variation
                    time_factor = xp.sin(2 * xp.pi * t / n_times)
                    beam_factor = xp.cos(xp.pi * b / n_beams)
                    
                    velocity[t, b, r] = base_velocity * time_factor * beam_factor + xp.random.normal(0, 50)
                    power[t, b, r] = 25 + 15 * xp.random.random() - 0.01 * (range_km - 400)
                    width[t, b, r] = xp.random.uniform(50, 250)
    
    # Package data
    data = {
        'velocity': velocity,
        'power': power, 
        'width': width,
        'time': [datetime.now() + timedelta(minutes=2*i) for i in range(n_times)],
        'beam': xp.arange(n_beams),
        'range': ranges,
        'metadata': {
            'radar': 'sas',
            'date': datetime.now().date(),
            'synthetic': True
        }
    }
    
    completeness = (~xp.isnan(velocity)).sum() / velocity.size * 100
    print(f"  📊 Sample data created: {velocity.shape}")
    print(f"  ✅ Data completeness: {completeness:.1f}%")
    
    return data

def demo_basic_plots(data, save_plots=False):
    """Demonstrate basic scientific plotting"""
    print("\n📈 Creating basic scientific plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SuperDARN Quick Start - Scientific Visualization', fontsize=16)
    
    # Velocity range-time plot
    plot_range_time(data, parameter='velocity', ax=axes[0,0], 
                   title='Line-of-sight Velocity (m/s)')
    
    # Power range-time plot
    plot_range_time(data, parameter='power', ax=axes[0,1],
                   title='Backscatter Power (dB)')
    
    # Spectral width
    plot_range_time(data, parameter='width', ax=axes[1,0],
                   title='Spectral Width (m/s)')
    
    # Fan plot at mid-time
    mid_time = data['velocity'].shape[0] // 2
    plot_fan(data, parameter='velocity', time_idx=mid_time, ax=axes[1,1],
            title=f'Velocity Fan Plot (t={mid_time})')
    
    plt.tight_layout()
    
    if save_plots:
        fig.savefig('superdarn_basic_plots.png', dpi=300, bbox_inches='tight')
        print("  💾 Saved as 'superdarn_basic_plots.png'")
    
    plt.show()
    print("  ✅ Basic plots complete!")

def demo_dashboard(data, save_plots=False):
    """Demonstrate dashboard creation"""
    print("\n🎨 Creating comprehensive dashboard...")
    
    # Simulate processing results
    xp = sd.get_backend()
    processing_results = {
        'raw_data': data,
        'acf': {
            'magnitude': xp.random.random(data['velocity'].shape + (10,)),
            'phase': xp.random.uniform(-3.14, 3.14, data['velocity'].shape + (10,))
        },
        'fitacf': {
            'velocity': data['velocity'],
            'power': data['power'],
            'width': data['width']
        },
        'quality': {
            'data_completeness': 0.73,
            'velocity_quality': 0.86,
            'power_quality': 0.91
        }
    }
    
    # Create dashboard
    dashboard_fig = create_processing_dashboard(
        data, 
        processing_results,
        title="SuperDARN Processing Dashboard - Quick Start Demo"
    )
    
    if save_plots:
        dashboard_fig.savefig('superdarn_dashboard.png', dpi=300, bbox_inches='tight')
        print("  💾 Saved as 'superdarn_dashboard.png'")
    
    plt.show()
    print("  ✅ Dashboard complete!")

def demo_interactive_explorer(data):
    """Demonstrate interactive data exploration"""
    print("\n🎛️ Creating interactive data explorer...")
    print("   Use the controls to explore the data interactively!")
    
    try:
        explorer = InteractiveExplorer(data, parameters=['velocity', 'power', 'width'])
        print("  ✅ Interactive explorer launched!")
        print("     • Use time slider to navigate through observations")
        print("     • Select different beams with beam slider")
        print("     • Switch parameters with radio buttons")
        print("     • Click 'Play' to animate through time")
        plt.show()
    except Exception as e:
        print(f"  ⚠️  Interactive explorer failed: {e}")
        print("     (This may happen in some environments - the basic plots still work!)")

def demo_performance_monitoring(data, gpu_demo=False):
    """Demonstrate performance monitoring"""
    print(f"\n⚡ Performance monitoring demo (GPU: {gpu_demo})...")
    
    try:
        # Create performance dashboard
        perf = PerformanceDashboard(update_interval=0.5)
        perf.start_monitoring()
        
        print("  🔄 Simulating processing stages...")
        
        # Simulate processing stages with delays
        import time
        stages = [
            ("Data Loading", 0.1, 25.4),
            ("ACF Calculation", 0.3, 45.2),
            ("FitACF Processing", 0.2, 38.1),
            ("Quality Control", 0.1, 12.3)
        ]
        
        for stage_name, delay, throughput_mbps in stages:
            print(f"    Processing {stage_name}...")
            time.sleep(delay)
            perf.log_stage_performance(stage_name, delay*1000, throughput_mbps)
        
        # Generate performance report
        print("  📊 Generating performance report...")
        perf.generate_performance_report()
        perf.stop_monitoring()
        
        plt.show()
        print("  ✅ Performance monitoring complete!")
        
    except Exception as e:
        print(f"  ⚠️  Performance monitoring failed: {e}")

def print_system_info():
    """Print system information"""
    print("\n🔧 System Information:")
    print(f"   SuperDARN GPU Version: {sd.__version__}")
    print(f"   Backend: {sd.get_backend().__name__}")
    
    if sd.GPU_AVAILABLE:
        print("   GPU: ✅ Available (CuPy installed)")
    else:
        print("   GPU: ⚠️  Not available (install CuPy for GPU acceleration)")
    
    print(f"   Python: {sys.version.split()[0]}")
    
    # Check matplotlib backend
    print(f"   Matplotlib backend: {plt.get_backend()}")

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='SuperDARN Visualization Quick Start')
    parser.add_argument('--gpu-demo', action='store_true', 
                       help='Enable GPU-specific demonstrations')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to PNG files')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive demonstrations')
    
    args = parser.parse_args()
    
    print("🚀 SuperDARN GPU Visualization - Quick Start Demo")
    print("=" * 55)
    
    # Print system information
    print_system_info()
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Run demonstrations
    try:
        # Basic plots
        demo_basic_plots(sample_data, save_plots=args.save_plots)
        
        # Dashboard
        demo_dashboard(sample_data, save_plots=args.save_plots)
        
        # Interactive explorer (skip if requested)
        if not args.no_interactive:
            demo_interactive_explorer(sample_data)
        
        # Performance monitoring
        demo_performance_monitoring(sample_data, gpu_demo=args.gpu_demo)
        
        print("\n🎉 Quick Start Demo Complete!")
        print("=" * 35)
        print("\n📚 What's Next?")
        print("   • Try the Jupyter notebook tutorial: superdarn_visualization_tutorial.ipynb")
        print("   • Run the comprehensive demo: demo_processing_visualization.py")
        print("   • Read the documentation: docs/visualization_guide.md")
        print("   • Apply these tools to your real SuperDARN data!")
        
        if not sd.GPU_AVAILABLE:
            print("\n💡 Pro Tip: Install CuPy for GPU acceleration:")
            print("   pip install cupy-cuda11x  # For CUDA 11.x")
            print("   pip install cupy-cuda12x  # For CUDA 12.x")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
