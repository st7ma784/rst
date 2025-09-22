"""
SuperDARN GPU Visualization Module
=================================

Comprehensive visualization system for SuperDARN data processing with:
- Real-time pipeline monitoring
- Scientific data visualization 
- Performance profiling displays
- Interactive processing exploration
- GPU utilization monitoring

Key Features:
- GPU-accelerated plotting with CuPy/Matplotlib integration
- Live processing visualizations for scientists
- Performance dashboards for optimization
- Interactive parameter exploration
- Publication-ready scientific plots
"""

from .realtime import RealtimeMonitor, ProcessingViewer
from .scientific import (
    plot_range_time, plot_fan, plot_grid, plot_convection_map,
    plot_acf, plot_spectrum, plot_elevation_angle
)
from .performance import (
    PerformanceDashboard, GPUMonitor, 
    plot_processing_times, plot_memory_usage
)
from .interactive import (
    InteractiveExplorer, ParameterTuner,
    ProcessingComparison, ValidationViewer
)
from .dashboards import (
    create_processing_dashboard, create_performance_dashboard,
    create_validation_dashboard
)

__all__ = [
    # Real-time monitoring
    'RealtimeMonitor', 'ProcessingViewer',
    
    # Scientific visualization
    'plot_range_time', 'plot_fan', 'plot_grid', 'plot_convection_map',
    'plot_acf', 'plot_spectrum', 'plot_elevation_angle',
    
    # Performance monitoring
    'PerformanceDashboard', 'GPUMonitor',
    'plot_processing_times', 'plot_memory_usage',
    
    # Interactive tools
    'InteractiveExplorer', 'ParameterTuner',
    'ProcessingComparison', 'ValidationViewer',
    
    # Dashboard creation
    'create_processing_dashboard', 'create_performance_dashboard',
    'create_validation_dashboard'
]
