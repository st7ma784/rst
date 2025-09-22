"""
Interactive Visualization Tools
==============================

Interactive widgets and tools for exploring SuperDARN data and
tuning processing parameters in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display
import threading
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from ..core.backends import get_backend
from .scientific import plot_range_time, plot_fan, plot_acf


class InteractiveExplorer:
    """
    Interactive explorer for SuperDARN data
    
    Provides sliders and controls to explore data across different
    dimensions (time, beam, range, frequency) with real-time updates.
    """
    
    def __init__(self, data, parameters=['velocity', 'power', 'width']):
        self.data = data
        self.parameters = parameters
        self.current_parameter = parameters[0]
        self.current_time_idx = 0
        self.current_beam_idx = 0
        
        # Extract data dimensions
        if isinstance(data, dict):
            sample_data = data[parameters[0]]
        else:
            sample_data = getattr(data, parameters[0])
            
        # Handle GPU data
        if hasattr(sample_data, 'get'):
            sample_data = sample_data.get()
            
        self.n_times, self.n_beams, self.n_ranges = sample_data.shape
        
        # Setup interactive figure
        self.fig = plt.figure(figsize=(16, 12))
        
        # Main plot area
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        
        # Range profile
        self.ax_range = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
        
        # Time series  
        self.ax_time = plt.subplot2grid((4, 4), (3, 0), colspan=3)
        
        # Control panel
        self.ax_controls = plt.subplot2grid((4, 4), (3, 3))
        self.ax_controls.set_visible(False)
        
        self._setup_controls()
        self._setup_initial_plots()
        
    def _setup_controls(self):
        """Setup interactive controls"""
        # Time slider
        ax_time_slider = plt.axes([0.1, 0.02, 0.5, 0.03])
        self.time_slider = Slider(ax_time_slider, 'Time', 0, self.n_times-1, 
                                 valinit=0, valfmt='%d')
        self.time_slider.on_changed(self._on_time_change)
        
        # Beam slider
        ax_beam_slider = plt.axes([0.1, 0.06, 0.5, 0.03])
        self.beam_slider = Slider(ax_beam_slider, 'Beam', 0, self.n_beams-1,
                                 valinit=0, valfmt='%d')
        self.beam_slider.on_changed(self._on_beam_change)
        
        # Parameter selection
        ax_param_radio = plt.axes([0.65, 0.02, 0.15, 0.08])
        self.param_radio = RadioButtons(ax_param_radio, self.parameters)
        self.param_radio.on_clicked(self._on_parameter_change)
        
        # Animation controls
        ax_play_btn = plt.axes([0.82, 0.06, 0.06, 0.04])
        self.play_btn = Button(ax_play_btn, 'Play')
        self.play_btn.on_clicked(self._toggle_animation)
        
        ax_reset_btn = plt.axes([0.89, 0.06, 0.06, 0.04])
        self.reset_btn = Button(ax_reset_btn, 'Reset')
        self.reset_btn.on_clicked(self._reset_view)
        
    def _setup_initial_plots(self):
        """Setup initial plots"""
        self._update_main_plot()
        self._update_range_profile()
        self._update_time_series()
        
        plt.tight_layout()
        
    def _update_main_plot(self):
        """Update main range-time plot"""
        self.ax_main.clear()
        
        # Get current data
        if isinstance(self.data, dict):
            plot_data = self.data[self.current_parameter]
        else:
            plot_data = getattr(self.data, self.current_parameter)
            
        # Handle GPU data
        if hasattr(plot_data, 'get'):
            plot_data = plot_data.get()
        
        # Create range-time plot
        im = self.ax_main.imshow(plot_data[:, self.current_beam_idx, :].T, 
                               aspect='auto', origin='lower',
                               cmap=self._get_colormap(self.current_parameter))
        
        # Add current time indicator
        self.ax_main.axvline(x=self.current_time_idx, color='white', 
                           linewidth=2, alpha=0.8)
        
        self.ax_main.set_title(f'{self.current_parameter.title()} - Beam {self.current_beam_idx}')
        self.ax_main.set_xlabel('Time Index')
        self.ax_main.set_ylabel('Range Gate')
        
        # Add colorbar
        plt.colorbar(im, ax=self.ax_main)
        
    def _update_range_profile(self):
        """Update range profile plot"""
        self.ax_range.clear()
        
        # Get current data
        if isinstance(self.data, dict):
            plot_data = self.data[self.current_parameter]
        else:
            plot_data = getattr(self.data, self.current_parameter)
            
        # Handle GPU data
        if hasattr(plot_data, 'get'):
            plot_data = plot_data.get()
        
        # Extract range profile at current time and beam
        range_profile = plot_data[self.current_time_idx, self.current_beam_idx, :]
        
        self.ax_range.plot(range_profile, range(len(range_profile)), 'b-', linewidth=2)
        self.ax_range.set_ylabel('Range Gate')
        self.ax_range.set_xlabel(f'{self.current_parameter.title()}')
        self.ax_range.set_title(f'Range Profile\\n(t={self.current_time_idx})')
        self.ax_range.grid(True, alpha=0.3)
        
    def _update_time_series(self):
        """Update time series plot"""
        self.ax_time.clear()
        
        # Get current data
        if isinstance(self.data, dict):
            plot_data = self.data[self.current_parameter]
        else:
            plot_data = getattr(self.data, self.current_parameter)
            
        # Handle GPU data
        if hasattr(plot_data, 'get'):
            plot_data = plot_data.get()
        
        # Average over range gates for time series
        time_series = np.nanmean(plot_data[:, self.current_beam_idx, :], axis=1)
        
        self.ax_time.plot(time_series, 'g-', linewidth=2)
        self.ax_time.axvline(x=self.current_time_idx, color='red', 
                           linewidth=2, alpha=0.8)
        
        self.ax_time.set_xlabel('Time Index')
        self.ax_time.set_ylabel(f'Mean {self.current_parameter.title()}')
        self.ax_time.set_title(f'Time Series (Beam {self.current_beam_idx})')
        self.ax_time.grid(True, alpha=0.3)
        
    def _get_colormap(self, parameter):
        """Get appropriate colormap for parameter"""
        colormaps = {
            'velocity': 'RdBu_r',
            'power': 'viridis',
            'width': 'plasma',
            'elevation': 'coolwarm'
        }
        return colormaps.get(parameter, 'viridis')
        
    def _on_time_change(self, val):
        """Handle time slider change"""
        self.current_time_idx = int(val)
        self._update_main_plot()
        self._update_range_profile()
        self._update_time_series()
        self.fig.canvas.draw()
        
    def _on_beam_change(self, val):
        """Handle beam slider change"""
        self.current_beam_idx = int(val)
        self._update_main_plot()
        self._update_range_profile()
        self._update_time_series()
        self.fig.canvas.draw()
        
    def _on_parameter_change(self, label):
        """Handle parameter selection change"""
        self.current_parameter = label
        self._update_main_plot()
        self._update_range_profile()
        self._update_time_series()
        self.fig.canvas.draw()
        
    def _toggle_animation(self, event):
        """Toggle animation playback"""
        if not hasattr(self, 'is_animating'):
            self.is_animating = False
            
        if not self.is_animating:
            self.is_animating = True
            self.play_btn.label.set_text('Stop')
            self._start_animation()
        else:
            self.is_animating = False
            self.play_btn.label.set_text('Play')
            
    def _start_animation(self):
        """Start animation thread"""
        def animate():
            while self.is_animating and self.current_time_idx < self.n_times - 1:
                self.current_time_idx += 1
                self.time_slider.set_val(self.current_time_idx)
                time.sleep(0.5)  # Animation speed
            
            self.is_animating = False
            self.play_btn.label.set_text('Play')
            
        animation_thread = threading.Thread(target=animate)
        animation_thread.daemon = True
        animation_thread.start()
        
    def _reset_view(self, event):
        """Reset view to initial state"""
        self.current_time_idx = 0
        self.current_beam_idx = 0
        self.time_slider.set_val(0)
        self.beam_slider.set_val(0)
        self._update_main_plot()
        self._update_range_profile()
        self._update_time_series()
        self.fig.canvas.draw()


class ParameterTuner:
    """
    Interactive parameter tuning tool for SuperDARN processing
    
    Allows real-time adjustment of processing parameters with
    immediate visualization of results.
    """
    
    def __init__(self, processing_function, default_params, data):
        self.processing_function = processing_function
        self.default_params = default_params
        self.data = data
        self.current_params = default_params.copy()
        
        # Setup figure
        self.fig, (self.ax_original, self.ax_processed) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('Parameter Tuning: Real-time Processing Comparison', fontsize=14)
        
        # Initial processing
        self.original_result = self.processing_function(data, **default_params)
        self.processed_result = self.original_result
        
        self._setup_controls()
        self._update_plots()
        
    def _setup_controls(self):
        """Setup parameter control sliders"""
        self.sliders = {}
        slider_height = 0.03
        slider_spacing = 0.04
        
        for i, (param_name, param_value) in enumerate(self.default_params.items()):
            # Determine slider range based on parameter type and value
            if isinstance(param_value, bool):
                # Skip boolean parameters for now
                continue
            elif isinstance(param_value, int):
                val_min = max(1, param_value // 2)
                val_max = param_value * 2
                valfmt = '%d'
            else:  # float
                val_min = param_value * 0.1
                val_max = param_value * 2.0
                valfmt = '%.2f'
            
            # Create slider
            ax_slider = plt.axes([0.1, 0.02 + i * slider_spacing, 0.3, slider_height])
            slider = Slider(ax_slider, param_name, val_min, val_max,
                           valinit=param_value, valfmt=valfmt)
            slider.on_changed(lambda val, name=param_name: self._on_parameter_change(name, val))
            
            self.sliders[param_name] = slider
        
        # Reset button
        ax_reset = plt.axes([0.45, 0.02, 0.08, 0.04])
        self.reset_btn = Button(ax_reset, 'Reset')
        self.reset_btn.on_clicked(self._reset_parameters)
        
    def _update_plots(self):
        """Update comparison plots"""
        # Original result (left)
        self.ax_original.clear()
        self.ax_original.set_title('Original Parameters')
        if isinstance(self.original_result, dict) and 'velocity' in self.original_result:
            data_to_plot = self.original_result['velocity']
            # Handle GPU data
            if hasattr(data_to_plot, 'get'):
                data_to_plot = data_to_plot.get()
            self.ax_original.imshow(np.mean(data_to_plot, axis=1).T, 
                                  aspect='auto', origin='lower', cmap='RdBu_r')
        
        # Processed result (right)
        self.ax_processed.clear()
        self.ax_processed.set_title('Current Parameters')
        if isinstance(self.processed_result, dict) and 'velocity' in self.processed_result:
            data_to_plot = self.processed_result['velocity']
            # Handle GPU data
            if hasattr(data_to_plot, 'get'):
                data_to_plot = data_to_plot.get()
            self.ax_processed.imshow(np.mean(data_to_plot, axis=1).T,
                                   aspect='auto', origin='lower', cmap='RdBu_r')
        
        self.fig.canvas.draw()
        
    def _on_parameter_change(self, param_name, value):
        """Handle parameter change"""
        # Update parameter value
        if isinstance(self.default_params[param_name], int):
            self.current_params[param_name] = int(value)
        else:
            self.current_params[param_name] = value
        
        # Reprocess data with new parameters
        try:
            self.processed_result = self.processing_function(self.data, **self.current_params)
            self._update_plots()
        except Exception as e:
            print(f"Processing error with {param_name}={value}: {e}")
            
    def _reset_parameters(self, event):
        """Reset all parameters to defaults"""
        self.current_params = self.default_params.copy()
        for param_name, slider in self.sliders.items():
            slider.set_val(self.default_params[param_name])
        
        self.processed_result = self.original_result
        self._update_plots()


class ProcessingComparison:
    """
    Side-by-side comparison of different processing approaches
    """
    
    def __init__(self, data, processors, labels=None):
        self.data = data
        self.processors = processors  # List of processing functions
        self.labels = labels or [f'Method {i+1}' for i in range(len(processors))]
        
        # Process data with all methods
        self.results = []
        self.processing_times = []
        
        for processor in processors:
            start_time = time.time()
            try:
                result = processor(data)
                self.results.append(result)
                self.processing_times.append(time.time() - start_time)
            except Exception as e:
                print(f"Processing error: {e}")
                self.results.append(None)
                self.processing_times.append(0)
        
        self._create_comparison_plot()
        
    def _create_comparison_plot(self):
        """Create comparison visualization"""
        n_methods = len(self.processors)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        fig.suptitle('Processing Method Comparison', fontsize=16)
        
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (result, label, proc_time) in enumerate(zip(self.results, self.labels, self.processing_times)):
            # Top row: velocity data
            if result is not None and 'velocity' in result:
                vel_data = result['velocity']
                # Handle GPU data
                if hasattr(vel_data, 'get'):
                    vel_data = vel_data.get()
                    
                im = axes[0, i].imshow(np.mean(vel_data, axis=1).T, 
                                     aspect='auto', origin='lower', cmap='RdBu_r',
                                     vmin=-1000, vmax=1000)
                axes[0, i].set_title(f'{label}\\nVelocity ({proc_time:.2f}s)')
                plt.colorbar(im, ax=axes[0, i])
            else:
                axes[0, i].text(0.5, 0.5, 'No Data', ha='center', va='center',
                              transform=axes[0, i].transAxes)
                axes[0, i].set_title(f'{label}\\nFailed')
            
            # Bottom row: power data
            if result is not None and 'power' in result:
                pow_data = result['power']
                # Handle GPU data
                if hasattr(pow_data, 'get'):
                    pow_data = pow_data.get()
                    
                im = axes[1, i].imshow(np.mean(pow_data, axis=1).T,
                                     aspect='auto', origin='lower', cmap='viridis',
                                     vmin=0, vmax=50)
                axes[1, i].set_title(f'{label}\\nPower')
                plt.colorbar(im, ax=axes[1, i])
            else:
                axes[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center',
                              transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'{label}\\nNo Power Data')
        
        plt.tight_layout()
        return fig


class ValidationViewer:
    """
    Interactive validation and quality assessment viewer
    """
    
    def __init__(self, original_data, processed_data, validation_metrics=None):
        self.original_data = original_data
        self.processed_data = processed_data
        self.validation_metrics = validation_metrics or {}
        
        # Setup validation figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Processing Validation Dashboard', fontsize=16)
        
        self._setup_validation_plots()
        
    def _setup_validation_plots(self):
        """Setup validation visualization plots"""
        # Original data (top-left)
        self.ax_orig = self.axes[0, 0]
        self.ax_orig.set_title('Original Data')
        
        # Processed data (top-center)
        self.ax_proc = self.axes[0, 1]
        self.ax_proc.set_title('Processed Data')
        
        # Difference plot (top-right)
        self.ax_diff = self.axes[0, 2]
        self.ax_diff.set_title('Difference (Processed - Original)')
        
        # Statistics plot (bottom-left)
        self.ax_stats = self.axes[1, 0]
        self.ax_stats.set_title('Statistical Comparison')
        
        # Quality metrics (bottom-center)
        self.ax_quality = self.axes[1, 1]
        self.ax_quality.set_title('Quality Metrics')
        
        # Validation summary (bottom-right)
        self.ax_summary = self.axes[1, 2]
        self.ax_summary.set_title('Validation Summary')
        
        self._update_validation_plots()
        
    def _update_validation_plots(self):
        """Update validation plots with current data"""
        # Extract velocity data for comparison
        if isinstance(self.original_data, dict):
            orig_vel = self.original_data.get('velocity', np.zeros((100, 16, 75)))
        else:
            orig_vel = getattr(self.original_data, 'velocity', np.zeros((100, 16, 75)))
            
        if isinstance(self.processed_data, dict):
            proc_vel = self.processed_data.get('velocity', np.zeros((100, 16, 75)))
        else:
            proc_vel = getattr(self.processed_data, 'velocity', np.zeros((100, 16, 75)))
        
        # Handle GPU data
        if hasattr(orig_vel, 'get'):
            orig_vel = orig_vel.get()
        if hasattr(proc_vel, 'get'):
            proc_vel = proc_vel.get()
        
        # Original data plot
        orig_avg = np.nanmean(orig_vel, axis=1)
        im1 = self.ax_orig.imshow(orig_avg.T, aspect='auto', origin='lower',
                                cmap='RdBu_r', vmin=-1000, vmax=1000)
        plt.colorbar(im1, ax=self.ax_orig)
        
        # Processed data plot
        proc_avg = np.nanmean(proc_vel, axis=1)
        im2 = self.ax_proc.imshow(proc_avg.T, aspect='auto', origin='lower',
                                cmap='RdBu_r', vmin=-1000, vmax=1000)
        plt.colorbar(im2, ax=self.ax_proc)
        
        # Difference plot
        diff = proc_avg - orig_avg
        im3 = self.ax_diff.imshow(diff.T, aspect='auto', origin='lower',
                                cmap='RdBu_r', vmin=-100, vmax=100)
        plt.colorbar(im3, ax=self.ax_diff)
        
        # Statistical comparison
        orig_flat = orig_vel.flatten()
        proc_flat = proc_vel.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(orig_flat) | np.isnan(proc_flat))
        orig_valid = orig_flat[valid_mask]
        proc_valid = proc_flat[valid_mask]
        
        if len(orig_valid) > 0:
            self.ax_stats.scatter(orig_valid[::100], proc_valid[::100], 
                                alpha=0.5, s=1)
            self.ax_stats.plot([orig_valid.min(), orig_valid.max()],
                             [orig_valid.min(), orig_valid.max()], 'r--', alpha=0.8)
            self.ax_stats.set_xlabel('Original Velocity (m/s)')
            self.ax_stats.set_ylabel('Processed Velocity (m/s)')
            
            # Calculate correlation
            correlation = np.corrcoef(orig_valid, proc_valid)[0, 1]
            self.ax_stats.text(0.05, 0.95, f'R = {correlation:.4f}',
                             transform=self.ax_stats.transAxes, va='top')
        
        # Quality metrics
        metrics = self.validation_metrics
        if metrics:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' 
                     for v in metric_values]
            
            bars = self.ax_quality.barh(metric_names, metric_values, color=colors)
            self.ax_quality.set_xlim(0, 1)
            self.ax_quality.set_xlabel('Quality Score')
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                self.ax_quality.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                   f'{value:.3f}', va='center')
        
        # Validation summary
        self.ax_summary.axis('off')
        
        if len(orig_valid) > 0:
            rmse = np.sqrt(np.mean((orig_valid - proc_valid)**2))
            mae = np.mean(np.abs(orig_valid - proc_valid))
            bias = np.mean(proc_valid - orig_valid)
            
            summary_text = f"""Validation Summary:
            
Correlation: {correlation:.4f}
RMSE: {rmse:.2f} m/s
MAE: {mae:.2f} m/s
Bias: {bias:.2f} m/s

Data Points: {len(orig_valid):,}
NaN Fraction: {1 - len(orig_valid)/len(orig_flat):.3f}
            """
        else:
            summary_text = "No valid data for comparison"
            
        self.ax_summary.text(0.1, 0.5, summary_text, transform=self.ax_summary.transAxes,
                           fontsize=11, verticalalignment='center')
        
        plt.tight_layout()
