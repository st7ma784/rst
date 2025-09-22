"""
Real-time Processing Visualization
==================================

Live monitoring and visualization of SuperDARN data processing pipeline.
Shows scientists what's happening during GPU-accelerated processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import time
import threading
from queue import Queue
from datetime import datetime, timedelta

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from ..core.backends import get_backend


class RealtimeMonitor:
    """
    Real-time monitoring of SuperDARN processing pipeline
    
    Shows live updates of:
    - Data throughput
    - Processing stages
    - GPU utilization
    - Quality metrics
    """
    
    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.is_running = False
        self.data_queue = Queue()
        
        # Monitoring data
        self.timestamps = []
        self.throughput = []
        self.gpu_memory = []
        self.processing_times = []
        self.quality_metrics = []
        
        # Setup figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('SuperDARN GPU Processing Monitor', fontsize=16)
        
        self._setup_plots()
        
    def _setup_plots(self):
        """Initialize all monitoring plots"""
        # Throughput plot
        self.ax_throughput = self.axes[0, 0]
        self.ax_throughput.set_title('Data Throughput (MB/s)')
        self.ax_throughput.set_xlabel('Time')
        self.ax_throughput.set_ylabel('Throughput')
        self.throughput_line, = self.ax_throughput.plot([], [], 'b-', lw=2)
        
        # GPU Memory usage
        self.ax_memory = self.axes[0, 1]
        self.ax_memory.set_title('GPU Memory Usage (%)')
        self.ax_memory.set_xlabel('Time')
        self.ax_memory.set_ylabel('Memory %')
        self.ax_memory.set_ylim(0, 100)
        self.memory_line, = self.ax_memory.plot([], [], 'r-', lw=2)
        
        # Processing times
        self.ax_timing = self.axes[1, 0]
        self.ax_timing.set_title('Stage Processing Times (ms)')
        self.ax_timing.set_xlabel('Processing Stage')
        self.ax_timing.set_ylabel('Time (ms)')
        
        # Quality metrics
        self.ax_quality = self.axes[1, 1]
        self.ax_quality.set_title('Data Quality Metrics')
        self.ax_quality.set_xlabel('Time')
        self.ax_quality.set_ylabel('Quality Score')
        self.quality_line, = self.ax_quality.plot([], [], 'g-', lw=2)
        
        plt.tight_layout()
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plots, interval=int(self.update_interval * 1000),
            blit=False, cache_frame_data=False
        )
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            # Collect monitoring data
            current_time = time.time()
            
            # Get GPU memory usage if available
            if GPU_AVAILABLE and cp.cuda.is_available():
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                total_bytes = mempool.total_bytes()
                if total_bytes > 0:
                    memory_percent = (used_bytes / total_bytes) * 100
                else:
                    memory_percent = 0
            else:
                memory_percent = 0
                
            # Add data point
            self.data_queue.put({
                'timestamp': current_time,
                'memory_percent': memory_percent,
                'throughput': np.random.uniform(50, 200),  # Placeholder
                'quality': np.random.uniform(0.8, 1.0)     # Placeholder
            })
            
            time.sleep(self.update_interval)
            
    def _update_plots(self, frame):
        """Update all plots with latest data"""
        # Process queued data
        while not self.data_queue.empty():
            data = self.data_queue.get()
            
            self.timestamps.append(data['timestamp'])
            self.throughput.append(data['throughput'])
            self.gpu_memory.append(data['memory_percent'])
            self.quality_metrics.append(data['quality'])
            
            # Keep only last 100 points
            if len(self.timestamps) > 100:
                self.timestamps.pop(0)
                self.throughput.pop(0)
                self.gpu_memory.pop(0)
                self.quality_metrics.pop(0)
        
        if len(self.timestamps) > 0:
            # Convert timestamps to relative time
            base_time = self.timestamps[0]
            rel_times = [(t - base_time) for t in self.timestamps]
            
            # Update throughput plot
            self.throughput_line.set_data(rel_times, self.throughput)
            self.ax_throughput.relim()
            self.ax_throughput.autoscale_view()
            
            # Update memory plot
            self.memory_line.set_data(rel_times, self.gpu_memory)
            self.ax_memory.relim()
            self.ax_memory.autoscale_view()
            
            # Update quality plot
            self.quality_line.set_data(rel_times, self.quality_metrics)
            self.ax_quality.relim()
            self.ax_quality.autoscale_view()
            
        return [self.throughput_line, self.memory_line, self.quality_line]
        
    def log_processing_stage(self, stage_name, duration_ms, data_size_mb=None):
        """Log completion of a processing stage"""
        timestamp = time.time()
        
        # Add to timing data
        if not hasattr(self, 'stage_timings'):
            self.stage_timings = {}
        
        if stage_name not in self.stage_timings:
            self.stage_timings[stage_name] = []
            
        self.stage_timings[stage_name].append({
            'timestamp': timestamp,
            'duration': duration_ms,
            'data_size': data_size_mb
        })
        
        # Update throughput if data size provided
        if data_size_mb is not None and duration_ms > 0:
            throughput_mbps = (data_size_mb * 1000) / duration_ms
            # Add to queue for real-time update
            self.data_queue.put({
                'timestamp': timestamp,
                'throughput': throughput_mbps,
                'memory_percent': self.gpu_memory[-1] if self.gpu_memory else 0,
                'quality': self.quality_metrics[-1] if self.quality_metrics else 0.9
            })
        
        # Update timing bar chart
        self._update_timing_chart()
        
    def _update_timing_chart(self):
        """Update the processing timing bar chart"""
        if not hasattr(self, 'stage_timings'):
            return
            
        # Calculate average times for each stage
        stage_names = list(self.stage_timings.keys())
        avg_times = []
        
        for stage in stage_names:
            recent_times = [entry['duration'] for entry in self.stage_timings[stage][-10:]]
            avg_times.append(np.mean(recent_times))
        
        # Update bar chart
        self.ax_timing.clear()
        self.ax_timing.bar(stage_names, avg_times, color='skyblue', edgecolor='navy')
        self.ax_timing.set_title('Stage Processing Times (ms)')
        self.ax_timing.set_ylabel('Time (ms)')
        
        # Rotate x-axis labels for readability
        plt.setp(self.ax_timing.get_xticklabels(), rotation=45, ha='right')
        

class ProcessingViewer:
    """
    Interactive viewer showing step-by-step processing results
    
    Allows scientists to see intermediate results and understand
    what each processing stage is doing to the data.
    """
    
    def __init__(self):
        self.processing_stages = []
        self.current_stage = 0
        
        # Setup interactive figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('SuperDARN Processing Pipeline Viewer', fontsize=16)
        
        self._setup_controls()
        self._setup_plots()
        
    def _setup_controls(self):
        """Setup interactive controls"""
        # Add navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.21, 0.02, 0.1, 0.04])
        ax_play = plt.axes([0.32, 0.02, 0.1, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_play = Button(ax_play, 'Play')
        
        self.btn_prev.on_clicked(self._prev_stage)
        self.btn_next.on_clicked(self._next_stage)
        self.btn_play.on_clicked(self._play_stages)
        
        # Stage slider
        ax_slider = plt.axes([0.5, 0.02, 0.3, 0.04])
        self.stage_slider = Slider(ax_slider, 'Stage', 0, 1, valinit=0, valfmt='%d')
        self.stage_slider.on_changed(self._on_stage_change)
        
    def _setup_plots(self):
        """Setup visualization plots"""
        # Raw data view
        self.ax_raw = self.axes[0, 0]
        self.ax_raw.set_title('Raw Data')
        
        # ACF view  
        self.ax_acf = self.axes[0, 1]
        self.ax_acf.set_title('Auto-Correlation Function')
        
        # FitACF results
        self.ax_fit = self.axes[0, 2]
        self.ax_fit.set_title('FitACF Results')
        
        # Range-time plot
        self.ax_range_time = self.axes[1, 0]
        self.ax_range_time.set_title('Range-Time Plot')
        
        # Spectral analysis
        self.ax_spectrum = self.axes[1, 1]
        self.ax_spectrum.set_title('Power Spectrum')
        
        # Quality metrics
        self.ax_metrics = self.axes[1, 2]
        self.ax_metrics.set_title('Quality Metrics')
        
        plt.tight_layout()
        
    def add_processing_stage(self, stage_name, data, metadata=None):
        """Add a processing stage result for visualization"""
        stage_info = {
            'name': stage_name,
            'data': data,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        
        self.processing_stages.append(stage_info)
        
        # Update slider range
        if len(self.processing_stages) > 1:
            self.stage_slider.valmax = len(self.processing_stages) - 1
            self.stage_slider.ax.set_xlim(0, len(self.processing_stages) - 1)
            
        # Auto-advance to new stage
        self.current_stage = len(self.processing_stages) - 1
        self._update_display()
        
    def _update_display(self):
        """Update all plots for current stage"""
        if not self.processing_stages:
            return
            
        current_data = self.processing_stages[self.current_stage]
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            
        self._setup_plots()  # Reset plot titles and labels
        
        # Update stage info
        stage_name = current_data['name']
        self.fig.suptitle(f'Processing Stage {self.current_stage + 1}/{len(self.processing_stages)}: {stage_name}', 
                         fontsize=16)
        
        # Visualize data based on stage type
        self._visualize_stage_data(current_data)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        
    def _visualize_stage_data(self, stage_data):
        """Visualize data for specific processing stage"""
        data = stage_data['data']
        stage_name = stage_data['name'].lower()
        
        # Handle different data types
        if isinstance(data, dict):
            # Multiple data arrays
            if 'rawacf' in data or 'raw' in stage_name:
                self._plot_rawacf_data(data)
            elif 'fitacf' in data or 'fit' in stage_name:
                self._plot_fitacf_data(data)
            elif 'grid' in data or 'grid' in stage_name:
                self._plot_grid_data(data)
        elif hasattr(data, 'shape'):
            # Single array - determine visualization based on shape
            if data.ndim == 2:
                # 2D data - show as image
                self.ax_raw.imshow(data, aspect='auto', cmap='viridis')
                self.ax_raw.set_title(f'{stage_data["name"]} - 2D Data')
            elif data.ndim == 1:
                # 1D data - show as line plot
                self.ax_raw.plot(data)
                self.ax_raw.set_title(f'{stage_data["name"]} - 1D Data')
                
    def _plot_rawacf_data(self, data):
        """Plot raw ACF data"""
        if 'acf' in data:
            acf_data = data['acf']
            # Plot ACF magnitude and phase
            if acf_data.ndim >= 2:
                self.ax_acf.imshow(np.abs(acf_data), aspect='auto', cmap='plasma')
                self.ax_acf.set_title('ACF Magnitude')
                
        if 'power' in data:
            self.ax_raw.plot(data['power'])
            self.ax_raw.set_title('Power Profile')
            
    def _plot_fitacf_data(self, data):
        """Plot FitACF results"""
        if 'velocity' in data:
            vel_data = data['velocity']
            self.ax_fit.imshow(vel_data, aspect='auto', cmap='RdBu_r', 
                              vmin=-1000, vmax=1000)
            self.ax_fit.set_title('Doppler Velocity (m/s)')
            
        if 'power' in data:
            pow_data = data['power']  
            self.ax_raw.imshow(pow_data, aspect='auto', cmap='viridis')
            self.ax_raw.set_title('Backscatter Power (dB)')
            
        if 'width' in data:
            width_data = data['width']
            self.ax_spectrum.imshow(width_data, aspect='auto', cmap='plasma')
            self.ax_spectrum.set_title('Spectral Width (m/s)')
            
    def _plot_grid_data(self, data):
        """Plot gridded data"""
        # Implementation depends on grid data structure
        pass
        
    def _prev_stage(self, event):
        """Go to previous processing stage"""
        if self.current_stage > 0:
            self.current_stage -= 1
            self.stage_slider.set_val(self.current_stage)
            self._update_display()
            
    def _next_stage(self, event):
        """Go to next processing stage"""
        if self.current_stage < len(self.processing_stages) - 1:
            self.current_stage += 1
            self.stage_slider.set_val(self.current_stage)
            self._update_display()
            
    def _play_stages(self, event):
        """Auto-play through processing stages"""
        def play_loop():
            for i in range(len(self.processing_stages)):
                if not self.is_playing:
                    break
                self.current_stage = i
                self.stage_slider.set_val(i)
                self._update_display()
                time.sleep(1.5)
            self.is_playing = False
            self.btn_play.label.set_text('Play')
            
        if not hasattr(self, 'is_playing'):
            self.is_playing = False
            
        if not self.is_playing:
            self.is_playing = True
            self.btn_play.label.set_text('Stop')
            play_thread = threading.Thread(target=play_loop)
            play_thread.daemon = True
            play_thread.start()
        else:
            self.is_playing = False
            self.btn_play.label.set_text('Play')
            
    def _on_stage_change(self, val):
        """Handle stage slider change"""
        self.current_stage = int(val)
        self._update_display()
        
    def save_processing_animation(self, filename='processing_animation.gif'):
        """Save an animated GIF showing the processing pipeline"""
        if len(self.processing_stages) < 2:
            print("Need at least 2 processing stages to create animation")
            return
            
        print(f"Creating processing animation with {len(self.processing_stages)} stages...")
        
        # Create animation frames
        frames = []
        for i in range(len(self.processing_stages)):
            self.current_stage = i
            self._update_display()
            
            # Convert figure to image
            self.fig.canvas.draw()
            frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            
        # Save as GIF using imageio if available
        try:
            import imageio
            imageio.mimsave(filename, frames, duration=2.0)
            print(f"Processing animation saved as {filename}")
        except ImportError:
            print("imageio not available - cannot save animation")
            print("Install with: pip install imageio")
