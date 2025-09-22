"""
Performance Monitoring and Visualization
=======================================

GPU performance monitoring, profiling, and optimization visualization
for SuperDARN processing pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import threading
from queue import Queue
from collections import deque, defaultdict
import psutil
import gc

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from ..core.backends import get_backend


class PerformanceDashboard:
    """
    Real-time performance monitoring dashboard
    
    Tracks:
    - GPU utilization and memory usage
    - Processing throughput and latency
    - CPU usage and system resources
    - Pipeline bottlenecks and optimization opportunities
    """
    
    def __init__(self, max_history=1000, update_interval=0.5):
        self.max_history = max_history
        self.update_interval = update_interval
        self.is_running = False
        
        # Performance metrics storage
        self.metrics = {
            'timestamp': deque(maxlen=max_history),
            'gpu_memory_used': deque(maxlen=max_history),
            'gpu_memory_total': deque(maxlen=max_history),
            'gpu_utilization': deque(maxlen=max_history),
            'cpu_percent': deque(maxlen=max_history),
            'ram_percent': deque(maxlen=max_history),
            'throughput_mbps': deque(maxlen=max_history),
            'processing_latency': deque(maxlen=max_history),
        }
        
        # Processing stage timing
        self.stage_timings = defaultdict(lambda: deque(maxlen=100))
        self.stage_data_sizes = defaultdict(lambda: deque(maxlen=100))
        
        # Setup dashboard figure
        self.fig, self.axes = plt.subplots(3, 3, figsize=(20, 15))
        self.fig.suptitle('SuperDARN GPU Processing Performance Dashboard', fontsize=16)
        
        self._setup_plots()
        self._initialize_gpu_monitoring()
        
    def _setup_plots(self):
        """Initialize all dashboard plots"""
        # GPU Memory Usage (top-left)
        self.ax_gpu_mem = self.axes[0, 0]
        self.ax_gpu_mem.set_title('GPU Memory Usage')
        self.ax_gpu_mem.set_ylabel('Memory (GB)')
        self.gpu_mem_line, = self.ax_gpu_mem.plot([], [], 'b-', label='Used')
        self.gpu_mem_fill = self.ax_gpu_mem.fill_between([], [], [], alpha=0.3, color='blue')
        self.ax_gpu_mem.legend()
        
        # GPU Utilization (top-center)
        self.ax_gpu_util = self.axes[0, 1]
        self.ax_gpu_util.set_title('GPU Utilization')
        self.ax_gpu_util.set_ylabel('Utilization (%)')
        self.ax_gpu_util.set_ylim(0, 100)
        self.gpu_util_line, = self.ax_gpu_util.plot([], [], 'g-', linewidth=2)
        
        # Throughput (top-right)
        self.ax_throughput = self.axes[0, 2]
        self.ax_throughput.set_title('Data Throughput')
        self.ax_throughput.set_ylabel('Throughput (MB/s)')
        self.throughput_line, = self.ax_throughput.plot([], [], 'r-', linewidth=2)
        
        # CPU and RAM usage (middle-left)
        self.ax_system = self.axes[1, 0]
        self.ax_system.set_title('System Resources')
        self.ax_system.set_ylabel('Usage (%)')
        self.ax_system.set_ylim(0, 100)
        self.cpu_line, = self.ax_system.plot([], [], 'orange', label='CPU')
        self.ram_line, = self.ax_system.plot([], [], 'purple', label='RAM')
        self.ax_system.legend()
        
        # Processing Latency (middle-center)
        self.ax_latency = self.axes[1, 1]
        self.ax_latency.set_title('Processing Latency')
        self.ax_latency.set_ylabel('Latency (ms)')
        self.latency_line, = self.ax_latency.plot([], [], 'brown', linewidth=2)
        
        # Stage Timing Bar Chart (middle-right)
        self.ax_stages = self.axes[1, 2]
        self.ax_stages.set_title('Processing Stage Times')
        self.ax_stages.set_ylabel('Time (ms)')
        
        # GPU Memory Breakdown Pie Chart (bottom-left)
        self.ax_mem_pie = self.axes[2, 0]
        self.ax_mem_pie.set_title('GPU Memory Breakdown')
        
        # Throughput vs Batch Size (bottom-center)
        self.ax_batch = self.axes[2, 1]
        self.ax_batch.set_title('Throughput vs Batch Size')
        self.ax_batch.set_xlabel('Batch Size')
        self.ax_batch.set_ylabel('Throughput (MB/s)')
        
        # Performance Heatmap (bottom-right)
        self.ax_heatmap = self.axes[2, 2]
        self.ax_heatmap.set_title('Performance Heatmap')
        
        plt.tight_layout()
        
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring if available"""
        self.gpu_info = {}
        if GPU_AVAILABLE and cp.cuda.is_available():
            try:
                # Get GPU device info
                device = cp.cuda.Device(0)
                self.gpu_info['name'] = device.name
                self.gpu_info['total_memory'] = device.mem_info[1]
                self.gpu_info['compute_capability'] = device.compute_capability
                
                print(f"GPU monitoring initialized: {self.gpu_info['name']}")
                print(f"Total GPU memory: {self.gpu_info['total_memory'] / (1024**3):.1f} GB")
                
            except Exception as e:
                print(f"GPU monitoring initialization failed: {e}")
                
    def start_monitoring(self):
        """Start performance monitoring"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self._update_dashboard,
                                     interval=int(self.update_interval * 1000),
                                     blit=False, cache_frame_data=False)
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_running = False
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
            
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            timestamp = time.time()
            
            # Collect system metrics
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            
            # Collect GPU metrics
            gpu_memory_used = 0
            gpu_memory_total = 0
            gpu_utilization = 0
            
            if GPU_AVAILABLE and cp.cuda.is_available():
                try:
                    mempool = cp.get_default_memory_pool()
                    gpu_memory_used = mempool.used_bytes()
                    gpu_memory_total = self.gpu_info.get('total_memory', 0)
                    
                    # Estimate GPU utilization (simplified)
                    gpu_utilization = min(100, (gpu_memory_used / gpu_memory_total) * 100)
                    
                except Exception as e:
                    print(f"GPU monitoring error: {e}")
            
            # Store metrics
            self.metrics['timestamp'].append(timestamp)
            self.metrics['gpu_memory_used'].append(gpu_memory_used / (1024**3))  # GB
            self.metrics['gpu_memory_total'].append(gpu_memory_total / (1024**3))  # GB
            self.metrics['gpu_utilization'].append(gpu_utilization)
            self.metrics['cpu_percent'].append(cpu_percent)
            self.metrics['ram_percent'].append(ram_percent)
            
            # Placeholder for throughput and latency (would be updated by processing stages)
            self.metrics['throughput_mbps'].append(
                self.metrics['throughput_mbps'][-1] if self.metrics['throughput_mbps'] else 0)
            self.metrics['processing_latency'].append(
                self.metrics['processing_latency'][-1] if self.metrics['processing_latency'] else 0)
            
            time.sleep(self.update_interval)
            
    def _update_dashboard(self, frame):
        """Update dashboard plots with latest data"""
        if not self.metrics['timestamp']:
            return
        
        # Convert timestamps to relative time (seconds)
        timestamps = list(self.metrics['timestamp'])
        if len(timestamps) > 1:
            base_time = timestamps[0]
            rel_times = [(t - base_time) for t in timestamps]
        else:
            rel_times = [0]
        
        # Update GPU memory plot
        if self.metrics['gpu_memory_used']:
            gpu_mem_used = list(self.metrics['gpu_memory_used'])
            gpu_mem_total = list(self.metrics['gpu_memory_total'])
            
            self.gpu_mem_line.set_data(rel_times, gpu_mem_used)
            self.ax_gpu_mem.relim()
            self.ax_gpu_mem.autoscale_view()
            
            # Show total memory as horizontal line
            if gpu_mem_total and gpu_mem_total[0] > 0:
                self.ax_gpu_mem.axhline(y=gpu_mem_total[0], color='red', 
                                       linestyle='--', alpha=0.7, label='Total')
        
        # Update GPU utilization
        if self.metrics['gpu_utilization']:
            self.gpu_util_line.set_data(rel_times, list(self.metrics['gpu_utilization']))
            self.ax_gpu_util.relim()
            self.ax_gpu_util.autoscale_view()
        
        # Update throughput
        if self.metrics['throughput_mbps']:
            self.throughput_line.set_data(rel_times, list(self.metrics['throughput_mbps']))
            self.ax_throughput.relim()
            self.ax_throughput.autoscale_view()
        
        # Update system resources
        if self.metrics['cpu_percent'] and self.metrics['ram_percent']:
            self.cpu_line.set_data(rel_times, list(self.metrics['cpu_percent']))
            self.ram_line.set_data(rel_times, list(self.metrics['ram_percent']))
            self.ax_system.relim()
            self.ax_system.autoscale_view()
        
        # Update latency
        if self.metrics['processing_latency']:
            self.latency_line.set_data(rel_times, list(self.metrics['processing_latency']))
            self.ax_latency.relim()
            self.ax_latency.autoscale_view()
        
        # Update stage timing bar chart
        self._update_stage_chart()
        
        # Update memory pie chart
        self._update_memory_pie()
        
    def _update_stage_chart(self):
        """Update processing stage timing bar chart"""
        if not self.stage_timings:
            return
            
        self.ax_stages.clear()
        self.ax_stages.set_title('Processing Stage Times')
        self.ax_stages.set_ylabel('Time (ms)')
        
        stage_names = list(self.stage_timings.keys())
        avg_times = []
        
        for stage in stage_names:
            if self.stage_timings[stage]:
                avg_times.append(np.mean(list(self.stage_timings[stage])[-10:]))
            else:
                avg_times.append(0)
        
        if stage_names and avg_times:
            bars = self.ax_stages.bar(stage_names, avg_times, color='skyblue', 
                                    edgecolor='navy', alpha=0.7)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, avg_times):
                if time_val > 0:
                    self.ax_stages.text(bar.get_x() + bar.get_width()/2, 
                                      bar.get_height() + max(avg_times) * 0.01,
                                      f'{time_val:.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.setp(self.ax_stages.get_xticklabels(), rotation=45, ha='right')
    
    def _update_memory_pie(self):
        """Update GPU memory breakdown pie chart"""
        if not GPU_AVAILABLE or not self.metrics['gpu_memory_used']:
            return
            
        self.ax_mem_pie.clear()
        self.ax_mem_pie.set_title('GPU Memory Breakdown')
        
        if self.metrics['gpu_memory_used'] and self.metrics['gpu_memory_total']:
            used = self.metrics['gpu_memory_used'][-1]
            total = self.metrics['gpu_memory_total'][-1]
            free = max(0, total - used)
            
            if total > 0:
                sizes = [used, free]
                labels = [f'Used ({used:.1f} GB)', f'Free ({free:.1f} GB)']
                colors = ['#ff9999', '#66b3ff']
                
                self.ax_mem_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                   startangle=90)
    
    def log_stage_performance(self, stage_name, duration_ms, data_size_mb=None):
        """Log performance data for a processing stage"""
        timestamp = time.time()
        
        # Store timing data
        self.stage_timings[stage_name].append(duration_ms)
        
        if data_size_mb is not None:
            self.stage_data_sizes[stage_name].append(data_size_mb)
            
            # Calculate throughput
            if duration_ms > 0:
                throughput = (data_size_mb * 1000) / duration_ms  # MB/s
                self.metrics['throughput_mbps'].append(throughput)
        
        # Update latency metric
        self.metrics['processing_latency'].append(duration_ms)
        
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.metrics['timestamp']:
            print("No performance data available")
            return
        
        print("SuperDARN GPU Performance Report")
        print("=" * 50)
        print(f"Monitoring period: {len(self.metrics['timestamp'])} samples")
        print(f"Update interval: {self.update_interval} seconds")
        print()
        
        # GPU metrics
        if GPU_AVAILABLE and self.metrics['gpu_memory_used']:
            avg_gpu_mem = np.mean(list(self.metrics['gpu_memory_used']))
            max_gpu_mem = np.max(list(self.metrics['gpu_memory_used']))
            avg_gpu_util = np.mean(list(self.metrics['gpu_utilization']))
            
            print(f"GPU Performance:")
            print(f"  Average memory usage: {avg_gpu_mem:.2f} GB")
            print(f"  Peak memory usage: {max_gpu_mem:.2f} GB")
            print(f"  Average utilization: {avg_gpu_util:.1f}%")
            print()
        
        # System metrics
        if self.metrics['cpu_percent']:
            avg_cpu = np.mean(list(self.metrics['cpu_percent']))
            avg_ram = np.mean(list(self.metrics['ram_percent']))
            
            print(f"System Performance:")
            print(f"  Average CPU usage: {avg_cpu:.1f}%")
            print(f"  Average RAM usage: {avg_ram:.1f}%")
            print()
        
        # Processing stage performance
        if self.stage_timings:
            print(f"Processing Stage Performance:")
            for stage_name, timings in self.stage_timings.items():
                if timings:
                    avg_time = np.mean(list(timings))
                    std_time = np.std(list(timings))
                    print(f"  {stage_name}: {avg_time:.1f} ± {std_time:.1f} ms")
            print()
        
        # Throughput analysis
        if self.metrics['throughput_mbps']:
            throughputs = [t for t in self.metrics['throughput_mbps'] if t > 0]
            if throughputs:
                avg_throughput = np.mean(throughputs)
                max_throughput = np.max(throughputs)
                print(f"Throughput Analysis:")
                print(f"  Average throughput: {avg_throughput:.1f} MB/s")
                print(f"  Peak throughput: {max_throughput:.1f} MB/s")
                print()


class GPUMonitor:
    """
    Standalone GPU monitoring utility
    """
    
    def __init__(self):
        self.monitoring = False
        
    def get_gpu_info(self):
        """Get detailed GPU information"""
        if not GPU_AVAILABLE:
            return {"status": "GPU not available"}
        
        info = {}
        try:
            device = cp.cuda.Device(0)
            info['name'] = device.name
            info['compute_capability'] = device.compute_capability
            info['total_memory'] = device.mem_info[1]
            info['free_memory'] = device.mem_info[0]
            info['used_memory'] = info['total_memory'] - info['free_memory']
            info['memory_utilization'] = (info['used_memory'] / info['total_memory']) * 100
            
            # Get memory pool info
            mempool = cp.get_default_memory_pool()
            info['pool_used'] = mempool.used_bytes()
            info['pool_total'] = mempool.total_bytes()
            
        except Exception as e:
            info['error'] = str(e)
            
        return info
    
    def print_gpu_status(self):
        """Print current GPU status"""
        info = self.get_gpu_info()
        
        if 'error' in info:
            print(f"GPU monitoring error: {info['error']}")
            return
        
        if 'status' in info:
            print(info['status'])
            return
        
        print("GPU Status:")
        print(f"  Device: {info['name']}")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Total Memory: {info['total_memory'] / (1024**3):.2f} GB")
        print(f"  Used Memory: {info['used_memory'] / (1024**3):.2f} GB")
        print(f"  Free Memory: {info['free_memory'] / (1024**3):.2f} GB")
        print(f"  Utilization: {info['memory_utilization']:.1f}%")
        print()
        print("Memory Pool:")
        print(f"  Pool Used: {info['pool_used'] / (1024**3):.2f} GB")
        print(f"  Pool Total: {info['pool_total'] / (1024**3):.2f} GB")


def plot_processing_times(timing_data, title="Processing Performance Analysis"):
    """
    Plot processing time analysis from benchmark data
    
    Parameters:
    -----------
    timing_data : dict
        Dictionary with stage names as keys and timing lists as values
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=14)
    
    # Bar chart of average times
    stage_names = list(timing_data.keys())
    avg_times = [np.mean(times) for times in timing_data.values()]
    std_times = [np.std(times) for times in timing_data.values()]
    
    bars = ax1.bar(stage_names, avg_times, yerr=std_times, 
                   capsize=5, color='skyblue', edgecolor='navy')
    ax1.set_title('Average Processing Times')
    ax1.set_ylabel('Time (ms)')
    
    # Add value labels
    for bar, avg_time in zip(bars, avg_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times) * 0.01,
                f'{avg_time:.1f}', ha='center', va='bottom')
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Box plot for distribution analysis
    timing_lists = list(timing_data.values())
    ax2.boxplot(timing_lists, labels=stage_names)
    ax2.set_title('Processing Time Distributions')
    ax2.set_ylabel('Time (ms)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_memory_usage(memory_data, title="Memory Usage Analysis"):
    """
    Plot memory usage over time
    
    Parameters:
    -----------
    memory_data : dict
        Dictionary containing memory usage data
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'timestamp' in memory_data and 'gpu_memory' in memory_data:
        timestamps = memory_data['timestamp']
        gpu_memory = memory_data['gpu_memory']
        
        ax.plot(timestamps, gpu_memory, 'b-', linewidth=2, label='GPU Memory')
        
        if 'cpu_memory' in memory_data:
            ax.plot(timestamps, memory_data['cpu_memory'], 'r-', 
                   linewidth=2, label='CPU Memory')
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Memory Usage (GB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
