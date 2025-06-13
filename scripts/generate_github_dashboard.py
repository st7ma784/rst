#!/usr/bin/env python3
"""
SuperDARN RST GitHub Actions Performance Dashboard Generator
==========================================================

This script generates comprehensive performance dashboards from GitHub Actions
test results, integrating with the automated testing workflow.

Features:
- Loads results from GitHub Actions artifacts
- Generates interactive Plotly dashboards
- Creates summary reports for PR comments
- Tracks performance trends over time
- Detects performance regressions
"""

import os
import sys
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
from pathlib import Path
import glob
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceDashboardGenerator:
    """Generator for comprehensive performance dashboards"""
    
    def __init__(self, theme='plotly_white'):
        self.theme = theme
        self.colors = {
            'standard': '#1f77b4',      # Blue
            'optimized': '#2ca02c',     # Green
            'improvement': '#ff7f0e',   # Orange
            'regression': '#d62728',    # Red
            'neutral': '#7f7f7f'        # Gray
        }
    
    def load_results_from_artifacts(self, results_dir):
        """Load performance results from GitHub Actions artifacts"""
        logger.info(f"Loading results from {results_dir}")
        
        results = {}
        
        for build_type in ['standard', 'optimized']:
            artifact_dir = Path(results_dir) / f'performance-results-{build_type}'
            
            if not artifact_dir.exists():
                logger.warning(f"Artifact directory not found: {artifact_dir}")
                continue
            
            # Find the most recent result directory
            result_dirs = [d for d in artifact_dir.iterdir() if d.is_dir()]
            if not result_dirs:
                logger.warning(f"No result directories found in {artifact_dir}")
                continue
            
            latest_result = max(result_dirs, key=lambda d: d.stat().st_mtime)
            logger.info(f"Loading {build_type} results from {latest_result}")
            
            build_results = self._load_build_results(latest_result)
            if build_results:
                results[build_type] = build_results
        
        return results
    
    def _load_build_results(self, result_dir):
        """Load results from a single build directory"""
        build_results = {
            'datasets': {},
            'system_info': {},
            'test_info': {}
        }
        
        # Load test info
        test_info_file = result_dir / 'test_info.txt'
        if test_info_file.exists():
            with open(test_info_file, 'r') as f:
                build_results['test_info'] = f.read()
        
        # Load system info
        system_info_file = result_dir / 'system_info.txt'
        if system_info_file.exists():
            with open(system_info_file, 'r') as f:
                build_results['system_info'] = f.read()
        
        # Load dataset results
        for dataset_dir in result_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                dataset_data = self._load_dataset_results(dataset_dir)
                if dataset_data:
                    build_results['datasets'][dataset_name] = dataset_data
        
        return build_results
    
    def _load_dataset_results(self, dataset_dir):
        """Load results for a single dataset"""
        dataset_data = {
            'files': [],
            'timing': {},
            'memory': {},
            'cpu': {}
        }
        
        # Load total time
        total_time_file = dataset_dir / 'total_time.txt'
        if total_time_file.exists():
            with open(total_time_file, 'r') as f:
                dataset_data['timing']['total_time'] = float(f.read().strip())
        
        # Load file count
        file_count_file = dataset_dir / 'file_count.txt'
        if file_count_file.exists():
            with open(file_count_file, 'r') as f:
                dataset_data['timing']['file_count'] = int(f.read().strip())
        
        # Load individual file timings
        time_files = list(dataset_dir.glob('*_time.csv'))
        for time_file in time_files:
            try:
                with open(time_file, 'r') as f:
                    line = f.read().strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            file_data = {
                                'name': time_file.stem.replace('_time', ''),
                                'time': float(parts[0]),
                                'memory': int(parts[1]),
                                'cpu': float(parts[2].rstrip('%'))
                            }
                            dataset_data['files'].append(file_data)
            except (ValueError, IOError) as e:
                logger.warning(f"Error loading time file {time_file}: {e}")
        
        # Load memory monitoring data
        memory_file = dataset_dir / 'memory.csv'
        if memory_file.exists():
            try:
                memory_data = []
                with open(memory_file, 'r') as f:
                    for line in f:
                        if ',' in line:
                            timestamp, memory_mb = line.strip().split(',')
                            memory_data.append({
                                'timestamp': float(timestamp),
                                'memory_mb': int(memory_mb)
                            })
                dataset_data['memory']['timeline'] = memory_data
                if memory_data:
                    dataset_data['memory']['peak'] = max(d['memory_mb'] for d in memory_data)
                    dataset_data['memory']['average'] = sum(d['memory_mb'] for d in memory_data) / len(memory_data)
            except (ValueError, IOError) as e:
                logger.warning(f"Error loading memory data: {e}")
        
        # Load CPU monitoring data
        cpu_file = dataset_dir / 'cpu.csv'
        if cpu_file.exists():
            try:
                cpu_data = []
                with open(cpu_file, 'r') as f:
                    for line in f:
                        if ',' in line:
                            timestamp, load_avg = line.strip().split(',')
                            cpu_data.append({
                                'timestamp': float(timestamp),
                                'load_avg': float(load_avg)
                            })
                dataset_data['cpu']['timeline'] = cpu_data
                if cpu_data:
                    dataset_data['cpu']['peak'] = max(d['load_avg'] for d in cpu_data)
                    dataset_data['cpu']['average'] = sum(d['load_avg'] for d in cpu_data) / len(cpu_data)
            except (ValueError, IOError) as e:
                logger.warning(f"Error loading CPU data: {e}")
        
        return dataset_data if dataset_data['files'] or dataset_data['timing'] else None
    
    def create_comprehensive_dashboard(self, results):
        """Create comprehensive performance dashboard"""
        logger.info("Creating comprehensive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Processing Time Comparison',
                'Memory Usage Comparison',
                'CPU Load Comparison',
                'File Processing Rate',
                'Performance Improvement Summary',
                'Resource Efficiency'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        if 'standard' in results and 'optimized' in results:
            datasets, comparison_data = self._prepare_comparison_data(results)
            
            # Plot 1: Processing Time Comparison
            self._add_timing_comparison(fig, datasets, comparison_data, row=1, col=1)
            
            # Plot 2: Memory Usage Comparison  
            self._add_memory_comparison(fig, datasets, comparison_data, row=1, col=2)
            
            # Plot 3: CPU Load Comparison
            self._add_cpu_comparison(fig, datasets, comparison_data, row=2, col=1)
            
            # Plot 4: File Processing Rate
            self._add_processing_rate(fig, datasets, comparison_data, row=2, col=2)
            
            # Plot 5: Performance Improvement Summary
            self._add_improvement_summary(fig, datasets, comparison_data, row=3, col=1)
            
            # Plot 6: Resource Efficiency
            self._add_resource_efficiency(fig, datasets, comparison_data, row=3, col=2)
        else:
            # Add placeholder text if no comparison data
            fig.add_annotation(
                text="No comparison data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"SuperDARN RST Performance Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                'x': 0.5,
                'font': {'size': 20}
            },
            height=1200,
            showlegend=True,
            template=self.theme,
            font=dict(size=12)
        )
        
        return fig
    
    def _prepare_comparison_data(self, results):
        """Prepare data for comparison plots"""
        datasets = []
        comparison_data = {
            'times': {'standard': [], 'optimized': []},
            'memory': {'standard': [], 'optimized': []},
            'cpu': {'standard': [], 'optimized': []},
            'files': {'standard': [], 'optimized': []},
            'rates': {'standard': [], 'optimized': []}
        }
        
        for dataset_name in results['standard']['datasets'].keys():
            if dataset_name in results['optimized']['datasets']:
                datasets.append(dataset_name)
                
                std_data = results['standard']['datasets'][dataset_name]
                opt_data = results['optimized']['datasets'][dataset_name]
                
                # Times
                std_time = std_data['timing'].get('total_time', 0)
                opt_time = opt_data['timing'].get('total_time', 0)
                comparison_data['times']['standard'].append(std_time)
                comparison_data['times']['optimized'].append(opt_time)
                
                # Memory
                std_mem = std_data['memory'].get('peak', 0)
                opt_mem = opt_data['memory'].get('peak', 0)
                comparison_data['memory']['standard'].append(std_mem)
                comparison_data['memory']['optimized'].append(opt_mem)
                
                # CPU
                std_cpu = std_data['cpu'].get('average', 0)
                opt_cpu = opt_data['cpu'].get('average', 0)
                comparison_data['cpu']['standard'].append(std_cpu)
                comparison_data['cpu']['optimized'].append(opt_cpu)
                
                # File counts and rates
                std_files = std_data['timing'].get('file_count', 1)
                opt_files = opt_data['timing'].get('file_count', 1)
                comparison_data['files']['standard'].append(std_files)
                comparison_data['files']['optimized'].append(opt_files)
                
                std_rate = std_files / max(std_time, 0.001)
                opt_rate = opt_files / max(opt_time, 0.001)
                comparison_data['rates']['standard'].append(std_rate)
                comparison_data['rates']['optimized'].append(opt_rate)
        
        return datasets, comparison_data
    
    def _add_timing_comparison(self, fig, datasets, data, row, col):
        """Add timing comparison plot"""
        fig.add_trace(
            go.Bar(
                name='Standard',
                x=datasets,
                y=data['times']['standard'],
                marker_color=self.colors['standard'],
                text=[f"{t:.1f}s" for t in data['times']['standard']],
                textposition='auto'
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(
                name='Optimized',
                x=datasets,
                y=data['times']['optimized'],
                marker_color=self.colors['optimized'],
                text=[f"{t:.1f}s" for t in data['times']['optimized']],
                textposition='auto'
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Time (seconds)", row=row, col=col)
    
    def _add_memory_comparison(self, fig, datasets, data, row, col):
        """Add memory usage comparison plot"""
        fig.add_trace(
            go.Bar(
                name='Standard Memory',
                x=datasets,
                y=data['memory']['standard'],
                marker_color=self.colors['standard'],
                text=[f"{m:.0f}MB" for m in data['memory']['standard']],
                textposition='auto',
                showlegend=False
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(
                name='Optimized Memory',
                x=datasets,
                y=data['memory']['optimized'],
                marker_color=self.colors['optimized'],
                text=[f"{m:.0f}MB" for m in data['memory']['optimized']],
                textposition='auto',
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Peak Memory (MB)", row=row, col=col)
    
    def _add_cpu_comparison(self, fig, datasets, data, row, col):
        """Add CPU load comparison plot"""
        fig.add_trace(
            go.Scatter(
                name='Standard CPU',
                x=datasets,
                y=data['cpu']['standard'],
                mode='lines+markers',
                line=dict(color=self.colors['standard']),
                marker=dict(size=8),
                showlegend=False
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                name='Optimized CPU',
                x=datasets,
                y=data['cpu']['optimized'],
                mode='lines+markers',
                line=dict(color=self.colors['optimized']),
                marker=dict(size=8),
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Average Load", row=row, col=col)
    
    def _add_processing_rate(self, fig, datasets, data, row, col):
        """Add file processing rate plot"""
        fig.add_trace(
            go.Scatter(
                name='Standard Rate',
                x=datasets,
                y=data['rates']['standard'],
                mode='lines+markers',
                line=dict(color=self.colors['standard'], dash='dash'),
                marker=dict(size=10),
                showlegend=False
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                name='Optimized Rate',
                x=datasets,
                y=data['rates']['optimized'],
                mode='lines+markers',
                line=dict(color=self.colors['optimized'], dash='dash'),
                marker=dict(size=10),
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Files/Second", row=row, col=col)
    
    def _add_improvement_summary(self, fig, datasets, data, row, col):
        """Add performance improvement summary"""
        improvements = []
        colors = []
        
        for i in range(len(datasets)):
            if data['times']['standard'][i] > 0:
                improvement = ((data['times']['standard'][i] - data['times']['optimized'][i]) / 
                             data['times']['standard'][i]) * 100
                improvements.append(improvement)
                colors.append(self.colors['improvement'] if improvement > 0 else self.colors['regression'])
            else:
                improvements.append(0)
                colors.append(self.colors['neutral'])
        
        fig.add_trace(
            go.Bar(
                name='Time Improvement %',
                x=datasets,
                y=improvements,
                marker_color=colors,
                text=[f"{imp:+.1f}%" for imp in improvements],
                textposition='auto',
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Improvement %", row=row, col=col)
    
    def _add_resource_efficiency(self, fig, datasets, data, row, col):
        """Add resource efficiency comparison"""
        efficiency_std = []
        efficiency_opt = []
        
        for i in range(len(datasets)):
            # Calculate efficiency as files processed per MB of memory used
            std_eff = data['files']['standard'][i] / max(data['memory']['standard'][i], 1)
            opt_eff = data['files']['optimized'][i] / max(data['memory']['optimized'][i], 1)
            
            efficiency_std.append(std_eff)
            efficiency_opt.append(opt_eff)
        
        fig.add_trace(
            go.Bar(
                name='Standard Efficiency',
                x=datasets,
                y=efficiency_std,
                marker_color=self.colors['standard'],
                text=[f"{e:.2f}" for e in efficiency_std],
                textposition='auto',
                showlegend=False
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Bar(
                name='Optimized Efficiency',
                x=datasets,
                y=efficiency_opt,
                marker_color=self.colors['optimized'],
                text=[f"{e:.2f}" for e in efficiency_opt],
                textposition='auto',
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Files/MB Memory", row=row, col=col)
    
    def generate_summary_report(self, results):
        """Generate summary report for GitHub PR comments"""
        logger.info("Generating summary report")
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'test_run_id': os.environ.get('GITHUB_SHA', 'unknown')[:8],
            'commit': os.environ.get('GITHUB_SHA', 'unknown'),
            'branch': os.environ.get('GITHUB_REF_NAME', 'unknown'),
            'datasets_tested': [],
            'overall_performance': {},
            'regression_detected': False,
            'quality_metrics': {}
        }
        
        if 'standard' in results and 'optimized' in results:
            total_std_time = 0
            total_opt_time = 0
            total_files = 0
            total_std_memory = 0
            total_opt_memory = 0
            
            for dataset_name in results['standard']['datasets'].keys():
                if dataset_name in results['optimized']['datasets']:
                    summary['datasets_tested'].append(dataset_name)
                    
                    std_data = results['standard']['datasets'][dataset_name]
                    opt_data = results['optimized']['datasets'][dataset_name]
                    
                    std_time = std_data['timing'].get('total_time', 0)
                    opt_time = opt_data['timing'].get('total_time', 0)
                    
                    total_std_time += std_time
                    total_opt_time += opt_time
                    total_files += std_data['timing'].get('file_count', 0)
                    total_std_memory += std_data['memory'].get('peak', 0)
                    total_opt_memory += opt_data['memory'].get('peak', 0)
            
            if total_std_time > 0:
                time_improvement = ((total_std_time - total_opt_time) / total_std_time) * 100
                memory_improvement = ((total_std_memory - total_opt_memory) / max(total_std_memory, 1)) * 100
                
                summary['overall_performance'] = {
                    'total_standard_time': total_std_time,
                    'total_optimized_time': total_opt_time,
                    'time_improvement_percent': time_improvement,
                    'total_files_processed': total_files,
                    'speedup_factor': total_std_time / max(total_opt_time, 0.001),
                    'memory_improvement_percent': memory_improvement,
                    'total_standard_memory': total_std_memory,
                    'total_optimized_memory': total_opt_memory
                }
                
                # Quality metrics
                summary['quality_metrics'] = {
                    'processing_rate_std': total_files / max(total_std_time, 0.001),
                    'processing_rate_opt': total_files / max(total_opt_time, 0.001),
                    'memory_efficiency_std': total_files / max(total_std_memory, 1),
                    'memory_efficiency_opt': total_files / max(total_opt_memory, 1)
                }
                
                # Check for regression (performance degradation > 5%)
                if time_improvement < -5.0:
                    summary['regression_detected'] = True
                    summary['regression_details'] = {
                        'type': 'time_regression',
                        'severity': 'high' if time_improvement < -10.0 else 'medium',
                        'degradation_percent': abs(time_improvement)
                    }
        
        return summary
    
    def save_dashboard(self, fig, output_path, include_plotlyjs='cdn'):
        """Save dashboard to HTML file"""
        logger.info(f"Saving dashboard to {output_path}")
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with custom configuration
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'superdarn_rst_performance',
                'height': 1200,
                'width': 1600,
                'scale': 2
            }
        }
        
        pyo.plot(
            fig,
            filename=str(output_path),
            auto_open=False,
            include_plotlyjs=include_plotlyjs,
            config=config
        )
        
        return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate SuperDARN RST performance dashboard from GitHub Actions results'
    )
    parser.add_argument(
        '--results-dir', '-r',
        required=True,
        help='Directory containing performance test results'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='dashboard',
        help='Output directory for generated dashboard (default: dashboard)'
    )
    parser.add_argument(
        '--theme',
        choices=['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn'],
        default='plotly_white',
        help='Dashboard theme (default: plotly_white)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize dashboard generator
    generator = PerformanceDashboardGenerator(theme=args.theme)
    
    try:
        # Load results
        results = generator.load_results_from_artifacts(args.results_dir)
        
        if not results:
            logger.error("No performance results found")
            # Create placeholder dashboard
            fig = go.Figure()
            fig.add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=24)
            )
            fig.update_layout(
                title="SuperDARN RST Performance Dashboard - No Data",
                height=600
            )
        else:
            # Generate dashboard
            fig = generator.create_comprehensive_dashboard(results)
        
        # Save dashboard
        output_path = Path(args.output_dir) / 'performance_dashboard.html'
        generator.save_dashboard(fig, output_path)
        
        # Generate summary report
        summary = generator.generate_summary_report(results)
        summary_path = Path(args.output_dir) / 'performance_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dashboard generated: {output_path}")
        logger.info(f"Summary report: {summary_path}")
        
        if summary.get('regression_detected'):
            logger.warning("Performance regression detected!")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
