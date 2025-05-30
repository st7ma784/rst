#!/usr/bin/env python3
"""
SuperDARN Comprehensive Performance Dashboard Generator
Creates interactive HTML dashboard for all component test results
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import glob

def load_test_results(results_dir):
    """Load all test results from JSON files"""
    results = {
        'libraries': {},
        'binaries': {},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_components': 0,
            'successful_builds': 0,
            'failed_builds': 0
        }
    }
    
    # Load library results
    lib_files = glob.glob(os.path.join(results_dir, 'libs', '*.json'))
    for lib_file in lib_files:
        try:
            with open(lib_file, 'r') as f:
                lib_data = json.load(f)
                results['libraries'].update(lib_data.get('libraries', {}))
        except Exception as e:
            print(f"Warning: Could not load {lib_file}: {e}")
    
    # Load binary results
    bin_files = glob.glob(os.path.join(results_dir, 'bins', '*.json'))
    for bin_file in bin_files:
        try:
            with open(bin_file, 'r') as f:
                bin_data = json.load(f)
                results['binaries'].update(bin_data.get('binaries', {}))
        except Exception as e:
            print(f"Warning: Could not load {bin_file}: {e}")
    
    # Calculate metadata
    total_libs = len(results['libraries'])
    total_bins = len(results['binaries'])
    results['metadata']['total_components'] = total_libs + total_bins
    
    # Count successful builds
    for lib_name, lib_data in results['libraries'].items():
        if lib_data.get('original', {}).get('build_status') == 'success':
            results['metadata']['successful_builds'] += 1
        else:
            results['metadata']['failed_builds'] += 1
    
    for bin_name, bin_data in results['binaries'].items():
        if bin_data.get('original', {}).get('build_status') == 'success':
            results['metadata']['successful_builds'] += 1
        else:
            results['metadata']['failed_builds'] += 1
    
    return results

def calculate_optimization_gains(component_data):
    """Calculate optimization gains relative to original version"""
    gains = {}
    original = component_data.get('original', {})
    optimized = component_data.get('optimized', {})
    
    if not original or original.get('build_status') != 'success':
        return gains
    
    original_build_time = original.get('build_time', 0)
    original_size = original.get('library_size', 0) or original.get('binary_size', 0)
    
    for opt_level, opt_data in optimized.items():
        if opt_data.get('build_status') != 'success':
            continue
            
        opt_build_time = opt_data.get('build_time', 0)
        opt_size = opt_data.get('library_size', 0) or opt_data.get('binary_size', 0)
        
        gains[opt_level] = {
            'build_time_ratio': opt_build_time / original_build_time if original_build_time > 0 else 1.0,
            'size_ratio': opt_size / original_size if original_size > 0 else 1.0,
            'build_time_improvement': ((original_build_time - opt_build_time) / original_build_time * 100) if original_build_time > 0 else 0,
            'size_change': ((opt_size - original_size) / original_size * 100) if original_size > 0 else 0
        }
    
    return gains

def generate_component_table(components, component_type):
    """Generate HTML table for component results"""
    html = f'''
    <div class="metric-card">
        <div class="metric-title">📚 {component_type.title()} Performance Results</div>
        <div class="table-responsive">
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Build Status</th>
                        <th>Original Build Time</th>
                        <th>Best Optimization</th>
                        <th>Build Time Improvement</th>
                        <th>Size Change</th>
                    </tr>
                </thead>
                <tbody>
    '''
    
    for comp_name, comp_data in components.items():
        original = comp_data.get('original', {})
        build_status = original.get('build_status', 'unknown')
        build_time = original.get('build_time', 0)
        
        # Find best optimization
        gains = calculate_optimization_gains(comp_data)
        best_opt = None
        best_improvement = 0
        
        for opt_level, opt_gains in gains.items():
            improvement = opt_gains['build_time_improvement']
            if improvement > best_improvement:
                best_improvement = improvement
                best_opt = opt_level
        
        # Status styling
        status_class = 'status-good' if build_status == 'success' else 'status-bad'
        improvement_class = 'status-good' if best_improvement > 0 else 'status-warning'
        
        size_change = gains.get(best_opt, {}).get('size_change', 0) if best_opt else 0
        size_class = 'status-good' if size_change < 0 else 'status-warning' if size_change > 0 else ''
        
        html += f'''
                    <tr>
                        <td><strong>{comp_name}</strong></td>
                        <td><span class="{status_class}">{build_status}</span></td>
                        <td>{build_time:.3f}s</td>
                        <td>{best_opt or 'N/A'}</td>
                        <td><span class="{improvement_class}">{best_improvement:.1f}%</span></td>
                        <td><span class="{size_class}">{size_change:+.1f}%</span></td>
                    </tr>
        '''
    
    html += '''
                </tbody>
            </table>
        </div>
    </div>
    '''
    
    return html

def generate_optimization_comparison_chart(results):
    """Generate optimization comparison data for visualization"""
    optimization_data = {
        'O2': {'components': [], 'improvements': []},
        'O3': {'components': [], 'improvements': []},
        'Ofast': {'components': [], 'improvements': []}
    }
    
    all_components = {**results['libraries'], **results['binaries']}
    
    for comp_name, comp_data in all_components.items():
        gains = calculate_optimization_gains(comp_data)
        
        for opt_level in ['O2', 'O3', 'Ofast']:
            if opt_level in gains:
                improvement = gains[opt_level]['build_time_improvement']
                optimization_data[opt_level]['components'].append(comp_name)
                optimization_data[opt_level]['improvements'].append(improvement)
    
    return optimization_data

def generate_summary_stats(results):
    """Generate summary statistics"""
    total_components = results['metadata']['total_components']
    successful_builds = results['metadata']['successful_builds']
    failed_builds = results['metadata']['failed_builds']
    
    all_components = {**results['libraries'], **results['binaries']}
    
    # Calculate average improvements
    total_improvements = {'O2': [], 'O3': [], 'Ofast': []}
    
    for comp_data in all_components.values():
        gains = calculate_optimization_gains(comp_data)
        for opt_level, opt_gains in gains.items():
            if opt_level in total_improvements:
                total_improvements[opt_level].append(opt_gains['build_time_improvement'])
    
    avg_improvements = {}
    for opt_level, improvements in total_improvements.items():
        avg_improvements[opt_level] = sum(improvements) / len(improvements) if improvements else 0
    
    return {
        'total_components': total_components,
        'successful_builds': successful_builds,
        'failed_builds': failed_builds,
        'success_rate': (successful_builds / total_components * 100) if total_components > 0 else 0,
        'avg_improvements': avg_improvements,
        'best_optimization': max(avg_improvements.items(), key=lambda x: x[1]) if avg_improvements else ('N/A', 0)
    }

def generate_html_dashboard(results, output_file):
    """Generate complete HTML dashboard"""
    
    summary_stats = generate_summary_stats(results)
    optimization_data = generate_optimization_comparison_chart(results)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuperDARN Comprehensive Performance Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }}
        .header h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .summary-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .summary-label {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #007acc;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .metric-title {{
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }}
        .table-responsive {{
            overflow-x: auto;
        }}
        .performance-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .performance-table th,
        .performance-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .performance-table th {{
            background-color: #007acc;
            color: white;
            font-weight: 600;
        }}
        .performance-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-good {{ color: #28a745; font-weight: bold; }}
        .status-warning {{ color: #ffc107; font-weight: bold; }}
        .status-bad {{ color: #dc3545; font-weight: bold; }}
        .optimization-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .opt-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
            text-align: center;
        }}
        .opt-card.best {{
            border-color: #28a745;
            background: #f8fff9;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            font-size: 14px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SuperDARN Comprehensive Performance Dashboard</h1>
            <p>Complete analysis of all SuperDARN components with optimization comparisons</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-number">{summary_stats['total_components']}</div>
                <div class="summary-label">Total Components</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{summary_stats['success_rate']:.1f}%</div>
                <div class="summary-label">Build Success Rate</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{summary_stats['best_optimization'][1]:.1f}%</div>
                <div class="summary-label">Best Avg Improvement ({summary_stats['best_optimization'][0]})</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{len(results['libraries'])}</div>
                <div class="summary-label">Libraries Tested</div>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">⚡ Optimization Performance Summary</div>
            <div class="optimization-summary">                <div class="opt-card{'' if summary_stats['best_optimization'][0] != 'O2' else ' best'}">
                    <h3>-O2 Optimization</h3>
                    <div class="summary-number" style="font-size: 1.8em;">{summary_stats['avg_improvements'].get('O2', 0):.1f}%</div>
                    <p>Average Improvement</p>
                </div>
                <div class="opt-card{'' if summary_stats['best_optimization'][0] != 'O3' else ' best'}">
                    <h3>-O3 Optimization</h3>
                    <div class="summary-number" style="font-size: 1.8em;">{summary_stats['avg_improvements'].get('O3', 0):.1f}%</div>
                    <p>Average Improvement</p>
                </div>
                <div class="opt-card{'' if summary_stats['best_optimization'][0] != 'Ofast' else ' best'}">
                    <h3>-Ofast Optimization</h3>
                    <div class="summary-number" style="font-size: 1.8em;">{summary_stats['avg_improvements'].get('Ofast', 0):.1f}%</div>
                    <p>Average Improvement</p>
                </div>
            </div>
        </div>
        
        <div class="metric-grid">
            {generate_component_table(results['libraries'], 'libraries')}
            {generate_component_table(results['binaries'], 'binaries')}
        </div>
        
        <div class="timestamp">
            Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            <br>
            SuperDARN Comprehensive Performance Testing Infrastructure v2.0
        </div>
    </div>
</body>
</html>'''

    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Generate SuperDARN comprehensive performance dashboard')
    parser.add_argument('--results-dir', required=True, help='Directory containing test results')
    parser.add_argument('--output', required=True, help='Output HTML file path')
    parser.add_argument('--timestamp', help='Test run timestamp')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} not found")
        sys.exit(1)
    
    # Load test results
    print("Loading test results...")
    results = load_test_results(args.results_dir)
    
    # Generate dashboard
    print(f"Generating dashboard: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_html_dashboard(results, args.output)
    
    print(f"Dashboard generated successfully: {args.output}")
    print(f"Total components: {results['metadata']['total_components']}")
    print(f"Successful builds: {results['metadata']['successful_builds']}")
    print(f"Failed builds: {results['metadata']['failed_builds']}")

if __name__ == '__main__':
    main()
