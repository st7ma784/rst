#!/usr/bin/env python3
"""
Generate comprehensive performance dashboard for SuperDARN testing
Processes test results and creates HTML dashboard with interactive charts
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime
from jinja2 import Template

def parse_fitacf_results(results_dir):
    """Parse FitACF array vs linked list test results"""
    fitacf_data = []
    
    fitacf_pattern = re.compile(r'fitacf-results-(\d+)-threads-(\w+)-data')
    
    for result_dir in Path(results_dir).iterdir():
        if result_dir.is_dir():
            match = fitacf_pattern.match(result_dir.name)
            if match:
                threads, data_size = match.groups()
                
                result_file = result_dir / f'fitacf_results_{threads}_{data_size}.txt'
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        content = f.read()
                        
                    # Parse performance metrics
                    speedup_match = re.search(r'Array speedup: ([\d.]+)x', content)
                    memory_match = re.search(r'Memory reduction: ([\d.]+)%', content)
                    accuracy_match = re.search(r'Accuracy test: (\w+)', content)
                    
                    if speedup_match:
                        fitacf_data.append({
                            'threads': int(threads),
                            'data_size': data_size,
                            'speedup': float(speedup_match.group(1)),
                            'memory_reduction': float(memory_match.group(1)) if memory_match else 0,
                            'accuracy': accuracy_match.group(1) if accuracy_match else 'Unknown'
                        })
    
    return pd.DataFrame(fitacf_data)

def parse_speck_removal_results(results_dir):
    """Parse fit_speck_removal performance test results"""
    speck_data = []
    
    speck_pattern = re.compile(r'speck-removal-results-(\w+)-(\w+)')
    
    for result_dir in Path(results_dir).iterdir():
        if result_dir.is_dir():
            match = speck_pattern.match(result_dir.name)
            if match:
                file_size, opt_level = match.groups()
                
                timing_file = result_dir / f'speck_timing_{file_size}_{opt_level}.txt'
                if timing_file.exists():
                    with open(timing_file, 'r') as f:
                        content = f.read()
                    
                    # Parse timing information
                    real_time_match = re.search(r'real\s+([\d.]+)m([\d.]+)s', content)
                    user_time_match = re.search(r'user\s+([\d.]+)m([\d.]+)s', content)
                    sys_time_match = re.search(r'sys\s+([\d.]+)m([\d.]+)s', content)
                    
                    if real_time_match:
                        real_minutes = float(real_time_match.group(1))
                        real_seconds = float(real_time_match.group(2))
                        real_total = real_minutes * 60 + real_seconds
                        
                        user_minutes = float(user_time_match.group(1)) if user_time_match else 0
                        user_seconds = float(user_time_match.group(2)) if user_time_match else 0
                        user_total = user_minutes * 60 + user_seconds
                        
                        speck_data.append({
                            'file_size': file_size,
                            'optimization': opt_level,
                            'real_time': real_total,
                            'user_time': user_total,
                            'efficiency': (user_total / real_total) if real_total > 0 else 0
                        })
    
    return pd.DataFrame(speck_data)

def create_fitacf_charts(df, output_dir):
    """Create charts for FitACF performance results"""
    if df.empty:
        return
    
    # Speedup by thread count and data size
    plt.figure(figsize=(12, 8))
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Speedup heatmap
    pivot_speedup = df.pivot(index='data_size', columns='threads', values='speedup')
    sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1)
    ax1.set_title('FitACF Array Speedup (vs Linked List)')
    ax1.set_xlabel('Thread Count')
    ax1.set_ylabel('Data Size')
    
    # Speedup line plot
    for size in df['data_size'].unique():
        size_data = df[df['data_size'] == size]
        ax2.plot(size_data['threads'], size_data['speedup'], marker='o', label=size)
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup Scaling by Thread Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Memory reduction
    pivot_memory = df.pivot(index='data_size', columns='threads', values='memory_reduction')
    sns.heatmap(pivot_memory, annot=True, fmt='.1f', cmap='Blues', ax=ax3)
    ax3.set_title('Memory Reduction (%)')
    ax3.set_xlabel('Thread Count')
    ax3.set_ylabel('Data Size')
    
    # Efficiency (speedup per thread)
    df['efficiency'] = df['speedup'] / df['threads']
    for size in df['data_size'].unique():
        size_data = df[df['data_size'] == size]
        ax4.plot(size_data['threads'], size_data['efficiency'], marker='s', label=size)
    ax4.set_xlabel('Thread Count')
    ax4.set_ylabel('Efficiency (Speedup/Thread)')
    ax4.set_title('Parallel Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fitacf_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_speck_removal_charts(df, output_dir):
    """Create charts for fit_speck_removal performance results"""
    if df.empty:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Runtime by optimization level and file size
    pivot_time = df.pivot(index='file_size', columns='optimization', values='real_time')
    sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax1)
    ax1.set_title('fit_speck_removal Runtime (seconds)')
    ax1.set_xlabel('Optimization Level')
    ax1.set_ylabel('File Size')
    
    # Runtime comparison bar chart
    df_grouped = df.groupby(['optimization', 'file_size'])['real_time'].mean().reset_index()
    pivot_for_plot = df_grouped.pivot(index='file_size', columns='optimization', values='real_time')
    pivot_for_plot.plot(kind='bar', ax=ax2)
    ax2.set_title('Runtime by Optimization Level')
    ax2.set_xlabel('File Size')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.legend(title='Optimization')
    ax2.tick_params(axis='x', rotation=45)
    
    # Speedup relative to O2
    o2_baseline = df[df['optimization'] == 'O2'].set_index('file_size')['real_time']
    speedup_data = []
    for _, row in df.iterrows():
        if row['file_size'] in o2_baseline.index:
            baseline_time = o2_baseline[row['file_size']]
            speedup = baseline_time / row['real_time']
            speedup_data.append({
                'file_size': row['file_size'],
                'optimization': row['optimization'],
                'speedup': speedup
            })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        pivot_speedup = speedup_df.pivot(index='file_size', columns='optimization', values='speedup')
        sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3)
        ax3.set_title('Speedup vs -O2 Baseline')
        ax3.set_xlabel('Optimization Level')
        ax3.set_ylabel('File Size')
    
    # CPU efficiency
    for size in df['file_size'].unique():
        size_data = df[df['file_size'] == size]
        ax4.bar(size_data['optimization'], size_data['efficiency'], alpha=0.7, label=size)
    ax4.set_xlabel('Optimization Level')
    ax4.set_ylabel('CPU Efficiency (user/real time)')
    ax4.set_title('CPU Utilization Efficiency')
    ax4.legend(title='File Size')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speck_removal_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_stats(fitacf_df, speck_df):
    """Generate summary statistics"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'fitacf': {},
        'speck_removal': {}
    }
    
    if not fitacf_df.empty:
        summary['fitacf'] = {
            'max_speedup': float(fitacf_df['speedup'].max()),
            'avg_speedup': float(fitacf_df['speedup'].mean()),
            'max_memory_reduction': float(fitacf_df['memory_reduction'].max()),
            'best_config': fitacf_df.loc[fitacf_df['speedup'].idxmax()].to_dict(),
            'accuracy_pass_rate': len(fitacf_df[fitacf_df['accuracy'] == 'PASS']) / len(fitacf_df) * 100
        }
    
    if not speck_df.empty:
        o2_times = speck_df[speck_df['optimization'] == 'O2']['real_time']
        ofast_times = speck_df[speck_df['optimization'] == 'Ofast']['real_time']
        
        if not o2_times.empty and not ofast_times.empty:
            avg_improvement = (o2_times.mean() - ofast_times.mean()) / o2_times.mean() * 100
        else:
            avg_improvement = 0
            
        summary['speck_removal'] = {
            'fastest_time': float(speck_df['real_time'].min()),
            'avg_improvement_ofast': float(avg_improvement),
            'best_efficiency': float(speck_df['efficiency'].max()),
            'best_config': speck_df.loc[speck_df['real_time'].idxmin()].to_dict()
        }
    
    return summary

def create_html_dashboard(summary, commit_sha, branch, output_dir):
    """Create HTML dashboard"""
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuperDARN Performance Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #eee; }
        .metric-card { background: #f8f9fa; padding: 20px; margin: 10px; border-radius: 6px; border-left: 4px solid #007bff; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #6c757d; font-size: 0.9em; margin-top: 5px; }
        .charts { display: grid; grid-template-columns: 1fr; gap: 30px; margin: 30px 0; }
        .chart-container { text-align: center; }
        .chart-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
        .section-title { font-size: 1.5em; margin: 30px 0 15px 0; padding-bottom: 10px; border-bottom: 1px solid #ddd; }
        .commit-info { background: #e9ecef; padding: 15px; border-radius: 4px; margin-bottom: 20px; font-family: monospace; }
        .pass { color: #28a745; }
        .fail { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SuperDARN Performance Dashboard</h1>
            <p>FitACF v3.0 Array Implementation & fit_speck_removal Optimization Results</p>
        </div>
        
        <div class="commit-info">
            <strong>Commit:</strong> {{ commit_sha[:8] }} | <strong>Branch:</strong> {{ branch }} | <strong>Generated:</strong> {{ summary.timestamp[:19] }}
        </div>

        <h2 class="section-title">📊 FitACF Array Implementation Results</h2>
        <div class="summary-grid">
            {% if summary.fitacf %}
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(summary.fitacf.max_speedup) }}x</div>
                <div class="metric-label">Maximum Speedup</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(summary.fitacf.max_memory_reduction) }}%</div>
                <div class="metric-label">Memory Reduction</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {% if summary.fitacf.accuracy_pass_rate == 100 %}pass{% else %}fail{% endif %}">
                    {{ "%.0f"|format(summary.fitacf.accuracy_pass_rate) }}%
                </div>
                <div class="metric-label">Accuracy Tests Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ summary.fitacf.best_config.threads }} threads</div>
                <div class="metric-label">Best Configuration</div>
            </div>
            {% else %}
            <div class="metric-card">
                <div class="metric-value">No Data</div>
                <div class="metric-label">FitACF tests not run</div>
            </div>
            {% endif %}
        </div>

        <h2 class="section-title">🔧 fit_speck_removal Optimization Results</h2>
        <div class="summary-grid">
            {% if summary.speck_removal %}
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(summary.speck_removal.fastest_time) }}s</div>
                <div class="metric-label">Fastest Runtime</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(summary.speck_removal.avg_improvement_ofast) }}%</div>
                <div class="metric-label">-Ofast vs -O2 Improvement</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.0f"|format(summary.speck_removal.best_efficiency * 100) }}%</div>
                <div class="metric-label">Best CPU Efficiency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">-{{ summary.speck_removal.best_config.optimization }}</div>
                <div class="metric-label">Best Optimization Level</div>
            </div>
            {% else %}
            <div class="metric-card">
                <div class="metric-value">No Data</div>
                <div class="metric-label">Speck removal tests not run</div>
            </div>
            {% endif %}
        </div>

        <h2 class="section-title">📈 Performance Charts</h2>
        <div class="charts">
            <div class="chart-container">
                <h3>FitACF Array vs Linked List Performance</h3>
                <img src="fitacf_performance.png" alt="FitACF Performance Charts">
            </div>
            <div class="chart-container">
                <h3>fit_speck_removal Optimization Comparison</h3>
                <img src="speck_removal_performance.png" alt="Speck Removal Performance Charts">
            </div>
        </div>

        <div style="margin-top: 40px; text-align: center; color: #6c757d; font-size: 0.9em;">
            Generated by SuperDARN Performance Testing Pipeline | 
            <a href="https://github.com/{{ github_repo }}">View on GitHub</a>
        </div>
    </div>
</body>
</html>
"""
    
    template = Template(template_str)
    html_content = template.render(
        summary=summary,
        commit_sha=commit_sha,
        branch=branch,
        github_repo=os.environ.get('GITHUB_REPOSITORY', 'superdarn/rst')
    )
    
    with open(output_dir / 'index.html', 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Generate SuperDARN performance dashboard')
    parser.add_argument('--results-dir', required=True, help='Directory containing test results')
    parser.add_argument('--output-dir', required=True, help='Output directory for dashboard')
    parser.add_argument('--commit-sha', required=True, help='Git commit SHA')
    parser.add_argument('--branch', required=True, help='Git branch name')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Parsing test results...")
    fitacf_df = parse_fitacf_results(results_dir)
    speck_df = parse_speck_removal_results(results_dir)
    
    print("Generating charts...")
    create_fitacf_charts(fitacf_df, output_dir)
    create_speck_removal_charts(speck_df, output_dir)
    
    print("Generating summary statistics...")
    summary = generate_summary_stats(fitacf_df, speck_df)
    
    # Save summary as JSON
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Creating HTML dashboard...")
    create_html_dashboard(summary, args.commit_sha, args.branch, output_dir)
    
    print(f"Dashboard generated successfully in {output_dir}")
    print(f"Open {output_dir}/index.html to view the results")

if __name__ == '__main__':
    main()
