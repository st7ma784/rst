#!/usr/bin/env python3
"""
SuperDARN Current Component Analysis Dashboard Generator
Creates interactive HTML dashboard from comprehensive_component_analysis.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

def load_component_analysis(json_file):
    """Load component analysis results"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None

def calculate_statistics(data):
    """Calculate summary statistics from component data"""
    total_libraries = len(data.get('libraries', {}))
    total_binaries = len(data.get('binaries', {}))
    total_components = total_libraries + total_binaries
    
    # Count successful analyses
    successful_libs = sum(1 for lib in data.get('libraries', {}).values() 
                         if lib.get('status') == 'analyzed')
    successful_bins = sum(1 for bin in data.get('binaries', {}).values() 
                         if bin.get('status') == 'analyzed')
    successful_total = successful_libs + successful_bins
    
    # Count total source files
    total_c_files = sum(lib.get('c_files', 0) for lib in data.get('libraries', {}).values())
    total_c_files += sum(bin.get('c_files', 0) for bin in data.get('binaries', {}).values())
    
    total_h_files = sum(lib.get('h_files', 0) for lib in data.get('libraries', {}).values())
    total_h_files += sum(bin.get('h_files', 0) for bin in data.get('binaries', {}).values())
    
    # Build complexity distribution
    complexity_counts = {'simple': 0, 'medium': 0, 'high': 0}
    all_components = {**data.get('libraries', {}), **data.get('binaries', {})}
    
    for comp in all_components.values():
        complexity = comp.get('build_complexity', 'unknown')
        if complexity in complexity_counts:
            complexity_counts[complexity] += 1
    
    return {
        'total_components': total_components,
        'total_libraries': total_libraries,
        'total_binaries': total_binaries,
        'successful_total': successful_total,
        'successful_libs': successful_libs,
        'successful_bins': successful_bins,
        'success_rate': (successful_total / total_components * 100) if total_components > 0 else 0,
        'total_c_files': total_c_files,
        'total_h_files': total_h_files,
        'total_source_files': total_c_files + total_h_files,
        'complexity_counts': complexity_counts,
        'analysis_time': data.get('analysis_time_seconds', 0)
    }

def generate_component_table(components, component_type, stats):
    """Generate HTML table for components"""
    html = f'''
    <div class="metric-card">
        <div class="metric-title">📚 {component_type.title()} Analysis Results</div>
        <div class="table-responsive">
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Status</th>
                        <th>Build Complexity</th>
                        <th>C Files</th>
                        <th>H Files</th>
                        <th>Makefile Lines</th>
                        <th>Estimated Build Time</th>
                    </tr>
                </thead>
                <tbody>
    '''
    
    for comp_name, comp_data in components.items():
        status = comp_data.get('status', 'unknown')
        complexity = comp_data.get('build_complexity', 'unknown')
        c_files = comp_data.get('c_files', 0)
        h_files = comp_data.get('h_files', 0)
        makefile_lines = comp_data.get('makefile_lines', 0)
        estimated_time = comp_data.get('estimated_build_time_seconds', 0)
        
        # Status styling
        status_class = 'status-good' if status == 'analyzed' else 'status-bad'
        complexity_class = {
            'simple': 'status-good',
            'medium': 'status-warning', 
            'high': 'status-bad'
        }.get(complexity, '')
        
        html += f'''
                    <tr>
                        <td><strong>{comp_name}</strong></td>
                        <td><span class="{status_class}">{status}</span></td>
                        <td><span class="{complexity_class}">{complexity}</span></td>
                        <td>{c_files}</td>
                        <td>{h_files}</td>
                        <td>{makefile_lines}</td>
                        <td>{estimated_time:.2f}s</td>
                    </tr>
        '''
    
    html += '''
                </tbody>
            </table>
        </div>
    </div>
    '''
    
    return html

def generate_html_dashboard(data, output_file):
    """Generate complete HTML dashboard"""
    
    stats = calculate_statistics(data)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuperDARN Component Analysis Dashboard</title>
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
        .complexity-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .complexity-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
            text-align: center;
        }}
        .complexity-card.simple {{
            border-color: #28a745;
            background: #f8fff9;
        }}
        .complexity-card.medium {{
            border-color: #ffc107;
            background: #fffef5;
        }}
        .complexity-card.high {{
            border-color: #dc3545;
            background: #fff5f5;
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
            <h1>🚀 SuperDARN Component Analysis Dashboard</h1>
            <p>Comprehensive analysis of all SuperDARN components</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-number">{stats['total_components']}</div>
                <div class="summary-label">Total Components</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{stats['successful_total']}</div>
                <div class="summary-label">Successfully Analyzed</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{stats['success_rate']:.1f}%</div>
                <div class="summary-label">Success Rate</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{stats['total_source_files']}</div>
                <div class="summary-label">Source Files</div>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">📊 Build Complexity Distribution</div>
            <div class="complexity-summary">
                <div class="complexity-card simple">
                    <h3>Simple</h3>
                    <div class="summary-number" style="font-size: 1.8em;">{stats['complexity_counts']['simple']}</div>
                    <p>Components</p>
                </div>
                <div class="complexity-card medium">
                    <h3>Medium</h3>
                    <div class="summary-number" style="font-size: 1.8em;">{stats['complexity_counts']['medium']}</div>
                    <p>Components</p>
                </div>
                <div class="complexity-card high">
                    <h3>High</h3>
                    <div class="summary-number" style="font-size: 1.8em;">{stats['complexity_counts']['high']}</div>
                    <p>Components</p>
                </div>
            </div>
        </div>
        
        <div class="metric-grid">
            {generate_component_table(data.get('libraries', {}), 'libraries', stats)}
            {generate_component_table(data.get('binaries', {}), 'binaries', stats)}
        </div>
        
        <div class="metric-card">
            <div class="metric-title">📈 Analysis Summary</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div>
                    <h4>Libraries</h4>
                    <p><strong>{stats['total_libraries']}</strong> total</p>
                    <p><strong>{stats['successful_libs']}</strong> analyzed</p>
                </div>
                <div>
                    <h4>Binaries</h4>
                    <p><strong>{stats['total_binaries']}</strong> total</p>
                    <p><strong>{stats['successful_bins']}</strong> analyzed</p>
                </div>
                <div>
                    <h4>Source Files</h4>
                    <p><strong>{stats['total_c_files']}</strong> C files</p>
                    <p><strong>{stats['total_h_files']}</strong> H files</p>
                </div>
                <div>
                    <h4>Performance</h4>
                    <p><strong>{stats['analysis_time']:.2f}s</strong> analysis time</p>
                    <p><strong>Parallel processing</strong></p>
                </div>
            </div>
        </div>
        
        <div class="timestamp">
            Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            <br>
            SuperDARN Component Analysis Infrastructure - Analysis completed: {data.get('timestamp', 'Unknown')}        </div>
    </div>
</body>
</html>'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard generated: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_current_dashboard.py <comprehensive_component_analysis.json> [output.html]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "component_analysis_dashboard.html"
    
    # Load data
    data = load_component_analysis(input_file)
    if not data:
        print("Failed to load component analysis data")
        sys.exit(1)
    
    # Generate dashboard
    generate_html_dashboard(data, output_file)
    print(f"SuperDARN Component Analysis Dashboard generated successfully!")
    print(f"Open {output_file} in your browser to view the results.")

if __name__ == "__main__":
    main()
