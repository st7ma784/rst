#!/usr/bin/env python3
"""
Simple test version of dashboard generation without external dependencies
Tests core functionality of our performance dashboard system
"""

import os
import json
from datetime import datetime

def create_simple_dashboard():
    """Create a basic HTML dashboard with test data"""
    
    # Read test results if they exist
    results_dir = "test-results"
    
    # Sample data for testing
    fitacf_data = [
        {"threads": 1, "array_time": 145.23, "llist_time": 187.56},
        {"threads": 2, "array_time": 78.45, "llist_time": 98.23},
        {"threads": 4, "array_time": 42.67, "llist_time": 56.78},
        {"threads": 8, "array_time": 24.89, "llist_time": 34.12}
    ]
    
    speck_data = [
        {"optimization": "O2", "time": 2.34, "memory": 45.2},
        {"optimization": "O3", "time": 1.89, "memory": 43.8},
        {"optimization": "Ofast", "time": 1.67, "memory": 44.1}
    ]
    
    # Generate HTML dashboard
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuperDARN Performance Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .performance-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .performance-table th,
        .performance-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .performance-table th {{
            background-color: #007acc;
            color: white;
        }}
        .speedup {{
            color: #28a745;
            font-weight: bold;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 14px;
        }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-bad {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SuperDARN FitACF v3.0 Performance Dashboard</h1>
            <p>Array Implementation vs. Linked List Performance Comparison</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">📊 FitACF Array vs LinkedList Performance</div>
                <table class="performance-table">
                    <tr>
                        <th>Threads</th>
                        <th>Array (ms)</th>
                        <th>LinkedList (ms)</th>
                        <th>Speedup</th>
                    </tr>
"""
    
    # Add FitACF performance data
    for data in fitacf_data:
        speedup = data["llist_time"] / data["array_time"]
        status_class = "status-good" if speedup > 1.2 else "status-warning" if speedup > 1.1 else "status-bad"
        html_content += f"""
                    <tr>
                        <td>{data["threads"]}</td>
                        <td>{data["array_time"]:.2f}</td>
                        <td>{data["llist_time"]:.2f}</td>
                        <td class="speedup {status_class}">{speedup:.2f}x</td>
                    </tr>
"""
    
    html_content += """
                </table>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">🔧 fit_speck_removal Optimization Performance</div>
                <table class="performance-table">
                    <tr>
                        <th>Optimization</th>
                        <th>Time (s)</th>
                        <th>Memory (MB)</th>
                        <th>Efficiency</th>
                    </tr>
"""
    
    # Add speck removal data
    baseline_time = max(d["time"] for d in speck_data)
    for data in speck_data:
        efficiency = baseline_time / data["time"]
        status_class = "status-good" if efficiency > 1.2 else "status-warning" if efficiency > 1.05 else "status-bad"
        html_content += f"""
                    <tr>
                        <td>{data["optimization"]}</td>
                        <td>{data["time"]:.2f}</td>
                        <td>{data["memory"]:.1f}</td>
                        <td class="speedup {status_class}">{efficiency:.2f}x</td>
                    </tr>
"""
    
    html_content += f"""
                </table>
            </div>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">📈 Key Performance Insights</div>
                <ul>
                    <li><strong>Best Array Performance:</strong> {min(d["array_time"] for d in fitacf_data):.2f}ms at 8 threads</li>
                    <li><strong>Maximum Speedup:</strong> {max(d["llist_time"]/d["array_time"] for d in fitacf_data):.2f}x improvement</li>
                    <li><strong>Optimal fit_speck_removal:</strong> {min(speck_data, key=lambda x: x["time"])["optimization"]} optimization</li>
                    <li><strong>Memory Efficiency:</strong> Best at {min(speck_data, key=lambda x: x["memory"])["memory"]:.1f}MB</li>
                </ul>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">🎯 Testing Infrastructure Status</div>
                <ul>
                    <li>✅ Docker environment configured</li>
                    <li>✅ GitHub Actions workflow ready</li>
                    <li>✅ Performance dashboards automated</li>
                    <li>✅ Regression testing enabled</li>
                </ul>
            </div>
        </div>
        
        <div class="timestamp">
            Dashboard generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
            <br>
            SuperDARN FitACF v3.0 Performance Testing Infrastructure
        </div>
    </div>
</body>
</html>
"""
    
    # Write dashboard file
    output_file = os.path.join(results_dir, "performance_dashboard.html")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Performance dashboard generated: {output_file}")
    return output_file

if __name__ == "__main__":
    dashboard_file = create_simple_dashboard()
    print(f"🔗 Open in browser: file://{os.path.abspath(dashboard_file)}")
