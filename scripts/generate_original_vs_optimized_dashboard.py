#!/usr/bin/env python3
"""
SuperDARN Original vs Optimized Dashboard Generator
==================================================

Generates a comprehensive dashboard showing:
1. Original components (as-is, no OpenMP assumptions)
2. Optimized components (when they exist)
3. Side-by-side performance comparisons
4. Optimization opportunities for components without optimized versions

This works with the output from superdarn_optimization_testing_framework.py
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

def load_optimization_test_results(results_file):
    """Load optimization test results from the testing framework"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def analyze_optimization_opportunities(components):
    """Analyze and categorize optimization opportunities"""
    opportunities = {
        'high_priority': [],
        'medium_priority': [],
        'low_priority': [],
        'already_optimized': []
    }
    
    for comp_name, comp_data in components.items():
        if comp_data.get('has_optimization', False):
            opportunities['already_optimized'].append(comp_name)
        else:
            original = comp_data.get('original', {})
            features = original.get('features', {}) if original else {}
            complexity = features.get('complexity_score', 0)
            has_loops = features.get('has_loops', False)
            has_linked_lists = features.get('has_linked_lists', False)
            
            # Categorize opportunity
            if has_linked_lists or (has_loops and complexity > 100):
                opportunities['high_priority'].append({
                    'name': comp_name,
                    'features': features,
                    'complexity': complexity
                })
            elif has_loops and complexity > 50:
                opportunities['medium_priority'].append({
                    'name': comp_name,
                    'features': features,
                    'complexity': complexity
                })
            else:
                opportunities['low_priority'].append({
                    'name': comp_name,
                    'features': features,
                    'complexity': complexity
                })
    
    return opportunities

def generate_component_comparison_table(components, component_type):
    """Generate HTML table showing original vs optimized comparison"""
    
    opportunities = analyze_optimization_opportunities(components)
    
    html = f'''
    <div class="metric-card">
        <div class="metric-title">🔬 {component_type.title()} Analysis: Original vs Optimized</div>
        <div class="table-responsive">
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Original Status</th>
                        <th>Optimized Status</th>
                        <th>Build Time Comparison</th>
                        <th>Performance Gain</th>
                        <th>Features Detected</th>
                        <th>Optimization Priority</th>
                    </tr>
                </thead>
                <tbody>
    '''
    
    for comp_name, comp_data in components.items():
        original = comp_data.get('original', {})
        optimized = comp_data.get('optimized', {})
        has_optimization = comp_data.get('has_optimization', False)
        performance_gain = comp_data.get('performance_gain', {})
        
        # Original status
        orig_status = original.get('build_status', 'unknown') if original else 'missing'
        orig_time = original.get('build_time', 0) if original else 0
        orig_features = original.get('features', {}) if original else {}
        
        # Optimized status
        if has_optimization and optimized:
            opt_status = optimized.get('build_status', 'unknown')
            opt_time = optimized.get('build_time', 0)
            opt_features = optimized.get('features', {})
        else:
            opt_status = 'not_created'
            opt_time = 0
            opt_features = {}
        
        # Performance metrics
        if performance_gain:
            improvement = performance_gain.get('build_time_improvement', 0)
            improvement_text = f"{improvement:+.1f}%"
            improvement_class = 'status-good' if improvement > 0 else 'status-warning'
        else:
            improvement_text = 'N/A'
            improvement_class = 'status-neutral'
        
        # Build time comparison
        if orig_time > 0 and opt_time > 0:
            time_text = f"{orig_time:.3f}s → {opt_time:.3f}s"
            time_class = 'improvement-good' if opt_time < orig_time else 'improvement-bad'
        elif orig_time > 0:
            time_text = f"{orig_time:.3f}s → ?"
            time_class = 'status-neutral'
        else:
            time_text = "No data"
            time_class = 'status-neutral'
        
        # Feature analysis
        features = []
        if orig_features.get('has_openmp') or opt_features.get('has_openmp'):
            features.append('OpenMP')
        if orig_features.get('has_simd') or opt_features.get('has_simd'):
            features.append('SIMD')
        if orig_features.get('has_linked_lists'):
            features.append('LinkedLists')
        if orig_features.get('has_loops'):
            features.append(f"Loops({orig_features.get('complexity_score', 0)})")
        
        features_text = ', '.join(features) if features else 'Basic'
        
        # Optimization priority
        if has_optimization:
            priority = '✅ Optimized'
            priority_class = 'status-good'
        else:
            complexity = orig_features.get('complexity_score', 0)
            has_loops = orig_features.get('has_loops', False)
            has_linked_lists = orig_features.get('has_linked_lists', False)
            
            if has_linked_lists or (has_loops and complexity > 100):
                priority = '🎯 HIGH'
                priority_class = 'status-high-priority'
            elif has_loops and complexity > 50:
                priority = '⚡ MEDIUM'
                priority_class = 'status-medium-priority'
            else:
                priority = '📝 LOW'
                priority_class = 'status-low-priority'
        
        # Status styling
        orig_status_class = 'status-good' if orig_status == 'success' else 'status-bad'
        
        if opt_status == 'success':
            opt_status_class = 'status-good'
        elif opt_status == 'not_created':
            opt_status_class = 'status-neutral'
        else:
            opt_status_class = 'status-bad'
        
        html += f'''
                    <tr>
                        <td><strong>{comp_name}</strong></td>
                        <td><span class="{orig_status_class}">{orig_status}</span></td>
                        <td><span class="{opt_status_class}">{opt_status.replace('_', ' ').title()}</span></td>
                        <td><span class="{time_class}">{time_text}</span></td>
                        <td><span class="{improvement_class}">{improvement_text}</span></td>
                        <td><span class="features-tag">{features_text}</span></td>
                        <td><span class="{priority_class}">{priority}</span></td>
                    </tr>
        '''
    
    html += '''
                </tbody>
            </table>
        </div>
    </div>
    '''
    
    return html

def generate_opportunity_summary(results):
    """Generate optimization opportunity summary"""
    all_components = {**results.get('libraries', {}), **results.get('binaries', {})}
    opportunities = analyze_optimization_opportunities(all_components)
    
    total_components = len(all_components)
    optimized_count = len(opportunities['already_optimized'])
    high_priority = len(opportunities['high_priority'])
    medium_priority = len(opportunities['medium_priority'])
    low_priority = len(opportunities['low_priority'])
    
    coverage_percent = (optimized_count / total_components * 100) if total_components > 0 else 0
    
    return {
        'total_components': total_components,
        'optimized_count': optimized_count,
        'coverage_percent': coverage_percent,
        'high_priority': high_priority,
        'medium_priority': medium_priority,
        'low_priority': low_priority,
        'opportunities': opportunities
    }

def generate_original_vs_optimized_dashboard(results, output_file):
    """Generate comprehensive original vs optimized dashboard"""
    
    summary = generate_opportunity_summary(results)
    metadata = results.get('metadata', {})
    timestamp = metadata.get('timestamp', datetime.now().isoformat())
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuperDARN Original vs Optimized Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1800px;
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
            font-size: 2.8em;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.3em;
            margin-bottom: 15px;
        }}
        .optimization-badges {{
            margin: 15px 0;
        }}
        .optimization-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 5px;
            font-size: 0.9em;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
        .summary-card.coverage {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .summary-card.opportunity {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
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
        .metric-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #007acc;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 30px;
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
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .comparison-table th,
        .comparison-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .comparison-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}
        .comparison-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-good {{
            background-color: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .status-bad {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .status-neutral {{
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .status-warning {{
            background-color: #fff3cd;
            color: #856404;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .status-high-priority {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .status-medium-priority {{
            background-color: #fff3cd;
            color: #856404;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .status-low-priority {{
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .improvement-good {{
            color: #28a745;
            font-weight: bold;
        }}
        .improvement-bad {{
            color: #dc3545;
            font-weight: bold;
        }}
        .features-tag {{
            background-color: #e9ecef;
            color: #495057;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .opportunity-breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .opportunity-card {{
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .opportunity-card.high {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
        }}
        .opportunity-card.medium {{
            background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
            color: white;
        }}
        .opportunity-card.low {{
            background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
            color: white;
        }}
        .opportunity-card.completed {{
            background: linear-gradient(135deg, #1dd1a1 0%, #10ac84 100%);
            color: white;
        }}
        .opportunity-number {{
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        .opportunity-label {{
            font-size: 1em;
            margin-bottom: 5px;
        }}
        .opportunity-desc {{
            font-size: 0.8em;
            opacity: 0.9;
        }}
        .progress-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .progress-bar {{
            width: 100%;
            height: 25px;
            background-color: #e9ecef;
            border-radius: 12px;
            overflow: hidden;
            margin: 15px 0;
            position: relative;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            transition: width 0.5s ease;
            border-radius: 12px;
        }}
        .progress-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: bold;
            font-size: 0.9em;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SuperDARN: Original vs Optimized</h1>
            <div class="subtitle">Systematic Performance Optimization Analysis</div>
            <div class="optimization-badges">
                <span class="optimization-badge">OpenMP Parallelization</span>
                <span class="optimization-badge">Linked List → Arrays</span>
                <span class="optimization-badge">Memory Optimization</span>
                <span class="optimization-badge">SIMD Vectorization</span>
            </div>
            <p><em>Generated: {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </div>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-number">{summary['total_components']}</div>
                <div class="summary-label">Total Components</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{summary['optimized_count']}</div>
                <div class="summary-label">Optimized Components</div>
            </div>
            <div class="summary-card coverage">
                <div class="summary-number">{summary['coverage_percent']:.1f}%</div>
                <div class="summary-label">Optimization Coverage</div>
            </div>
            <div class="summary-card opportunity">
                <div class="summary-number">{summary['high_priority']}</div>
                <div class="summary-label">High Priority Targets</div>
            </div>
        </div>

        <!-- Progress Section -->
        <div class="progress-section">
            <h3>🎯 Optimization Progress</h3>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {summary['coverage_percent']:.1f}%"></div>
                <div class="progress-text">{summary['optimized_count']}/{summary['total_components']} Components Optimized</div>
            </div>
        </div>

        <!-- Opportunity Breakdown -->
        <div class="metric-card">
            <div class="metric-title">📊 Optimization Opportunities Breakdown</div>
            <div class="opportunity-breakdown">
                <div class="opportunity-card completed">
                    <div class="opportunity-number">{summary['optimized_count']}</div>
                    <div class="opportunity-label">✅ Already Optimized</div>
                    <div class="opportunity-desc">Components with both original and optimized versions</div>
                </div>
                <div class="opportunity-card high">
                    <div class="opportunity-number">{summary['high_priority']}</div>
                    <div class="opportunity-label">🎯 High Priority</div>
                    <div class="opportunity-desc">Complex loops, linked lists, high impact potential</div>
                </div>
                <div class="opportunity-card medium">
                    <div class="opportunity-number">{summary['medium_priority']}</div>
                    <div class="opportunity-label">⚡ Medium Priority</div>
                    <div class="opportunity-desc">Loops present, moderate optimization potential</div>
                </div>
                <div class="opportunity-card low">
                    <div class="opportunity-number">{summary['low_priority']}</div>
                    <div class="opportunity-label">📝 Low Priority</div>
                    <div class="opportunity-desc">Basic code, minimal optimization impact</div>
                </div>
            </div>
        </div>

        {generate_component_comparison_table(results.get('libraries', {}), 'Libraries')}
        
        {generate_component_comparison_table(results.get('binaries', {}), 'Binaries')}

        <div class="footer">
            <p><strong>SuperDARN Optimization Testing Framework</strong> - Systematically identifying and tracking optimization opportunities</p>
            <p>🎯 = High Priority (Complex loops/linked lists) | ⚡ = Medium Priority (Loops) | 📝 = Low Priority (Basic code) | ✅ = Already Optimized</p>
            <p>This framework tests components as-is and provides space for optimized versions to be added later.</p>
        </div>
    </div>
</body>
</html>'''

    # Write the dashboard
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate SuperDARN Original vs Optimized Dashboard')
    parser.add_argument('--results', default='test-results/optimization_test_results.json', 
                       help='Path to optimization test results JSON file')
    parser.add_argument('--output', default='test-results/superdarn_original_vs_optimized_dashboard.html',
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    # Load results
    results = load_optimization_test_results(args.results)
    if not results:
        print("❌ Failed to load test results")
        return 1
    
    # Generate dashboard
    print("🎨 Generating original vs optimized dashboard...")
    output_file = generate_original_vs_optimized_dashboard(results, args.output)
    
    print(f"✅ Dashboard generated: {output_file}")
    
    # Print summary
    summary = generate_opportunity_summary(results)
    print(f"\n📈 OPTIMIZATION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Components: {summary['total_components']}")
    print(f"Already Optimized: {summary['optimized_count']} ({summary['coverage_percent']:.1f}%)")
    print(f"High Priority Opportunities: {summary['high_priority']}")
    print(f"Medium Priority Opportunities: {summary['medium_priority']}")
    print(f"Low Priority Opportunities: {summary['low_priority']}")
    
    if summary['high_priority'] > 0:
        print(f"\n🎯 High Priority Components (Complex loops/linked lists):")
        for opp in summary['opportunities']['high_priority'][:5]:  # Show first 5
            features = opp['features']
            linked_lists = "LinkedLists " if features.get('has_linked_lists') else ""
            openmp = "OpenMP " if features.get('has_openmp') else ""
            complexity = f"Complexity:{opp['complexity']}"
            print(f"  • {opp['name']} - {linked_lists}{openmp}{complexity}")

if __name__ == "__main__":
    sys.exit(main())
