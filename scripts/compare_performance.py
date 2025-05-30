#!/usr/bin/env python3
"""
Compare performance results between current and baseline versions
Generates markdown report for GitHub PR comments
"""

import json
import argparse
from pathlib import Path
import sys

def load_summary(path):
    """Load performance summary JSON"""
    try:
        with open(path / 'summary.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def compare_fitacf_performance(current, baseline):
    """Compare FitACF performance metrics"""
    if not current or not baseline:
        return "❌ **FitACF Comparison**: Missing data for comparison"
    
    current_fitacf = current.get('fitacf', {})
    baseline_fitacf = baseline.get('fitacf', {})
    
    if not current_fitacf or not baseline_fitacf:
        return "❌ **FitACF Comparison**: No FitACF data available"
    
    # Compare speedup
    current_speedup = current_fitacf.get('max_speedup', 0)
    baseline_speedup = baseline_fitacf.get('max_speedup', 0)
    speedup_change = ((current_speedup - baseline_speedup) / baseline_speedup * 100) if baseline_speedup > 0 else 0
    
    # Compare memory reduction
    current_memory = current_fitacf.get('max_memory_reduction', 0)
    baseline_memory = baseline_fitacf.get('max_memory_reduction', 0)
    memory_change = current_memory - baseline_memory
    
    # Compare accuracy
    current_accuracy = current_fitacf.get('accuracy_pass_rate', 0)
    baseline_accuracy = baseline_fitacf.get('accuracy_pass_rate', 0)
    accuracy_change = current_accuracy - baseline_accuracy
    
    # Generate status emoji
    if speedup_change >= 0 and accuracy_change >= 0:
        status = "✅"
    elif speedup_change < -5:  # More than 5% regression
        status = "❌"
    else:
        status = "⚠️"
    
    return f"""
{status} **FitACF Array Implementation Performance**

| Metric | Current | Baseline | Change |
|--------|---------|----------|--------|
| Max Speedup | {current_speedup:.2f}x | {baseline_speedup:.2f}x | {speedup_change:+.1f}% |
| Memory Reduction | {current_memory:.1f}% | {baseline_memory:.1f}% | {memory_change:+.1f}% |
| Accuracy Pass Rate | {current_accuracy:.0f}% | {baseline_accuracy:.0f}% | {accuracy_change:+.1f}% |
"""

def compare_speck_removal_performance(current, baseline):
    """Compare fit_speck_removal performance metrics"""
    if not current or not baseline:
        return "❌ **Speck Removal Comparison**: Missing data for comparison"
    
    current_speck = current.get('speck_removal', {})
    baseline_speck = baseline.get('speck_removal', {})
    
    if not current_speck or not baseline_speck:
        return "❌ **Speck Removal Comparison**: No speck removal data available"
    
    # Compare runtime
    current_time = current_speck.get('fastest_time', 0)
    baseline_time = baseline_speck.get('fastest_time', 0)
    time_change = ((current_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
    
    # Compare optimization improvement
    current_improvement = current_speck.get('avg_improvement_ofast', 0)
    baseline_improvement = baseline_speck.get('avg_improvement_ofast', 0)
    improvement_change = current_improvement - baseline_improvement
    
    # Compare efficiency
    current_efficiency = current_speck.get('best_efficiency', 0)
    baseline_efficiency = baseline_speck.get('best_efficiency', 0)
    efficiency_change = (current_efficiency - baseline_efficiency) * 100
    
    # Generate status emoji
    if time_change <= 0 and efficiency_change >= 0:  # Faster runtime and better efficiency
        status = "✅"
    elif time_change > 10:  # More than 10% slower
        status = "❌"
    else:
        status = "⚠️"
    
    return f"""
{status} **fit_speck_removal Optimization Performance**

| Metric | Current | Baseline | Change |
|--------|---------|----------|--------|
| Fastest Runtime | {current_time:.2f}s | {baseline_time:.2f}s | {time_change:+.1f}% |
| -Ofast Improvement | {current_improvement:.1f}% | {baseline_improvement:.1f}% | {improvement_change:+.1f}% |
| Best CPU Efficiency | {current_efficiency*100:.0f}% | {baseline_efficiency*100:.0f}% | {efficiency_change:+.1f}% |
"""

def check_regressions(current, baseline, threshold=10):
    """Check for significant performance regressions"""
    regressions = []
    
    if current and baseline:
        # Check FitACF regressions
        current_fitacf = current.get('fitacf', {})
        baseline_fitacf = baseline.get('fitacf', {})
        
        if current_fitacf and baseline_fitacf:
            current_speedup = current_fitacf.get('max_speedup', 0)
            baseline_speedup = baseline_fitacf.get('max_speedup', 0)
            
            if baseline_speedup > 0:
                speedup_change = ((current_speedup - baseline_speedup) / baseline_speedup * 100)
                if speedup_change < -threshold:
                    regressions.append(f"FitACF speedup regression: {speedup_change:.1f}%")
        
        # Check speck removal regressions
        current_speck = current.get('speck_removal', {})
        baseline_speck = baseline.get('speck_removal', {})
        
        if current_speck and baseline_speck:
            current_time = current_speck.get('fastest_time', 0)
            baseline_time = baseline_speck.get('fastest_time', 0)
            
            if baseline_time > 0:
                time_change = ((current_time - baseline_time) / baseline_time * 100)
                if time_change > threshold:
                    regressions.append(f"fit_speck_removal runtime regression: {time_change:.1f}%")
    
    return regressions

def generate_comparison_report(current_path, baseline_path, threshold):
    """Generate complete comparison report"""
    current = load_summary(current_path) if current_path.exists() else None
    baseline = load_summary(baseline_path) if baseline_path.exists() else None
    
    # Header
    report = "# 🚀 SuperDARN Performance Comparison Report\n\n"
    
    if not current:
        report += "❌ **Error**: No current performance data found\n"
        return report
    
    if not baseline:
        report += "⚠️ **Warning**: No baseline data found for comparison. This may be the first run.\n\n"
        # Still show current results
        if current.get('fitacf'):
            report += f"## Current FitACF Results\n"
            report += f"- **Max Speedup**: {current['fitacf'].get('max_speedup', 0):.2f}x\n"
            report += f"- **Memory Reduction**: {current['fitacf'].get('max_memory_reduction', 0):.1f}%\n"
            report += f"- **Accuracy**: {current['fitacf'].get('accuracy_pass_rate', 0):.0f}% pass rate\n\n"
        
        if current.get('speck_removal'):
            report += f"## Current fit_speck_removal Results\n"
            report += f"- **Fastest Runtime**: {current['speck_removal'].get('fastest_time', 0):.2f}s\n"
            report += f"- **Optimization Improvement**: {current['speck_removal'].get('avg_improvement_ofast', 0):.1f}%\n"
            report += f"- **CPU Efficiency**: {current['speck_removal'].get('best_efficiency', 0)*100:.0f}%\n\n"
        
        return report
    
    # Performance comparisons
    report += compare_fitacf_performance(current, baseline)
    report += "\n"
    report += compare_speck_removal_performance(current, baseline)
    report += "\n"
    
    # Check for regressions
    regressions = check_regressions(current, baseline, threshold)
    
    if regressions:
        report += "## ⚠️ Performance Regressions Detected\n\n"
        for regression in regressions:
            report += f"- {regression}\n"
        report += "\n"
    else:
        report += "## ✅ No Significant Performance Regressions\n\n"
    
    # Add links to detailed results
    report += "## 📊 Detailed Results\n\n"
    report += "View the complete performance dashboard with interactive charts in the Actions artifacts.\n\n"
    
    # Footer
    report += "---\n"
    report += "*Generated by SuperDARN Performance Testing Pipeline*"
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Compare performance results')
    parser.add_argument('--current', required=True, help='Directory with current results')
    parser.add_argument('--baseline', required=True, help='Directory with baseline results')
    parser.add_argument('--output', default='comparison.md', help='Output markdown file')
    parser.add_argument('--threshold', type=float, default=10.0, help='Regression threshold percentage')
    
    args = parser.parse_args()
    
    current_path = Path(args.current)
    baseline_path = Path(args.baseline)
    
    report = generate_comparison_report(current_path, baseline_path, args.threshold)
    
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Comparison report generated: {args.output}")
    
    # Also print to stdout for logging
    print("\n" + "="*50)
    print(report)
    print("="*50)

if __name__ == '__main__':
    main()
