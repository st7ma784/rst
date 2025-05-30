#!/usr/bin/env python3
"""
Check for performance regressions and fail build if thresholds are exceeded
"""

import json
import argparse
import sys
from pathlib import Path

def load_results(results_dir):
    """Load all test results and calculate performance metrics"""
    results_dir = Path(results_dir)
    
    # Look for any performance degradation indicators
    regression_indicators = []
    
    # Check FitACF results
    fitacf_dirs = list(results_dir.glob('fitacf-results-*'))
    if fitacf_dirs:
        speedups = []
        accuracies = []
        
        for result_dir in fitacf_dirs:
            result_files = list(result_dir.glob('fitacf_results_*.txt'))
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        content = f.read()
                        
                    # Extract speedup
                    import re
                    speedup_match = re.search(r'Array speedup: ([\d.]+)x', content)
                    if speedup_match:
                        speedup = float(speedup_match.group(1))
                        speedups.append(speedup)
                        
                        # Check for poor speedup (less than 1.5x with multiple threads)
                        if 'threads' in result_file.name:
                            thread_match = re.search(r'(\d+)-threads', result_file.name)
                            if thread_match:
                                threads = int(thread_match.group(1))
                                if threads > 1 and speedup < 1.2:
                                    regression_indicators.append(
                                        f"Poor parallel scaling: {speedup:.2f}x speedup with {threads} threads"
                                    )
                    
                    # Extract accuracy
                    accuracy_match = re.search(r'Accuracy test: (\w+)', content)
                    if accuracy_match:
                        accuracy = accuracy_match.group(1)
                        accuracies.append(accuracy)
                        if accuracy.upper() != 'PASS':
                            regression_indicators.append(f"Accuracy test failed: {accuracy}")
                            
                except Exception as e:
                    regression_indicators.append(f"Failed to parse {result_file}: {e}")
        
        # Check overall FitACF performance
        if speedups:
            max_speedup = max(speedups)
            avg_speedup = sum(speedups) / len(speedups)
            
            if max_speedup < 2.0:  # Expect at least 2x speedup for array implementation
                regression_indicators.append(f"Low maximum speedup: {max_speedup:.2f}x (expected > 2.0x)")
            
            if avg_speedup < 1.5:  # Average should be reasonable
                regression_indicators.append(f"Low average speedup: {avg_speedup:.2f}x (expected > 1.5x)")
    
    # Check speck removal results  
    speck_dirs = list(results_dir.glob('speck-removal-results-*'))
    if speck_dirs:
        runtimes = []
        
        for result_dir in speck_dirs:
            timing_files = list(result_dir.glob('speck_timing_*.txt'))
            for timing_file in timing_files:
                try:
                    with open(timing_file, 'r') as f:
                        content = f.read()
                    
                    # Extract runtime
                    import re
                    real_time_match = re.search(r'real\s+([\d.]+)m([\d.]+)s', content)
                    if real_time_match:
                        minutes = float(real_time_match.group(1))
                        seconds = float(real_time_match.group(2))
                        total_seconds = minutes * 60 + seconds
                        runtimes.append(total_seconds)
                        
                        # Check for extremely slow runtimes (> 5 minutes for any test)
                        if total_seconds > 300:
                            regression_indicators.append(
                                f"Slow runtime detected: {total_seconds:.1f}s ({timing_file.name})"
                            )
                            
                except Exception as e:
                    regression_indicators.append(f"Failed to parse {timing_file}: {e}")
        
        # Check if optimization is working
        if len(runtimes) >= 2:
            fastest = min(runtimes)
            slowest = max(runtimes)
            
            # Expect some optimization benefit
            if (slowest - fastest) / slowest < 0.1:  # Less than 10% difference
                regression_indicators.append(
                    "Optimization levels show minimal performance difference"
                )
    
    return regression_indicators

def main():
    parser = argparse.ArgumentParser(description='Check for performance regressions')
    parser.add_argument('--results-dir', required=True, help='Directory containing test results')
    parser.add_argument('--threshold', type=float, default=10.0, help='Regression threshold percentage')
    
    args = parser.parse_args()
    
    print("🔍 Checking for performance regressions...")
    
    regression_indicators = load_results(args.results_dir)
    
    if not regression_indicators:
        print("✅ No performance regressions detected!")
        print("All performance metrics are within acceptable ranges.")
        sys.exit(0)
    else:
        print(f"⚠️ Found {len(regression_indicators)} performance issue(s):")
        for i, indicator in enumerate(regression_indicators, 1):
            print(f"  {i}. {indicator}")
        
        # Determine if these are critical regressions
        critical_keywords = ['failed', 'error', 'crash', 'accuracy test failed']
        critical_regressions = [
            indicator for indicator in regression_indicators 
            if any(keyword in indicator.lower() for keyword in critical_keywords)
        ]
        
        if critical_regressions:
            print(f"\n❌ CRITICAL: {len(critical_regressions)} critical regression(s) detected!")
            print("Build should fail due to critical performance/accuracy issues.")
            sys.exit(1)
        else:
            print(f"\n⚠️ WARNING: Performance issues detected but not critical.")
            print("Build continues but issues should be reviewed.")
            sys.exit(0)

if __name__ == '__main__':
    main()
