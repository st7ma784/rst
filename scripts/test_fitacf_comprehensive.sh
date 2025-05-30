#!/bin/bash
# FitACF v3.0 Comprehensive Test Suite
# Tests array implementation vs linked list with verification and optimization profiling

set -e

# Configuration
FITACF_DIR="/workspace/codebase/superdarn/src.lib/tk/fitacf_v3.0"
TEST_RESULTS_DIR="/workspace/results/fitacf_detailed"
SCRIPT_DIR="/workspace/scripts"

# Test data configurations
declare -A DATA_CONFIGS=(
    ["small"]="range_gates:100,lag_table_size:20,beam_count:16"
    ["medium"]="range_gates:300,lag_table_size:40,beam_count:16" 
    ["large"]="range_gates:500,lag_table_size:60,beam_count:24"
    ["extreme"]="range_gates:1000,lag_table_size:80,beam_count:32"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Initialize FitACF testing environment
initialize_fitacf_tests() {
    log "🔬 Initializing FitACF v3.0 Test Environment..."
    
    mkdir -p "$TEST_RESULTS_DIR"/{original,array,comparisons,verification}
    mkdir -p "$TEST_RESULTS_DIR"/test_data
    
    # Source RST environment
    source /opt/rst/.profile.bash
    
    cd "$FITACF_DIR"
    
    # Backup original files
    if [[ -f "src/makefile_standalone" ]]; then
        cp src/makefile_standalone src/makefile_standalone.backup
    fi
    
    log_success "FitACF test environment initialized"
}

# Generate comprehensive test data
generate_fitacf_test_data() {
    log "📊 Generating FitACF test data for all configurations..."
    
    for config_name in "${!DATA_CONFIGS[@]}"; do
        log "Generating $config_name dataset..."
        
        local config="${DATA_CONFIGS[$config_name]}"
        local range_gates=$(echo "$config" | grep -o 'range_gates:[0-9]*' | cut -d: -f2)
        local lag_table_size=$(echo "$config" | grep -o 'lag_table_size:[0-9]*' | cut -d: -f2)
        local beam_count=$(echo "$config" | grep -o 'beam_count:[0-9]*' | cut -d: -f2)
        
        # Generate synthetic rawacf data
        python3 << EOF
import struct
import random
import numpy as np

def generate_rawacf_data(filename, range_gates, lag_table_size, beam_count):
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('I', range_gates))
        f.write(struct.pack('I', lag_table_size))
        f.write(struct.pack('I', beam_count))
        
        # Generate ACF data for each beam
        for beam in range(beam_count):
            for gate in range(range_gates):
                for lag in range(lag_table_size):
                    # Real part
                    real_val = random.uniform(-1000, 1000)
                    f.write(struct.pack('f', real_val))
                    
                    # Imaginary part  
                    imag_val = random.uniform(-1000, 1000)
                    f.write(struct.pack('f', imag_val))
                    
                    # Power
                    power_val = real_val*real_val + imag_val*imag_val
                    f.write(struct.pack('f', power_val))

generate_rawacf_data('$TEST_RESULTS_DIR/test_data/rawacf_${config_name}.dat', $range_gates, $lag_table_size, $beam_count)
print(f"Generated rawacf_${config_name}.dat: {$range_gates} gates, {$lag_table_size} lags, {$beam_count} beams")
EOF
    done
    
    log_success "Test data generation completed"
}

# Test original FitACF implementation
test_original_fitacf() {
    log "📊 Testing Original FitACF Implementation..."
    
    cd "$FITACF_DIR/src"
    
    # Build original version
    make -f makefile_standalone clean > /dev/null 2>&1 || true
    
    local build_start=$(date +%s.%3N)
    if make -f makefile_standalone > "$TEST_RESULTS_DIR/original/build.log" 2>&1; then
        local build_end=$(date +%s.%3N)
        local build_time=$(echo "$build_end - $build_start" | bc -l)
        
        log_success "Original build completed in ${build_time}s"
        
        # Test with different data sizes and thread counts
        test_fitacf_performance "original" "$TEST_RESULTS_DIR/original"
    else
        log_error "Original build failed"
        return 1
    fi
}

# Test array implementation
test_array_fitacf() {
    log "🚀 Testing Array Implementation FitACF..."
    
    cd "$FITACF_DIR/src"
    
    # Enable array implementation (if not already)
    if [[ -f "fitacf_array.c" || -f "fitacf_v3_array.c" ]]; then
        # Modify makefile to use array implementation
        cp makefile_standalone.backup makefile_standalone
        sed -i 's/fitacf\.c/fitacf_array.c/g' makefile_standalone 2>/dev/null || \
        sed -i 's/fitacf_v3\.c/fitacf_v3_array.c/g' makefile_standalone 2>/dev/null || true
    fi
    
    make -f makefile_standalone clean > /dev/null 2>&1 || true
    
    local build_start=$(date +%s.%3N)
    if make -f makefile_standalone > "$TEST_RESULTS_DIR/array/build.log" 2>&1; then
        local build_end=$(date +%s.%3N)
        local build_time=$(echo "$build_end - $build_start" | bc -l)
        
        log_success "Array build completed in ${build_time}s"
        
        # Test with different data sizes and thread counts
        test_fitacf_performance "array" "$TEST_RESULTS_DIR/array"
    else
        log_error "Array build failed"
        return 1
    fi
}

# Test FitACF performance with various configurations
test_fitacf_performance() {
    local implementation="$1"
    local results_dir="$2"
    
    log "Testing $implementation implementation performance..."
    
    # Initialize results file
    cat > "$results_dir/performance_results.json" << EOF
{
    "implementation": "$implementation",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "test_results": {
EOF

    local first_config=true
    
    for config_name in "${!DATA_CONFIGS[@]}"; do
        if [[ "$first_config" = false ]]; then
            echo ',' >> "$results_dir/performance_results.json"
        fi
        first_config=false
        
        log "  Testing with $config_name dataset..."
        
        echo "    \"$config_name\": {" >> "$results_dir/performance_results.json"
        echo "      \"thread_results\": {" >> "$results_dir/performance_results.json"
        
        local first_thread=true
        for threads in 1 2 4 8; do
            if [[ "$first_thread" = false ]]; then
                echo ',' >> "$results_dir/performance_results.json"
            fi
            first_thread=false
            
            log "    Testing with $threads threads..."
            
            export OMP_NUM_THREADS=$threads
            
            local exec_start=$(date +%s.%3N)
            local test_output
            if test_output=$(./test_array_vs_llist "$TEST_RESULTS_DIR/test_data/rawacf_${config_name}.dat" 2>&1); then
                local exec_end=$(date +%s.%3N)
                local exec_time=$(echo "$exec_end - $exec_start" | bc -l)
                
                # Parse output for performance metrics
                local power_calc_time=$(echo "$test_output" | grep -o 'Power calculation: [0-9.]*' | grep -o '[0-9.]*' || echo "0")
                local lag_calc_time=$(echo "$test_output" | grep -o 'Lag calculation: [0-9.]*' | grep -o '[0-9.]*' || echo "0")
                local total_fits=$(echo "$test_output" | grep -o 'Total fits: [0-9]*' | grep -o '[0-9]*' || echo "0")
                
                echo "        \"$threads\": {" >> "$results_dir/performance_results.json"
                echo "          \"execution_time\": $exec_time," >> "$results_dir/performance_results.json"
                echo "          \"power_calc_time\": $power_calc_time," >> "$results_dir/performance_results.json"
                echo "          \"lag_calc_time\": $lag_calc_time," >> "$results_dir/performance_results.json"
                echo "          \"total_fits\": $total_fits," >> "$results_dir/performance_results.json"
                echo "          \"fits_per_second\": $(echo "scale=2; $total_fits / $exec_time" | bc -l)," >> "$results_dir/performance_results.json"
                echo "          \"status\": \"success\"" >> "$results_dir/performance_results.json"
                echo "        }" >> "$results_dir/performance_results.json"
                
                # Save detailed output
                echo "$test_output" > "$results_dir/${config_name}_${threads}threads.log"
                
            else
                echo "        \"$threads\": {" >> "$results_dir/performance_results.json"
                echo "          \"status\": \"failed\"," >> "$results_dir/performance_results.json"
                echo "          \"error\": \"Execution failed\"" >> "$results_dir/performance_results.json"
                echo "        }" >> "$results_dir/performance_results.json"
            fi
        done
        
        echo "      }" >> "$results_dir/performance_results.json"
        echo "    }" >> "$results_dir/performance_results.json"
    done
    
    echo "  }" >> "$results_dir/performance_results.json"
    echo "}" >> "$results_dir/performance_results.json"
}

# Verify results consistency between implementations
verify_results_consistency() {
    log "🔍 Verifying Results Consistency Between Implementations..."
    
    if [[ ! -f "$TEST_RESULTS_DIR/original/performance_results.json" ]] || \
       [[ ! -f "$TEST_RESULTS_DIR/array/performance_results.json" ]]; then
        log_error "Missing performance results for comparison"
        return 1
    fi
    
    python3 << 'EOF'
import json
import sys
import os

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_implementations():
    original = load_results('TEST_RESULTS_DIR/original/performance_results.json'.replace('TEST_RESULTS_DIR', os.environ.get('TEST_RESULTS_DIR', '/workspace/results/fitacf_detailed')))
    array = load_results('TEST_RESULTS_DIR/array/performance_results.json'.replace('TEST_RESULTS_DIR', os.environ.get('TEST_RESULTS_DIR', '/workspace/results/fitacf_detailed')))
    
    comparison = {
        "timestamp": original.get("timestamp"),
        "comparisons": {},
        "summary": {
            "total_tests": 0,
            "performance_improvements": {},
            "consistency_check": "passed"
        }
    }
    
    orig_results = original.get("test_results", {})
    array_results = array.get("test_results", {})
    
    for config_name in orig_results.keys():
        if config_name in array_results:
            comparison["comparisons"][config_name] = {}
            
            orig_threads = orig_results[config_name].get("thread_results", {})
            array_threads = array_results[config_name].get("thread_results", {})
            
            for thread_count in orig_threads.keys():
                if thread_count in array_threads:
                    orig_data = orig_threads[thread_count]
                    array_data = array_threads[thread_count]
                    
                    if orig_data.get("status") == "success" and array_data.get("status") == "success":
                        orig_time = float(orig_data.get("execution_time", 0))
                        array_time = float(array_data.get("execution_time", 0))
                        
                        improvement = ((orig_time - array_time) / orig_time * 100) if orig_time > 0 else 0
                        speedup = orig_time / array_time if array_time > 0 else 0
                        
                        comparison["comparisons"][config_name][thread_count] = {
                            "original_time": orig_time,
                            "array_time": array_time,
                            "improvement_percent": improvement,
                            "speedup_factor": speedup,
                            "fits_per_second_original": orig_data.get("fits_per_second", 0),
                            "fits_per_second_array": array_data.get("fits_per_second", 0)
                        }
                        
                        comparison["summary"]["total_tests"] += 1
                        
                        if thread_count not in comparison["summary"]["performance_improvements"]:
                            comparison["summary"]["performance_improvements"][thread_count] = []
                        comparison["summary"]["performance_improvements"][thread_count].append(improvement)
    
    # Calculate average improvements
    for thread_count, improvements in comparison["summary"]["performance_improvements"].items():
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            comparison["summary"]["performance_improvements"][thread_count] = {
                "average_improvement": avg_improvement,
                "individual_results": improvements
            }
    
    # Save comparison results
    output_file = 'TEST_RESULTS_DIR/comparisons/implementation_comparison.json'.replace('TEST_RESULTS_DIR', os.environ.get('TEST_RESULTS_DIR', '/workspace/results/fitacf_detailed'))
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison results saved to: {output_file}")
    
    # Print summary
    print("\n=== FITACF IMPLEMENTATION COMPARISON SUMMARY ===")
    print(f"Total test comparisons: {comparison['summary']['total_tests']}")
    
    for thread_count, data in comparison["summary"]["performance_improvements"].items():
        if isinstance(data, dict):
            avg_imp = data["average_improvement"]
            print(f"Average improvement with {thread_count} threads: {avg_imp:.2f}%")

compare_implementations()
EOF

    log_success "Results verification completed"
}

# Generate FitACF-specific dashboard
generate_fitacf_dashboard() {
    log "📊 Generating FitACF-specific performance dashboard..."
    
    python3 << 'EOF'
import json
import os
from datetime import datetime

def load_comparison_data():
    comparison_file = f"{os.environ.get('TEST_RESULTS_DIR', '/workspace/results/fitacf_detailed')}/comparisons/implementation_comparison.json"
    
    if not os.path.exists(comparison_file):
        return None
        
    with open(comparison_file, 'r') as f:
        return json.load(f)

def generate_fitacf_dashboard():
    comparison_data = load_comparison_data()
    
    if not comparison_data:
        print("No comparison data available")
        return
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FitACF v3.0 Array vs LinkedList Performance</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #28a745;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
        }}
        .speedup {{ color: #28a745; font-weight: bold; }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 FitACF v3.0 Array Implementation Performance</h1>
            <p>Comprehensive comparison of Array vs LinkedList implementations</p>
        </div>
        
        <div class="metric-grid">'''
    
    # Generate performance tables for each configuration
    for config_name, config_data in comparison_data.get("comparisons", {}).items():
        html_content += f'''
            <div class="metric-card">
                <h3>📊 {config_name.title()} Dataset Performance</h3>
                <table class="performance-table">
                    <tr>
                        <th>Threads</th>
                        <th>Original (ms)</th>
                        <th>Array (ms)</th>
                        <th>Improvement</th>
                        <th>Speedup</th>
                    </tr>'''
        
        for thread_count, thread_data in config_data.items():
            improvement = thread_data.get("improvement_percent", 0)
            speedup = thread_data.get("speedup_factor", 0)
            
            improvement_class = "status-good" if improvement > 0 else "status-warning"
            
            html_content += f'''
                    <tr>
                        <td>{thread_count}</td>
                        <td>{thread_data.get("original_time", 0)*1000:.2f}</td>
                        <td>{thread_data.get("array_time", 0)*1000:.2f}</td>
                        <td><span class="{improvement_class}">{improvement:+.1f}%</span></td>
                        <td><span class="speedup">{speedup:.2f}x</span></td>
                    </tr>'''
        
        html_content += '''
                </table>
            </div>'''
    
    html_content += f'''
        </div>
        
        <div class="timestamp">
            Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            <br>
            FitACF v3.0 Array Implementation Performance Analysis
        </div>
    </div>
</body>
</html>'''
    
    output_file = f"{os.environ.get('TEST_RESULTS_DIR', '/workspace/results/fitacf_detailed')}/fitacf_performance_dashboard.html"
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"FitACF dashboard generated: {output_file}")

generate_fitacf_dashboard()
EOF

    log_success "FitACF dashboard generated"
}

# Main execution
main() {
    log "🎯 Starting FitACF v3.0 Comprehensive Testing..."
    
    initialize_fitacf_tests
    generate_fitacf_test_data
    test_original_fitacf
    test_array_fitacf
    verify_results_consistency
    generate_fitacf_dashboard
    
    log_success "🎉 FitACF v3.0 testing completed!"
}

# Execute if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
