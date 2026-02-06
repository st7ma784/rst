#!/bin/bash

# Comprehensive CUDA Module Analysis and Conversion Plan
# Analyzes ALL SuperDARN modules for CUDA conversion potential
# Creates systematic conversion plan with native CUDA data structures

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_DIR="$SCRIPT_DIR/codebase/superdarn/src.lib/tk"
ANALYSIS_LOG="$SCRIPT_DIR/cuda_comprehensive_analysis.log"

# Initialize log
echo "Comprehensive CUDA Module Analysis" > "$ANALYSIS_LOG"
echo "Generated: $(date)" >> "$ANALYSIS_LOG"
echo "=================================" >> "$ANALYSIS_LOG"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$ANALYSIS_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[SUCCESS] $1" >> "$ANALYSIS_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $1" >> "$ANALYSIS_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$ANALYSIS_LOG"
}

log_priority() {
    echo -e "${PURPLE}[PRIORITY]${NC} $1"
    echo "[PRIORITY] $1" >> "$ANALYSIS_LOG"
}

# Function to analyze module for CUDA potential
analyze_module_cuda_potential() {
    local module_name="$1"
    local module_path="$CODEBASE_DIR/$module_name"
    
    if [ ! -d "$module_path" ]; then
        return 1
    fi
    
    local cuda_score=0
    local reasons=()
    local data_structures=()
    local computational_patterns=()
    
    # Check if already has CUDA support
    if [ -f "$module_path/makefile.cuda" ]; then
        echo "EXISTING_CUDA"
        return 0
    fi
    
    # Analyze source files for CUDA potential indicators
    if [ -d "$module_path/src" ]; then
        local src_files=$(find "$module_path/src" -name "*.c" 2>/dev/null | wc -l)
        
        if [ $src_files -gt 0 ]; then
            # Check for computational patterns
            local math_ops=$(find "$module_path/src" -name "*.c" -exec grep -l "sin\|cos\|sqrt\|exp\|log\|pow\|atan\|fft" {} \; 2>/dev/null | wc -l)
            local loops=$(find "$module_path/src" -name "*.c" -exec grep -c "for\s*(" {} \; 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
            local arrays=$(find "$module_path/src" -name "*.c" -exec grep -c "\[\|malloc\|calloc" {} \; 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
            local matrix_ops=$(find "$module_path/src" -name "*.c" -exec grep -l "matrix\|linear\|solve\|invert" {} \; 2>/dev/null | wc -l)
            local complex_ops=$(find "$module_path/src" -name "*.c" -exec grep -l "complex\|real\|imag\|phase" {} \; 2>/dev/null | wc -l)
            local signal_processing=$(find "$module_path/src" -name "*.c" -exec grep -l "filter\|fft\|spectrum\|frequency" {} \; 2>/dev/null | wc -l)
            local interpolation=$(find "$module_path/src" -name "*.c" -exec grep -l "interpolat\|grid\|mesh\|spatial" {} \; 2>/dev/null | wc -l)
            
            # Score based on computational intensity
            if [ $math_ops -gt 0 ]; then
                cuda_score=$((cuda_score + 20))
                reasons+=("Mathematical operations detected")
            fi
            
            if [ $loops -gt 10 ]; then
                cuda_score=$((cuda_score + 15))
                reasons+=("High loop count ($loops)")
            fi
            
            if [ $arrays -gt 5 ]; then
                cuda_score=$((cuda_score + 10))
                reasons+=("Array processing detected")
                data_structures+=("Arrays")
            fi
            
            if [ $matrix_ops -gt 0 ]; then
                cuda_score=$((cuda_score + 25))
                reasons+=("Matrix operations detected")
                data_structures+=("Matrices")
                computational_patterns+=("Linear Algebra")
            fi
            
            if [ $complex_ops -gt 0 ]; then
                cuda_score=$((cuda_score + 15))
                reasons+=("Complex number operations")
                data_structures+=("Complex Numbers")
                computational_patterns+=("Complex Arithmetic")
            fi
            
            if [ $signal_processing -gt 0 ]; then
                cuda_score=$((cuda_score + 20))
                reasons+=("Signal processing operations")
                computational_patterns+=("DSP")
            fi
            
            if [ $interpolation -gt 0 ]; then
                cuda_score=$((cuda_score + 15))
                reasons+=("Spatial/grid operations")
                computational_patterns+=("Interpolation")
            fi
            
            # Check for specific SuperDARN patterns
            local range_processing=$(find "$module_path/src" -name "*.c" -exec grep -l "range\|gate\|beam" {} \; 2>/dev/null | wc -l)
            local time_series=$(find "$module_path/src" -name "*.c" -exec grep -l "time\|sequence\|lag" {} \; 2>/dev/null | wc -l)
            local fitting=$(find "$module_path/src" -name "*.c" -exec grep -l "fit\|least.*square\|regression" {} \; 2>/dev/null | wc -l)
            
            if [ $range_processing -gt 0 ]; then
                cuda_score=$((cuda_score + 10))
                reasons+=("Range/beam processing")
                computational_patterns+=("Range Processing")
            fi
            
            if [ $time_series -gt 0 ]; then
                cuda_score=$((cuda_score + 10))
                reasons+=("Time series processing")
                computational_patterns+=("Time Series")
            fi
            
            if [ $fitting -gt 0 ]; then
                cuda_score=$((cuda_score + 20))
                reasons+=("Fitting algorithms")
                computational_patterns+=("Curve Fitting")
            fi
        fi
    fi
    
    # Output results
    echo "$cuda_score|$(IFS=';'; echo "${reasons[*]}")|$(IFS=';'; echo "${data_structures[*]}")|$(IFS=';'; echo "${computational_patterns[*]}")"
}

# Function to categorize modules by CUDA potential
categorize_modules() {
    log_info "Analyzing all SuperDARN modules for CUDA conversion potential..."
    
    declare -A module_scores
    declare -A module_reasons
    declare -A module_data_structures
    declare -A module_patterns
    declare -a existing_cuda_modules
    declare -a high_priority_modules
    declare -a medium_priority_modules
    declare -a low_priority_modules
    declare -a utility_modules
    
    # Get all modules
    local all_modules=($(ls -1 "$CODEBASE_DIR" | grep -E "^[a-zA-Z]" | sort))
    
    log_info "Found ${#all_modules[@]} total modules to analyze"
    
    for module in "${all_modules[@]}"; do
        log_info "Analyzing module: $module"
        
        local analysis_result=$(analyze_module_cuda_potential "$module")
        
        if [ "$analysis_result" = "EXISTING_CUDA" ]; then
            existing_cuda_modules+=("$module")
            log_success "$module - Already has CUDA support"
        else
            local score=$(echo "$analysis_result" | cut -d'|' -f1)
            local reasons=$(echo "$analysis_result" | cut -d'|' -f2)
            local data_structs=$(echo "$analysis_result" | cut -d'|' -f3)
            local patterns=$(echo "$analysis_result" | cut -d'|' -f4)
            
            module_scores["$module"]=$score
            module_reasons["$module"]=$reasons
            module_data_structures["$module"]=$data_structs
            module_patterns["$module"]=$patterns
            
            if [ $score -ge 40 ]; then
                high_priority_modules+=("$module")
                log_priority "$module - HIGH PRIORITY (Score: $score)"
            elif [ $score -ge 20 ]; then
                medium_priority_modules+=("$module")
                log_warning "$module - MEDIUM PRIORITY (Score: $score)"
            elif [ $score -ge 10 ]; then
                low_priority_modules+=("$module")
                log_info "$module - LOW PRIORITY (Score: $score)"
            else
                utility_modules+=("$module")
                log_info "$module - UTILITY/IO MODULE (Score: $score)"
            fi
        fi
    done
    
    # Generate comprehensive report
    cat > "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md" << EOF
# Comprehensive SuperDARN CUDA Conversion Plan

Generated: $(date)
Total Modules Analyzed: ${#all_modules[@]}

## Executive Summary

- **Existing CUDA Modules**: ${#existing_cuda_modules[@]}
- **High Priority for Conversion**: ${#high_priority_modules[@]}
- **Medium Priority for Conversion**: ${#medium_priority_modules[@]}
- **Low Priority for Conversion**: ${#low_priority_modules[@]}
- **Utility/IO Modules**: ${#utility_modules[@]}

## Existing CUDA-Enabled Modules (${#existing_cuda_modules[@]})

EOF

    for module in "${existing_cuda_modules[@]}"; do
        echo "- ✅ **$module** - Already CUDA-enabled" >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md"
    done

    cat >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md" << EOF

## High Priority Conversion Targets (${#high_priority_modules[@]})

These modules show strong computational patterns ideal for GPU acceleration:

EOF

    for module in "${high_priority_modules[@]}"; do
        local score=${module_scores["$module"]}
        local reasons=${module_reasons["$module"]}
        local data_structs=${module_data_structures["$module"]}
        local patterns=${module_patterns["$module"]}
        
        cat >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md" << EOF
### $module (Score: $score)
- **Conversion Reasons**: ${reasons//;/, }
- **CUDA Data Structures**: ${data_structs//;/, }
- **Computational Patterns**: ${patterns//;/, }
- **Priority**: HIGH - Immediate conversion recommended

EOF
    done

    cat >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md" << EOF

## Medium Priority Conversion Targets (${#medium_priority_modules[@]})

These modules have moderate computational benefits from GPU acceleration:

EOF

    for module in "${medium_priority_modules[@]}"; do
        local score=${module_scores["$module"]}
        local reasons=${module_reasons["$module"]}
        local data_structs=${module_data_structures["$module"]}
        local patterns=${module_patterns["$module"]}
        
        cat >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md" << EOF
### $module (Score: $score)
- **Conversion Reasons**: ${reasons//;/, }
- **CUDA Data Structures**: ${data_structs//;/, }
- **Computational Patterns**: ${patterns//;/, }
- **Priority**: MEDIUM - Convert after high priority modules

EOF
    done

    cat >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md" << EOF

## Low Priority Conversion Targets (${#low_priority_modules[@]})

These modules may benefit from CUDA but with limited performance gains:

EOF

    for module in "${low_priority_modules[@]}"; do
        local score=${module_scores["$module"]}
        local reasons=${module_reasons["$module"]}
        
        echo "- **$module** (Score: $score) - ${reasons//;/, }" >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md"
    done

    cat >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md" << EOF

## Utility/IO Modules (${#utility_modules[@]})

These modules are primarily I/O or utility focused with minimal computational benefit:

EOF

    for module in "${utility_modules[@]}"; do
        local score=${module_scores["$module"]}
        echo "- **$module** (Score: $score) - Utility/IO module" >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md"
    done

    cat >> "$SCRIPT_DIR/comprehensive_cuda_conversion_plan.md" << EOF

## CUDA Native Data Structure Migration Plan

### Phase 1: Core Data Structures
- **Arrays**: Convert to \`cudaMallocManaged\` or \`cuda_array_t\`
- **Matrices**: Use cuBLAS-compatible layouts
- **Complex Numbers**: Use \`cuComplex\` and \`cuDoubleComplex\`
- **Linked Lists**: Convert to GPU-compatible array-based structures

### Phase 2: Computational Patterns
- **Linear Algebra**: Integrate cuBLAS and cuSOLVER
- **DSP Operations**: Use cuFFT for frequency domain processing
- **Interpolation**: Implement CUDA texture memory for spatial operations
- **Time Series**: Use CUDA streams for pipeline processing

### Phase 3: Memory Optimization
- **Unified Memory**: Use \`cudaMallocManaged\` for seamless CPU/GPU access
- **Memory Pools**: Implement custom allocators for frequent allocations
- **Texture Memory**: Use for read-only data with spatial locality
- **Shared Memory**: Optimize for inter-thread communication

## Conversion Implementation Strategy

1. **Batch Conversion**: Process modules in priority order
2. **Native Data Structures**: Migrate to CUDA-native types during conversion
3. **Performance Validation**: Benchmark each conversion
4. **API Compatibility**: Maintain drop-in replacement capability
5. **Documentation**: Update usage guides for each converted module

## Expected Performance Improvements

- **High Priority Modules**: 5-15x speedup expected
- **Medium Priority Modules**: 2-8x speedup expected
- **Low Priority Modules**: 1.5-3x speedup expected

## Resource Requirements

- **Development Time**: ~2-4 hours per high priority module
- **Testing Time**: ~1 hour per module for validation
- **Documentation**: ~30 minutes per module
- **Total Estimated Time**: ~40-60 hours for complete conversion

EOF

    # Print summary
    echo ""
    echo "========================================="
    echo "COMPREHENSIVE CUDA ANALYSIS COMPLETE"
    echo "========================================="
    echo "Total Modules: ${#all_modules[@]}"
    echo "Existing CUDA: ${#existing_cuda_modules[@]}"
    echo "High Priority: ${#high_priority_modules[@]}"
    echo "Medium Priority: ${#medium_priority_modules[@]}"
    echo "Low Priority: ${#low_priority_modules[@]}"
    echo "Utility/IO: ${#utility_modules[@]}"
    echo ""
    echo "Report saved to: comprehensive_cuda_conversion_plan.md"
    echo "========================================="
    
    # Return arrays for further processing
    echo "HIGH_PRIORITY_MODULES=(${high_priority_modules[*]})"
    echo "MEDIUM_PRIORITY_MODULES=(${medium_priority_modules[*]})"
    echo "LOW_PRIORITY_MODULES=(${low_priority_modules[*]})"
}

# Main execution
main() {
    log_info "Starting comprehensive CUDA module analysis..."
    
    # Check CUDA availability
    if command -v nvcc &> /dev/null; then
        log_success "CUDA detected - ready for comprehensive conversion"
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "NVCC version: $NVCC_VERSION"
    else
        log_warning "CUDA not detected - will create compatibility layers"
    fi
    
    # Perform comprehensive analysis
    categorize_modules
    
    log_success "Comprehensive CUDA analysis completed!"
    log_info "Next step: Begin systematic conversion of high-priority modules"
}

# Execute main function
main "$@"
