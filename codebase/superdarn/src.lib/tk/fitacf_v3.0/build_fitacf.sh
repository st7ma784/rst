#!/bin/bash
# build_fitacf.sh - Build script for SuperDARN FitACF v3.0 Array Implementation
#
# This script automates the build process for both linked list and array
# implementations with various configuration options.

set -e  # Exit on any error

# Configuration
PROJECT_NAME="SuperDARN FitACF v3.0"
BUILD_DIR="build"
SRC_DIR="src"
TEST_DIR="test"

# Default options
BUILD_LLIST=true
BUILD_ARRAY=true
BUILD_TESTS=true
ENABLE_OPENMP=true
ENABLE_CUDA=false
BUILD_TYPE="Release"
VERBOSE=false
CLEAN_BUILD=false
RUN_TESTS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -c, --clean             Clean build (remove build directory)"
    echo "  -t, --tests             Build and run tests"
    echo "  -v, --verbose           Verbose output"
    echo "  --debug                 Debug build"
    echo "  --release               Release build (default)"
    echo "  --no-llist              Skip linked list implementation"
    echo "  --no-array              Skip array implementation"
    echo "  --no-openmp             Disable OpenMP"
    echo "  --enable-cuda           Enable CUDA support"
    echo "  --make-only             Use traditional makefile instead of CMake"
    echo ""
    echo "Examples:"
    echo "  $0                      # Build everything with default settings"
    echo "  $0 --debug -t           # Debug build with tests"
    echo "  $0 --enable-cuda        # Release build with CUDA support"
    echo "  $0 --no-llist --tests   # Array implementation only with tests"
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check for C compiler
    if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
        print_error "No C compiler found (gcc or clang required)"
        exit 1
    fi
    
    # Check for make
    if ! command -v make &> /dev/null; then
        print_error "make not found"
        exit 1
    fi
    
    # Check for CMake if using CMake build
    if [ "$USE_MAKEFILE" != "true" ] && ! command -v cmake &> /dev/null; then
        print_warning "CMake not found, falling back to traditional makefile"
        USE_MAKEFILE=true
    fi
    
    # Check for OpenMP support
    if [ "$ENABLE_OPENMP" = "true" ]; then
        if gcc -fopenmp -xc /dev/null -o /dev/null 2>/dev/null; then
            print_status "OpenMP support detected"
        else
            print_warning "OpenMP not supported by compiler, disabling"
            ENABLE_OPENMP=false
        fi
    fi
    
    # Check for CUDA
    if [ "$ENABLE_CUDA" = "true" ]; then
        if command -v nvcc &> /dev/null; then
            print_status "CUDA compiler detected"
        else
            print_error "CUDA requested but nvcc not found"
            exit 1
        fi
    fi
    
    print_success "Dependencies check completed"
}

# Function to clean build directory
clean_build() {
    print_status "Cleaning build directory..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
    fi
    
    # Clean object files in src directory
    if [ -d "$SRC_DIR" ]; then
        cd "$SRC_DIR"
        rm -f *.o *.a test_baseline test_comparison
        cd ..
        print_success "Source directory cleaned"
    fi
}

# Function to build with CMake
build_with_cmake() {
    print_status "Building with CMake..."
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure CMake options
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    
    if [ "$BUILD_LLIST" = "false" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DBUILD_LLIST_IMPLEMENTATION=OFF"
    fi
    
    if [ "$BUILD_ARRAY" = "false" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DBUILD_ARRAY_IMPLEMENTATION=OFF"
    fi
    
    if [ "$BUILD_TESTS" = "false" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DBUILD_TESTS=OFF"
    fi
    
    if [ "$ENABLE_OPENMP" = "false" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DENABLE_OPENMP=OFF"
    fi
    
    if [ "$ENABLE_CUDA" = "true" ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DENABLE_CUDA=ON"
    fi
    
    # Configure
    print_status "Configuring build..."
    if [ "$VERBOSE" = "true" ]; then
        cmake .. $CMAKE_ARGS
    else
        cmake .. $CMAKE_ARGS > cmake_config.log 2>&1
    fi
    
    # Build
    print_status "Compiling..."
    if [ "$VERBOSE" = "true" ]; then
        make -j$(nproc)
    else
        make -j$(nproc) > make_build.log 2>&1
    fi
    
    cd ..
    print_success "CMake build completed"
}

# Function to build with traditional makefile
build_with_makefile() {
    print_status "Building with traditional makefile..."
    
    cd "$SRC_DIR"
    
    # Build targets based on options
    MAKE_TARGETS=""
    
    if [ "$BUILD_LLIST" = "true" ]; then
        MAKE_TARGETS="$MAKE_TARGETS fitacf_llist"
    fi
    
    if [ "$BUILD_ARRAY" = "true" ]; then
        MAKE_TARGETS="$MAKE_TARGETS fitacf_array"
    fi
    
    if [ "$BUILD_TESTS" = "true" ]; then
        MAKE_TARGETS="$MAKE_TARGETS tests"
    fi
    
    # Set build type
    if [ "$BUILD_TYPE" = "Debug" ]; then
        MAKE_TARGETS="debug $MAKE_TARGETS"
    fi
    
    # Build
    if [ "$VERBOSE" = "true" ]; then
        make -f makefile_array $MAKE_TARGETS
    else
        make -f makefile_array $MAKE_TARGETS > make_build.log 2>&1
    fi
    
    cd ..
    print_success "Makefile build completed"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    if [ "$USE_MAKEFILE" = "true" ]; then
        cd "$SRC_DIR"
        if [ -f "test_baseline" ]; then
            print_status "Running baseline tests..."
            ./test_baseline
        fi
        if [ -f "test_comparison" ]; then
            print_status "Running comparison tests..."
            ./test_comparison
        fi
        cd ..
    else
        cd "$BUILD_DIR"
        if [ -f "test_baseline" ]; then
            print_status "Running baseline tests..."
            ./test_baseline
        fi
        if [ -f "test_comparison" ]; then
            print_status "Running comparison tests..."
            ./test_comparison
        fi
        cd ..
    fi
    
    print_success "Tests completed"
}

# Function to show build summary
show_summary() {
    echo ""
    echo "==============================================="
    echo "  $PROJECT_NAME Build Summary"
    echo "==============================================="
    echo "Build type:             $BUILD_TYPE"
    echo "Linked list impl:       $BUILD_LLIST"
    echo "Array impl:             $BUILD_ARRAY"
    echo "Tests:                  $BUILD_TESTS"
    echo "OpenMP:                 $ENABLE_OPENMP"
    echo "CUDA:                   $ENABLE_CUDA"
    echo "Build system:           $([ "$USE_MAKEFILE" = "true" ] && echo "Makefile" || echo "CMake")"
    echo "==============================================="
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -t|--tests)
            BUILD_TESTS=true
            RUN_TESTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --no-llist)
            BUILD_LLIST=false
            shift
            ;;
        --no-array)
            BUILD_ARRAY=false
            shift
            ;;
        --no-openmp)
            ENABLE_OPENMP=false
            shift
            ;;
        --enable-cuda)
            ENABLE_CUDA=true
            shift
            ;;
        --make-only)
            USE_MAKEFILE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main build process
main() {
    echo "Building $PROJECT_NAME"
    echo "======================="
    
    # Clean if requested
    if [ "$CLEAN_BUILD" = "true" ]; then
        clean_build
    fi
    
    # Check dependencies
    check_dependencies
    
    # Show build configuration
    show_summary
    
    # Build
    if [ "$USE_MAKEFILE" = "true" ]; then
        build_with_makefile
    else
        build_with_cmake
    fi
    
    # Run tests if requested
    if [ "$RUN_TESTS" = "true" ]; then
        run_tests
    fi
    
    print_success "Build process completed successfully!"
    
    # Show next steps
    echo ""
    echo "Next steps:"
    if [ "$BUILD_TESTS" = "true" ] && [ "$RUN_TESTS" = "false" ]; then
        echo "  Run tests: ./build_fitacf.sh --tests"
    fi
    echo "  See build outputs in: $([ "$USE_MAKEFILE" = "true" ] && echo "src/" || echo "build/")"
    echo "  Integration guide: See documentation in docs/"
}

# Run main function
main
