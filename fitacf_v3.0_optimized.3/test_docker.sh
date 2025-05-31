#!/bin/bash
# Simple test script for FitACF v3.0 in Docker
echo "========================================"
echo "FitACF v3.0 Docker Test"
echo "========================================"
echo "Current directory: $(pwd)"
echo "Available files:"
ls -la
echo ""
echo "Source files:"
ls -la src/ || echo "No src directory"
echo ""
echo "Test files:"  
ls -la test/ || echo "No test directory"
echo ""
echo "Include files:"
ls -la include/ || echo "No include directory"
echo ""
echo "Testing GCC availability:"
gcc --version
echo ""
echo "Testing OpenMP support:"
echo '#include <omp.h>' > test_omp.c
echo 'int main() { return omp_get_max_threads(); }' >> test_omp.c
if gcc -fopenmp test_omp.c -o test_omp 2>/dev/null; then
    echo "OpenMP is available"
    ./test_omp && echo "OpenMP test successful"
else
    echo "OpenMP not available"
fi
rm -f test_omp.c test_omp
echo ""
echo "========================================"
