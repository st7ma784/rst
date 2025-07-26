#!/bin/bash

# Set paths
FIT_SRC_DIR=../src
FIT_INCLUDE=../include
SUPERDARN_INCLUDE=../../../../include
LIB_DIR=../../../../lib

# Create lib directory if it doesn't exist
mkdir -p ${LIB_DIR}

# Compile the fit library
echo "Compiling fit.1.35 library..."
cd ${FIT_SRC_DIR}
gcc -c -Wall -O3 -fPIC -I${FIT_INCLUDE} -I${SUPERDARN_INCLUDE} fit.c fitscan.c fitcfit.c fitread.c fitwrite.c fitseek.c fitinx.c

# Create static library
echo "Creating static library..."
ar rcs libfit.1.35.a fit.o fitscan.o fitcfit.o fitread.o fitwrite.o fitseek.o fitinx.o
mv libfit.1.35.a ${LIB_DIR}/
cd -

# Compile the profiling test
echo "Compiling profiling test..."
gcc -Wall -O3 -fopenmp -I${FIT_INCLUDE} -I${SUPERDARN_INCLUDE} -c profile_fit.c

# Link the test program
echo "Linking test program..."
gcc -o profile_fit profile_fit.o -L${LIB_DIR} -lfit.1.35 -lm -fopenmp

# Run the test
if [ $? -eq 0 ]; then
    echo "\n=== Running profiling test ==="
    ./profile_fit
else
    echo "\nCompilation failed. Please check the error messages above."
fi
