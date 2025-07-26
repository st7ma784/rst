#!/bin/bash

# Set up paths
TOPDIR=/home/user/rst/codebase/superdarn
FITDIR=${TOPDIR}/src.lib/tk/fit.1.35
CFITDIR=${TOPDIR}/src.lib/tk/cfit.1.19

# Create include directory
mkdir -p ${FITDIR}/include

# Copy required headers
cp ${CFITDIR}/include/cfitdata.h ${FITDIR}/include/
cp ${CFITDIR}/include/cfitread.h ${FITDIR}/include/ 2>/dev/null || true
cp ${CFITDIR}/include/cfitwrite.h ${FITDIR}/include/ 2>/dev/null || true

# Build the library
echo "Building fit.1.35 library..."
cd ${FITDIR}
make -f Makefile.standalone

# Build the test program
echo -e "\nBuilding test program..."
gcc -Wall -O3 -I${FITDIR}/include -I${CFITDIR}/include -I${TOPDIR}/include \
    -I${TOPDIR}/src.lib/tk/fit.1.35/include \
    -I${TOPDIR}/src.lib/tk/fit.1.35/src \
    test_profile.c -L. -lfit.1.35 -o test_profile -lm

if [ $? -eq 0 ]; then
    echo -e "\nBuild successful! Running tests...\n"
    ./test_profile
else
    echo -e "\nBuild failed. Please check the error messages above."
    exit 1
fi
