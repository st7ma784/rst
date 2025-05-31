# CMakeLists.txt for SuperDARN Grid Parallel Library v1.24
# Cross-platform build system with advanced optimization support

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GridParallel VERSION 1.24.0 LANGUAGES C CXX)

# C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(ENABLE_CUDA "Enable CUDA acceleration" ON)
option(ENABLE_AVX2 "Enable AVX2 SIMD instructions" ON)
option(ENABLE_TESTING "Build test suite" ON)
option(ENABLE_BENCHMARKS "Build performance benchmarks" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ENABLE_LTO "Enable Link Time Optimization" OFF)

# Platform detection
if(WIN32)
    set(PLATFORM_WINDOWS TRUE)
elseif(APPLE)
    set(PLATFORM_MACOS TRUE)
else()
    set(PLATFORM_LINUX TRUE)
endif()

# Compiler-specific flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -march=native -mtune=native")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -march=native")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "/Od /Zi /DDEBUG")
endif()

# Link Time Optimization
if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Find required packages
find_package(Threads REQUIRED)

# OpenMP support
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_C_FOUND)
        add_compile_definitions(OPENMP_ENABLED)
    endif()
endif()

# CUDA support
if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        add_compile_definitions(CUDA_ENABLED)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    else()
        message(WARNING "CUDA not found, disabling CUDA support")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# AVX2 support detection
if(ENABLE_AVX2)
    include(CheckCSourceCompiles)
    check_c_source_compiles("
        #include <immintrin.h>
        int main() {
            __m256d a = _mm256_set1_pd(1.0);
            return 0;
        }" HAVE_AVX2)
    
    if(HAVE_AVX2)
        add_compile_definitions(AVX2_ENABLED)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            add_compile_options(-mavx2 -mfma)
        elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
            add_compile_options(/arch:AVX2)
        endif()
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../base/src.lib/misc/include
)

# Source files
file(GLOB_RECURSE C_SOURCES "src/*.c")
file(GLOB_RECURSE CU_SOURCES "src/*.cu")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create library
set(LIBRARY_SOURCES ${C_SOURCES})
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND LIBRARY_SOURCES ${CU_SOURCES})
endif()

add_library(grid_parallel ${LIBRARY_SOURCES})

# Set library properties
set_target_properties(grid_parallel PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${HEADERS}"
)

# Link libraries
target_link_libraries(grid_parallel 
    PRIVATE 
    Threads::Threads
    m
)

# OpenMP linking
if(ENABLE_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(grid_parallel PRIVATE OpenMP::OpenMP_C)
endif()

# CUDA linking
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(grid_parallel PRIVATE 
        CUDA::cudart 
        CUDA::cublas
    )
endif()

# Testing
if(ENABLE_TESTING)
    enable_testing()
    
    # Find test sources
    file(GLOB TEST_SOURCES "test/*.c")
    
    foreach(TEST_SOURCE ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE grid_parallel)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        
        # Set test properties
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test
        )
    endforeach()
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    file(GLOB BENCHMARK_SOURCES "test/benchmark_*.c")
    
    foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
        get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
        add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE})
        target_link_libraries(${BENCHMARK_NAME} PRIVATE grid_parallel)
        
        set_target_properties(${BENCHMARK_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark
        )
    endforeach()
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS grid_parallel
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/grid_parallel
)

# pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/grid_parallel.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/grid_parallel.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grid_parallel.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# CPack configuration
set(CPACK_PACKAGE_NAME "grid-parallel")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "SuperDARN Grid Parallel Processing Library")
set(CPACK_PACKAGE_VENDOR "SuperDARN")
set(CPACK_PACKAGE_CONTACT "support@superdarn.org")

if(PLATFORM_WINDOWS)
    set(CPACK_GENERATOR "ZIP;NSIS")
elseif(PLATFORM_MACOS)
    set(CPACK_GENERATOR "ZIP;DragNDrop")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libgomp1")
    set(CPACK_RPM_PACKAGE_REQUIRES "glibc, libgomp")
endif()

include(CPack)

# Custom targets
add_custom_target(format
    COMMAND clang-format -i ${C_SOURCES} ${HEADERS}
    COMMENT "Formatting source code"
)

add_custom_target(analyze
    COMMAND cppcheck --enable=all --std=c99 ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/include/
    COMMENT "Running static analysis"
)

# Print configuration summary
message(STATUS "Grid Parallel Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP: ${ENABLE_OPENMP}")
message(STATUS "  CUDA: ${ENABLE_CUDA}")
message(STATUS "  AVX2: ${ENABLE_AVX2}")
message(STATUS "  Testing: ${ENABLE_TESTING}")
message(STATUS "  Benchmarks: ${ENABLE_BENCHMARKS}")
message(STATUS "  Shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "  LTO: ${ENABLE_LTO}")
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
