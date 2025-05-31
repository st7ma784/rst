# CMakeLists.txt for SuperDARN FitACF v3.0 Array Implementation
#
# This CMake configuration builds both the original linked list implementation
# and the new array-based implementation with OpenMP and optional CUDA support
#

cmake_minimum_required(VERSION 3.12)

project(SuperDARN_FitACF_v3 
    VERSION 3.0.0
    DESCRIPTION "SuperDARN FitACF with Array-Based Parallelization"
    LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_ARRAY_IMPLEMENTATION "Build array-based implementation" ON)
option(BUILD_LLIST_IMPLEMENTATION "Build linked list implementation" ON)
option(BUILD_TESTS "Build test suite" ON)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
option(ENABLE_PERFORMANCE_PROFILING "Enable performance profiling" OFF)

# Find required packages
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
endif()

# Compiler-specific flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG_ARRAY")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ftree-vectorize -DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DDEBUG_ARRAY")
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/include
)

# Source files for linked list implementation
set(LLIST_SOURCES
    determinations.c
    fitacftoplevel.c
    fitting.c
    leastsquares.c
    llist.c
    preprocessing.c
)

set(LLIST_HEADERS
    ../include/determinations.h
    ../include/fit_structures.h
    ../include/leastsquares.h
    ../include/preprocessing.h
    ../include/fitacftoplevel.h
    ../include/fitting.h
    ../include/llist.h
)

# Source files for array implementation
set(ARRAY_SOURCES
    determinations.c
    fitacftoplevel_array.c
    fitting_array.c
    leastsquares.c
    fit_structures_array.c
    preprocessing_array.c
)

set(ARRAY_HEADERS
    ../include/determinations.h
    ../include/fit_structures_array.h
    ../include/leastsquares.h
    ../include/preprocessing.h
    ../include/fitacftoplevel.h
    ../include/fitting.h
)

# Test source files
set(TEST_SOURCES
    ../test/test_fitacf_comprehensive.c
    ../test/test_array_vs_llist.c
)

# Build linked list implementation
if(BUILD_LLIST_IMPLEMENTATION)
    add_library(fitacf_llist STATIC ${LLIST_SOURCES})
    target_include_directories(fitacf_llist PRIVATE ${PROJECT_SOURCE_DIR}/../include)
    target_compile_definitions(fitacf_llist PRIVATE USE_LLIST_IMPLEMENTATION)
    
    # Install headers
    install(FILES ${LLIST_HEADERS} DESTINATION include/superdarn/fitacf)
    install(TARGETS fitacf_llist DESTINATION lib)
endif()

# Build array implementation
if(BUILD_ARRAY_IMPLEMENTATION)
    add_library(fitacf_array STATIC ${ARRAY_SOURCES})
    target_include_directories(fitacf_array PRIVATE ${PROJECT_SOURCE_DIR}/../include)
    target_compile_definitions(fitacf_array PRIVATE USE_ARRAY_IMPLEMENTATION)
    
    # OpenMP support
    if(ENABLE_OPENMP AND OpenMP_C_FOUND)
        target_link_libraries(fitacf_array PUBLIC OpenMP::OpenMP_C)
        target_compile_definitions(fitacf_array PRIVATE OPENMP_ENABLED)
        message(STATUS "OpenMP enabled for array implementation")
    endif()
    
    # CUDA support
    if(ENABLE_CUDA)
        target_compile_definitions(fitacf_array PRIVATE CUDA_ENABLED)
        target_link_libraries(fitacf_array PUBLIC ${CUDA_LIBRARIES})
        message(STATUS "CUDA enabled for array implementation")
    endif()
    
    # Performance profiling
    if(ENABLE_PERFORMANCE_PROFILING)
        target_compile_definitions(fitacf_array PRIVATE ENABLE_PROFILING)
    endif()
    
    # Install headers
    install(FILES ${ARRAY_HEADERS} DESTINATION include/superdarn/fitacf)
    install(TARGETS fitacf_array DESTINATION lib)
endif()

# Build test suite
if(BUILD_TESTS)
    # Baseline test for linked list implementation
    if(BUILD_LLIST_IMPLEMENTATION)
        add_executable(test_baseline ../test/test_fitacf_comprehensive.c)
        target_link_libraries(test_baseline fitacf_llist)
        target_include_directories(test_baseline PRIVATE ${PROJECT_SOURCE_DIR}/../include)
    endif()
    
    # Comparison test between implementations
    if(BUILD_ARRAY_IMPLEMENTATION AND BUILD_LLIST_IMPLEMENTATION)
        add_executable(test_comparison ../test/test_array_vs_llist.c)
        target_link_libraries(test_comparison fitacf_array fitacf_llist)
        target_include_directories(test_comparison PRIVATE ${PROJECT_SOURCE_DIR}/../include)
        target_compile_definitions(test_comparison PRIVATE USE_ARRAY_IMPLEMENTATION)
        
        if(ENABLE_OPENMP AND OpenMP_C_FOUND)
            target_link_libraries(test_comparison OpenMP::OpenMP_C)
        endif()
    endif()
    
    # Array-only performance test
    if(BUILD_ARRAY_IMPLEMENTATION)
        add_executable(test_array_performance ../test/test_array_vs_llist.c)
        target_link_libraries(test_array_performance fitacf_array)
        target_include_directories(test_array_performance PRIVATE ${PROJECT_SOURCE_DIR}/../include)
        target_compile_definitions(test_array_performance PRIVATE 
            USE_ARRAY_IMPLEMENTATION 
            PERFORMANCE_TEST_ONLY)
        
        if(ENABLE_OPENMP AND OpenMP_C_FOUND)
            target_link_libraries(test_array_performance OpenMP::OpenMP_C)
        endif()
    endif()
endif()

# Custom targets for convenience
add_custom_target(run_tests
    COMMAND $<$<TARGET_EXISTS:test_baseline>:$<TARGET_FILE:test_baseline>>
    COMMAND $<$<TARGET_EXISTS:test_comparison>:$<TARGET_FILE:test_comparison>>
    DEPENDS $<$<TARGET_EXISTS:test_baseline>:test_baseline> 
            $<$<TARGET_EXISTS:test_comparison>:test_comparison>
    COMMENT "Running all tests"
)

add_custom_target(benchmark
    COMMAND $<$<TARGET_EXISTS:test_array_performance>:$<TARGET_FILE:test_array_performance>>
    DEPENDS $<$<TARGET_EXISTS:test_array_performance>:test_array_performance>
    COMMENT "Running performance benchmarks"
)

# Print configuration summary
message(STATUS "")
message(STATUS "SuperDARN FitACF v3.0 Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Linked list implementation: ${BUILD_LLIST_IMPLEMENTATION}")
message(STATUS "  Array implementation: ${BUILD_ARRAY_IMPLEMENTATION}")
message(STATUS "  Test suite: ${BUILD_TESTS}")
message(STATUS "  OpenMP: ${ENABLE_OPENMP}")
message(STATUS "  CUDA: ${ENABLE_CUDA}")
message(STATUS "  Performance profiling: ${ENABLE_PERFORMANCE_PROFILING}")
message(STATUS "")

# Documentation
add_custom_target(doc
    COMMAND echo "SuperDARN FitACF v3.0 Array Implementation"
    COMMAND echo "========================================"
    COMMAND echo ""
    COMMAND echo "Build targets:"
    COMMAND echo "  fitacf_llist - Original linked list implementation"
    COMMAND echo "  fitacf_array - New array-based implementation"
    COMMAND echo "  test_baseline - Test linked list implementation"
    COMMAND echo "  test_comparison - Compare both implementations"
    COMMAND echo "  test_array_performance - Performance test for arrays"
    COMMAND echo ""
    COMMAND echo "Custom targets:"
    COMMAND echo "  run_tests - Execute all test suites"
    COMMAND echo "  benchmark - Run performance benchmarks"
    COMMAND echo "  doc - Show this documentation"
    COMMENT "Showing documentation"
)
