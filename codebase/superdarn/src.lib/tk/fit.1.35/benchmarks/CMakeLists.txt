# Benchmark configuration for fit.1.35 module

# Add include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../..  # For SuperDARN includes
)

# Find Google Benchmark
find_package(benchmark QUIET)

if(benchmark_FOUND)
  # Create benchmark executable
  add_executable(fit_benchmarks
    matrix_benchmark.c
    fft_benchmark.c
  )
  
  # Link against the library and benchmark
  target_link_libraries(fit_benchmarks
    PRIVATE
      fit_optimized  # or fit_original for non-optimized version
      benchmark::benchmark
      benchmark::benchmark_main
  )
  
  # Add a custom target for running benchmarks
  add_custom_target(run_benchmarks
    COMMAND fit_benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks..."
    DEPENDS fit_benchmarks
  )
  
  # Make sure benchmarks are built with the main build
  add_dependencies(benchmark run_benchmarks)
else()
  message(WARNING "Google Benchmark not found, benchmarks will not be built")
endif()
