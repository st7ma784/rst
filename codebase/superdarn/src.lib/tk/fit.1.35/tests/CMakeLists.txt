# Test configuration for fit.1.35 module

# Add include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../..  # For SuperDARN includes
)

# Find GoogleTest
find_package(GTest QUIET)

# Build the profiling tool
add_executable(profile_fit
  profile_fit.c
)

# Set output directory for the executable
set_target_properties(profile_fit PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
)

target_link_libraries(profile_fit
  PRIVATE
    fit_optimized
    m  # For math library
)

# Build the benchmark tool
add_executable(benchmark_fitcfit
  benchmark_fitcfit.c
)

# Set output directory for the benchmark executable
set_target_properties(benchmark_fitcfit PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
)

target_link_libraries(benchmark_fitcfit
  PRIVATE
    fit_optimized
    m  # For math library
)

# Add install target for the profiling tool
install(TARGETS profile_fit
  RUNTIME DESTINATION bin
)

if(GTEST_FOUND)
  # Create a test executable
  add_executable(fit_tests
    test_fit.c
  )
  
  # Link against the library and GTest
  target_link_libraries(fit_tests
    PRIVATE
      fit_optimized  # or fit_original for non-optimized version
      GTest::GTest
      GTest::Main
  )
  
  # Add the test to CTest
  add_test(NAME fit_tests COMMAND fit_tests)
  
  # Enable test discovery for IDEs
  include(GoogleTest)
  gtest_discover_tests(fit_tests)
  
  # Add a custom target for running tests
  add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running tests..."
  )
  
  # Make sure tests are built with the main build
  add_dependencies(fit_tests run_tests)
else()
  message(WARNING "Google Test not found, tests will not be built")
endif()
