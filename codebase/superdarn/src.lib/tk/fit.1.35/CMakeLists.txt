cmake_minimum_required(VERSION 3.10)
project(superdarn_fit LANGUAGES C CXX CUDA)

# Set C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Options for different build types and optimizations
option(BUILD_ORIGINAL "Build original implementation" ON)
option(ENABLE_AVX "Enable AVX/AVX2/AVX-512 optimizations" ON)
option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../..  # For SuperDARN includes
)

# Set common compiler flags
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../..  # For SuperDARN includes
)

# Set compiler warnings and basic flags
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
  
  # Find and configure OpenMP if requested
  if(OpenMP_FOUND)
    find_package(OpenMP REQUIRED)
    add_compile_options(${OpenMP_C_FLAGS})
  endif()
  
  # Set optimization and debug flags based on build type
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -DDEBUG)
  else()
    add_compile_options(-O3 -DNDEBUG)
  endif()
  
  # Check for AVX support if explicitly enabled
  if(ENABLE_AVX)
    include(CheckCCompilerFlag)
    # Check for AVX2 first (more widely supported)
    check_c_compiler_flag(-mavx2 HAS_AVX2)
    if(HAS_AVX2)
      message(STATUS "AVX2 support detected and enabled")
      add_compile_options(-mavx2 -mfma)
      # Only check for AVX-512 if explicitly requested
      if(ENABLE_AVX512)
        check_c_compiler_flag(-mavx512f HAS_AVX512)
        if(HAS_AVX512)
          message(STATUS "AVX-512 support detected and enabled")
          add_compile_options(-mavx512f -mavx512cd)
        else()
          message(STATUS "AVX-512 not available, using AVX2")
        endif()
      endif()
    else()
      message(STATUS "AVX2 not available, using SSE4.2")
      add_compile_options(-msse4.2)
    endif()
  else()
    message(STATUS "AVX optimizations disabled")
  endif()
endif()

# CUDA configuration
if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  
  # Set CUDA architecture flags
  set(CUDA_ARCH "70;75;80"  # Default architectures (Volta, Turing, Ampere)
    CACHE STRING "List of target CUDA architectures")
  
  # Generate CUDA flags for each architecture
  foreach(ARCH ${CUDA_ARCH})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ARCH},code=sm_${ARCH}")
  endforeach()
  
  # Add common CUDA flags
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xcompiler ${COMMON_FLAGS}")
  
  # Add CUDA include directories
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  
  message(STATUS "CUDA support enabled")
endif()

# Source files for the library
set(SOURCES
  src/fit.c
  src/fitscan.c
  src/fitcfit.c
  src/fitread.c
  src/fitwrite.c
  src/fitseek.c
  src/fitinx.c
  src/stubs.c  # Add stubs for missing dependencies
)

# Source files for the optimized library
set(FIT_OPTIMIZED_SOURCES
  src/fit.c
  src/fitread.c
  src/fitwrite.c
  src/fitseek.c
  src/fitinx.c
  src/fitcfit.c
  src/fitcfit_optimized.c
  src/stubs.c
)

# Set up sources for optimized build
set(OPTIMIZED_SOURCES ${FIT_OPTIMIZED_SOURCES})

# Add AVX optimizations if enabled
if(ENABLE_AVX)
  # Add AVX-specific source files if they exist
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/avx")
    file(GLOB AVX_SOURCES "src/avx/*.c")
    if(AVX_SOURCES)
      list(APPEND OPTIMIZED_SOURCES ${AVX_SOURCES})
      set(COMPILE_OPTIONS ${COMPILE_OPTIONS} -mavx2 -mfma)
    endif()
  endif()
endif()

# Create optimized library
add_library(fit_optimized STATIC ${OPTIMIZED_SOURCES})

# Set target properties
target_include_directories(fit_optimized
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set compiler flags
target_compile_options(fit_optimized
  PRIVATE
    ${COMMON_FLAGS}
    ${COMPILE_OPTIONS}
)

# Link libraries
target_link_libraries(fit_optimized
  PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
)

# Add CUDA support if enabled
if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  
  # Add CUDA source files if they exist
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")
    file(GLOB CUDA_SOURCES "src/cuda/*.cu")
    if(CUDA_SOURCES)
      target_sources(fit_optimized PRIVATE ${CUDA_SOURCES})
    endif()
  endif()
  
  # Set CUDA compilation flags
  set_target_properties(fit_optimized PROPERTIES
    CUDA_ARCHITECTURES "native"
    CUDA_SEPARABLE_COMPILATION ON
  )
  
  target_compile_options(fit_optimized
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
  
  target_link_libraries(fit_optimized PRIVATE CUDA::cudart CUDA::cufft)
  target_compile_definitions(fit_optimized PRIVATE USE_OPTIMIZATIONS)
endif()

# Create original version if requested
if(BUILD_ORIGINAL)
  add_library(fit_original STATIC ${SOURCES})
  target_compile_definitions(fit_original PRIVATE ORIGINAL_IMPLEMENTATION)
  set_target_properties(fit_original PROPERTIES OUTPUT_NAME "fit_original")
  
  # Set target properties for original version
  target_include_directories(fit_original
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  
  target_compile_options(fit_original
    PRIVATE
      ${COMMON_FLAGS}
  )
  
  target_link_libraries(fit_original
    PRIVATE
      ${CMAKE_THREAD_LIBS_INIT}
  )
endif()

# Set the default library to optimized version
if(ENABLE_AVX OR ENABLE_CUDA)
  add_library(fit ALIAS fit_optimized)
else()
  add_library(fit ALIAS fit_original)
endif()

# Build tests if requested
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Build benchmarks if requested
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS fit_original fit_optimized
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Only configure and install package files if we have targets to export
if(TARGET fit_optimized OR TARGET fit_original)
  # Package configuration
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fit-config-version.cmake"
    VERSION 1.35
    COMPATIBILITY SameMajorVersion
  )

  # Create a list of targets to export
  set(EXPORT_TARGETS)
  if(TARGET fit_optimized)
    list(APPEND EXPORT_TARGETS fit_optimized)
  endif()
  if(TARGET fit_original)
    list(APPEND EXPORT_TARGETS fit_original)
  endif()
  
  # Only proceed if we have targets to export
  if(EXPORT_TARGETS)
    # Create the export set
    install(TARGETS ${EXPORT_TARGETS}
      EXPORT fit-targets
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin
    )
    
    # Configure and install the package config file
    configure_package_config_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fit-config.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/fit-config.cmake
      INSTALL_DESTINATION lib/cmake/fit
    )

    # Install the config and version files
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/fit-config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/fit-config-version.cmake"
      DESTINATION lib/cmake/fit
    )
    
    # Install the export set
    install(EXPORT fit-targets
      FILE fit-targets.cmake
      NAMESPACE fit::
      DESTINATION lib/cmake/fit
    )
  endif()
endif()

# Add uninstall target
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY
)

add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)
