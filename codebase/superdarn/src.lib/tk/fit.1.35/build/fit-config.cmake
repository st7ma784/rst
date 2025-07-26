
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was fit-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

# Find required dependencies
find_dependency(Threads)

if(ON)
  find_package(CUDAToolkit REQUIRED)
endif()

# Include the targets file
include("${CMAKE_CURRENT_LIST_DIR}/superdarn_fit-targets.cmake")

# Add include directories
get_target_property(FIT_INCLUDE_DIRS superdarn_fit_
  INTERFACE_INCLUDE_DIRECTORIES
)

# Add library directories
get_target_property(FIT_LIBRARY_DIRS superdarn_fit_
  INTERFACE_LINK_DIRECTORIES
)

# Add link libraries
get_target_property(FIT_LIBRARIES superdarn_fit_
  INTERFACE_LINK_LIBRARIES
)

# Create imported targets for easier use
add_library(superdarn_fit::superdarn_fit ALIAS superdarn_fit_)

# Version information
set(superdarn_fit_VERSION )

# Handle standard args
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(superdarn_fit
  REQUIRED_VARS superdarn_fit_LIBRARIES superdarn_fit_INCLUDE_DIRS
  VERSION_VAR superdarn_fit_VERSION
)
