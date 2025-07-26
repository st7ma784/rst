#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "fit::fit_optimized" for configuration "Release"
set_property(TARGET fit::fit_optimized APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(fit::fit_optimized PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libfit_optimized.a"
  )

list(APPEND _cmake_import_check_targets fit::fit_optimized )
list(APPEND _cmake_import_check_files_for_fit::fit_optimized "${_IMPORT_PREFIX}/lib/libfit_optimized.a" )

# Import target "fit::fit_original" for configuration "Release"
set_property(TARGET fit::fit_original APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(fit::fit_original PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libfit_original.a"
  )

list(APPEND _cmake_import_check_targets fit::fit_original )
list(APPEND _cmake_import_check_files_for_fit::fit_original "${_IMPORT_PREFIX}/lib/libfit_original.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
