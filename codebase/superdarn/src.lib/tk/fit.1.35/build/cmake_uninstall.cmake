if(NOT EXISTS "/home/user/rst/codebase/superdarn/src.lib/tk/fit.1.35/build/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: /home/user/rst/codebase/superdarn/src.lib/tk/fit.1.35/build/install_manifest.txt")
endif(NOT EXISTS "/home/user/rst/codebase/superdarn/src.lib/tk/fit.1.35/build/install_manifest.txt")

file(READ "/home/user/rst/codebase/superdarn/src.lib/tk/fit.1.35/build/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" "${files}" files)
list(REVERSE files)

foreach(file ${files})
  message(STATUS "Uninstalling: ${file}")
  if(IS_SYMLINK "${file}" OR EXISTS "${file}")
    exec_program(
      "/usr/bin/cmake" ARGS "-E remove -f \"${file}\""
      OUTPUT_VARIABLE rm_out
      RETURN_VALUE rm_retval
    )
    if(NOT "${rm_retval}" STREQUAL 0)
      message(FATAL_ERROR "Problem when removing '${file}'")
    endif(NOT "${rm_retval}" STREQUAL 0)
  else(IS_SYMLINK "${file}" OR EXISTS "${file}")
    message(STATUS "File '${file}' does not exist.")
  endif(IS_SYMLINK "${file}" OR EXISTS "${file}")
endforeach(file)
