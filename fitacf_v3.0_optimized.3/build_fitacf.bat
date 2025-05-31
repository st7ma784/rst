@echo off
REM build_fitacf.bat - Build script for SuperDARN FitACF v3.0 Array Implementation (Windows)
REM
REM This script automates the build process for both linked list and array
REM implementations with various configuration options on Windows systems.

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_NAME=SuperDARN FitACF v3.0
set BUILD_DIR=build
set SRC_DIR=src
set TEST_DIR=test

REM Default options
set BUILD_LLIST=true
set BUILD_ARRAY=true
set BUILD_TESTS=true
set ENABLE_OPENMP=true
set ENABLE_CUDA=false
set BUILD_TYPE=Release
set VERBOSE=false
set CLEAN_BUILD=false
set RUN_TESTS=false
set USE_MAKEFILE=false

REM Function to show usage
:show_usage
echo Usage: %0 [options]
echo.
echo Options:
echo   -h, --help              Show this help message
echo   -c, --clean             Clean build (remove build directory)
echo   -t, --tests             Build and run tests
echo   -v, --verbose           Verbose output
echo   --debug                 Debug build
echo   --release               Release build (default)
echo   --no-llist              Skip linked list implementation
echo   --no-array              Skip array implementation
echo   --no-openmp             Disable OpenMP
echo   --enable-cuda           Enable CUDA support
echo   --make-only             Use traditional makefile instead of CMake
echo.
echo Examples:
echo   %0                      # Build everything with default settings
echo   %0 --debug -t           # Debug build with tests
echo   %0 --enable-cuda        # Release build with CUDA support
echo   %0 --no-llist --tests   # Array implementation only with tests
goto :eof

REM Function to print status messages
:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM Function to check dependencies
:check_dependencies
call :print_status "Checking dependencies..."

REM Check for C compiler (Visual Studio or MinGW)
where cl >nul 2>&1
if !errorlevel! equ 0 (
    set COMPILER=msvc
    call :print_status "Visual Studio C compiler detected"
) else (
    where gcc >nul 2>&1
    if !errorlevel! equ 0 (
        set COMPILER=gcc
        call :print_status "GCC compiler detected"
    ) else (
        call :print_error "No C compiler found (Visual Studio or MinGW/GCC required)"
        exit /b 1
    )
)

REM Check for make/nmake
if "!COMPILER!"=="msvc" (
    where nmake >nul 2>&1
    if !errorlevel! neq 0 (
        call :print_error "nmake not found"
        exit /b 1
    )
) else (
    where make >nul 2>&1
    if !errorlevel! neq 0 (
        call :print_error "make not found"
        exit /b 1
    )
)

REM Check for CMake
where cmake >nul 2>&1
if !errorlevel! neq 0 (
    call :print_warning "CMake not found, falling back to traditional makefile"
    set USE_MAKEFILE=true
)

REM Check for OpenMP support
if "!ENABLE_OPENMP!"=="true" (
    if "!COMPILER!"=="gcc" (
        echo int main(){return 0;} > test_openmp.c
        gcc -fopenmp test_openmp.c -o test_openmp.exe >nul 2>&1
        if !errorlevel! equ 0 (
            call :print_status "OpenMP support detected"
        ) else (
            call :print_warning "OpenMP not supported by compiler, disabling"
            set ENABLE_OPENMP=false
        )
        del test_openmp.c test_openmp.exe >nul 2>&1
    )
)

REM Check for CUDA
if "!ENABLE_CUDA!"=="true" (
    where nvcc >nul 2>&1
    if !errorlevel! equ 0 (
        call :print_status "CUDA compiler detected"
    ) else (
        call :print_error "CUDA requested but nvcc not found"
        exit /b 1
    )
)

call :print_success "Dependencies check completed"
goto :eof

REM Function to clean build directory
:clean_build
call :print_status "Cleaning build directory..."
if exist "!BUILD_DIR!" (
    rmdir /s /q "!BUILD_DIR!"
    call :print_success "Build directory cleaned"
)

REM Clean object files in src directory
if exist "!SRC_DIR!" (
    pushd "!SRC_DIR!"
    del /q *.o *.obj *.lib *.a test_baseline.exe test_comparison.exe >nul 2>&1
    popd
    call :print_success "Source directory cleaned"
)
goto :eof

REM Function to build with CMake
:build_with_cmake
call :print_status "Building with CMake..."

if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
pushd "!BUILD_DIR!"

REM Configure CMake options
set CMAKE_ARGS=-DCMAKE_BUILD_TYPE=!BUILD_TYPE!

if "!BUILD_LLIST!"=="false" (
    set CMAKE_ARGS=!CMAKE_ARGS! -DBUILD_LLIST_IMPLEMENTATION=OFF
)

if "!BUILD_ARRAY!"=="false" (
    set CMAKE_ARGS=!CMAKE_ARGS! -DBUILD_ARRAY_IMPLEMENTATION=OFF
)

if "!BUILD_TESTS!"=="false" (
    set CMAKE_ARGS=!CMAKE_ARGS! -DBUILD_TESTS=OFF
)

if "!ENABLE_OPENMP!"=="false" (
    set CMAKE_ARGS=!CMAKE_ARGS! -DENABLE_OPENMP=OFF
)

if "!ENABLE_CUDA!"=="true" (
    set CMAKE_ARGS=!CMAKE_ARGS! -DENABLE_CUDA=ON
)

REM Configure
call :print_status "Configuring build..."
if "!VERBOSE!"=="true" (
    cmake .. !CMAKE_ARGS!
) else (
    cmake .. !CMAKE_ARGS! > cmake_config.log 2>&1
)

if !errorlevel! neq 0 (
    call :print_error "CMake configuration failed"
    popd
    exit /b 1
)

REM Build
call :print_status "Compiling..."
if "!VERBOSE!"=="true" (
    cmake --build . --config !BUILD_TYPE!
) else (
    cmake --build . --config !BUILD_TYPE! > cmake_build.log 2>&1
)

if !errorlevel! neq 0 (
    call :print_error "Build failed"
    popd
    exit /b 1
)

popd
call :print_success "CMake build completed"
goto :eof

REM Function to build with traditional makefile
:build_with_makefile
call :print_status "Building with traditional makefile..."

pushd "!SRC_DIR!"

REM Build targets based on options
set MAKE_TARGETS=

if "!BUILD_LLIST!"=="true" (
    set MAKE_TARGETS=!MAKE_TARGETS! fitacf_llist
)

if "!BUILD_ARRAY!"=="true" (
    set MAKE_TARGETS=!MAKE_TARGETS! fitacf_array
)

if "!BUILD_TESTS!"=="true" (
    set MAKE_TARGETS=!MAKE_TARGETS! tests
)

REM Set build type
if "!BUILD_TYPE!"=="Debug" (
    set MAKE_TARGETS=debug !MAKE_TARGETS!
)

REM Choose make command based on compiler
if "!COMPILER!"=="msvc" (
    set MAKE_CMD=nmake /f makefile_array.nmake
) else (
    set MAKE_CMD=make -f makefile_array
)

REM Build
if "!VERBOSE!"=="true" (
    !MAKE_CMD! !MAKE_TARGETS!
) else (
    !MAKE_CMD! !MAKE_TARGETS! > make_build.log 2>&1
)

if !errorlevel! neq 0 (
    call :print_error "Build failed"
    popd
    exit /b 1
)

popd
call :print_success "Makefile build completed"
goto :eof

REM Function to run tests
:run_tests
call :print_status "Running tests..."

if "!USE_MAKEFILE!"=="true" (
    pushd "!SRC_DIR!"
    if exist "test_baseline.exe" (
        call :print_status "Running baseline tests..."
        test_baseline.exe
    )
    if exist "test_comparison.exe" (
        call :print_status "Running comparison tests..."
        test_comparison.exe
    )
    popd
) else (
    pushd "!BUILD_DIR!"
    if exist "test_baseline.exe" (
        call :print_status "Running baseline tests..."
        test_baseline.exe
    )
    if exist "test_comparison.exe" (
        call :print_status "Running comparison tests..."
        test_comparison.exe
    )
    popd
)

call :print_success "Tests completed"
goto :eof

REM Function to show build summary
:show_summary
echo.
echo ===============================================
echo   !PROJECT_NAME! Build Summary
echo ===============================================
echo Build type:             !BUILD_TYPE!
echo Linked list impl:       !BUILD_LLIST!
echo Array impl:             !BUILD_ARRAY!
echo Tests:                  !BUILD_TESTS!
echo OpenMP:                 !ENABLE_OPENMP!
echo CUDA:                   !ENABLE_CUDA!
if "!USE_MAKEFILE!"=="true" (
    echo Build system:           Makefile
) else (
    echo Build system:           CMake
)
echo Compiler:               !COMPILER!
echo ===============================================
echo.
goto :eof

REM Parse command line arguments
:parse_args
if "%1"=="" goto :end_parse

if "%1"=="-h" goto :help
if "%1"=="--help" goto :help
if "%1"=="-c" set CLEAN_BUILD=true& shift& goto :parse_args
if "%1"=="--clean" set CLEAN_BUILD=true& shift& goto :parse_args
if "%1"=="-t" set BUILD_TESTS=true& set RUN_TESTS=true& shift& goto :parse_args
if "%1"=="--tests" set BUILD_TESTS=true& set RUN_TESTS=true& shift& goto :parse_args
if "%1"=="-v" set VERBOSE=true& shift& goto :parse_args
if "%1"=="--verbose" set VERBOSE=true& shift& goto :parse_args
if "%1"=="--debug" set BUILD_TYPE=Debug& shift& goto :parse_args
if "%1"=="--release" set BUILD_TYPE=Release& shift& goto :parse_args
if "%1"=="--no-llist" set BUILD_LLIST=false& shift& goto :parse_args
if "%1"=="--no-array" set BUILD_ARRAY=false& shift& goto :parse_args
if "%1"=="--no-openmp" set ENABLE_OPENMP=false& shift& goto :parse_args
if "%1"=="--enable-cuda" set ENABLE_CUDA=true& shift& goto :parse_args
if "%1"=="--make-only" set USE_MAKEFILE=true& shift& goto :parse_args

call :print_error "Unknown option: %1"
goto :help

:help
call :show_usage
exit /b 0

:end_parse
goto :eof

REM Main build process
:main
echo Building !PROJECT_NAME!
echo =======================

REM Parse arguments
call :parse_args %*

REM Clean if requested
if "!CLEAN_BUILD!"=="true" (
    call :clean_build
)

REM Check dependencies
call :check_dependencies
if !errorlevel! neq 0 exit /b 1

REM Show build configuration
call :show_summary

REM Build
if "!USE_MAKEFILE!"=="true" (
    call :build_with_makefile
) else (
    call :build_with_cmake
)

if !errorlevel! neq 0 exit /b 1

REM Run tests if requested
if "!RUN_TESTS!"=="true" (
    call :run_tests
)

call :print_success "Build process completed successfully!"

REM Show next steps
echo.
echo Next steps:
if "!BUILD_TESTS!"=="true" if "!RUN_TESTS!"=="false" (
    echo   Run tests: %0 --tests
)
if "!USE_MAKEFILE!"=="true" (
    echo   See build outputs in: src/
) else (
    echo   See build outputs in: build/
)
echo   Integration guide: See documentation in docs/

goto :eof

REM Entry point
call :main %*
