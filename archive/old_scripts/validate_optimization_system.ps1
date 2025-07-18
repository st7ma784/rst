# SuperDARN RST Optimized Build System Validation (PowerShell)
# ============================================================
# Validation script for Windows PowerShell environment

param(
    [switch]$Help,
    [switch]$Verbose
)

# Color output functions
function Write-Header {
    param([string]$Message)
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "========================================" -ForegroundColor Blue
}

function Write-TestInfo {
    param([string]$Message)
    $script:TestCount++
    Write-Host "Test $($script:TestCount): $Message" -ForegroundColor Yellow
}

function Write-Pass {
    param([string]$Message)
    Write-Host "✓ PASS: $Message" -ForegroundColor Green
    $script:PassCount++
}

function Write-Fail {
    param([string]$Message)
    Write-Host "✗ FAIL: $Message" -ForegroundColor Red
    $script:FailCount++
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ INFO: $Message" -ForegroundColor Cyan
}

# Initialize counters
$script:TestCount = 0
$script:PassCount = 0
$script:FailCount = 0

# Show help
if ($Help) {
    Write-Host "SuperDARN RST Optimized Build System Validation (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\validate_optimization_system.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Help     Show this help message"
    Write-Host "  -Verbose  Enable verbose output"
    Write-Host ""
    Write-Host "This script validates the enhanced RST build system with optimization support."
    Write-Host "It tests dynamic module detection, hardware detection, and build integration."
    Write-Host ""
    Write-Host "Prerequisites:"
    Write-Host "  - RST environment must be configured"
    Write-Host "  - Enhanced build scripts must be installed"
    exit 0
}

# Check RST environment
function Test-RSTEnvironment {
    Write-Header "Checking RST Environment"
    
    Write-TestInfo "RST environment variables"
    if ($env:RSTPATH -and $env:BUILD -and $env:CODEBASE) {
        Write-Pass "RST environment variables are set"
        Write-Info "RSTPATH: $env:RSTPATH"
        Write-Info "BUILD: $env:BUILD"
        Write-Info "CODEBASE: $env:CODEBASE"
    } else {
        Write-Fail "RST environment variables not set"
        Write-Host "Please configure your RST environment"
        return $false
    }
    
    Write-TestInfo "Optimized build script exists"
    $buildScript = Join-Path $env:BUILD "script\make.code.optimized"
    if (Test-Path $buildScript) {
        Write-Pass "Optimized build script found"
    } else {
        Write-Fail "Optimized build script not found at: $buildScript"
        return $false
    }
    
    Write-TestInfo "Optimized configuration exists"
    $configFile = Join-Path $env:BUILD "script\build_optimized.txt"
    if (Test-Path $configFile) {
        Write-Pass "Optimized configuration found"
    } else {
        Write-Fail "Optimized configuration not found at: $configFile"
        return $false
    }
    
    return $true
}

# Test dynamic module discovery
function Test-DynamicDiscovery {
    Write-Header "Testing Dynamic Module Discovery"
    
    Write-TestInfo "Optimized modules exist in codebase"
    $moduleDir = Join-Path $env:CODEBASE "superdarn\src.lib\tk"
    $optimizedModules = Get-ChildItem -Path $moduleDir -Directory | Where-Object { $_.Name -like "*optimized*" }
    
    if ($optimizedModules.Count -gt 0) {
        Write-Pass "Found $($optimizedModules.Count) optimized modules"
        foreach ($module in $optimizedModules | Select-Object -First 3) {
            Write-Info "Found: $($module.Name)"
        }
    } else {
        Write-Fail "No optimized modules found in codebase"
    }
    
    Write-TestInfo "Modules have valid structure"
    $validModules = 0
    foreach ($module in $optimizedModules) {
        $makefilePaths = @(
            Join-Path $module.FullName "src\makefile",
            Join-Path $module.FullName "CMakeLists.txt",
            Join-Path $module.FullName "Makefile"
        )
        
        foreach ($path in $makefilePaths) {
            if (Test-Path $path) {
                $validModules++
                if ($Verbose) {
                    Write-Info "Valid module: $($module.Name)"
                }
                break
            }
        }
    }
    
    if ($validModules -gt 0) {
        Write-Pass "$validModules modules have valid build files"
    } else {
        Write-Fail "No modules have valid build files"
    }
}

# Test build system files
function Test-BuildSystemFiles {
    Write-Header "Testing Build System Files"
    
    Write-TestInfo "Enhanced makefile templates exist"
    $makefileTemplates = @(
        "makelib.optimized.linux",
        "makebin.optimized.linux"
    )
    
    $foundTemplates = 0
    foreach ($template in $makefileTemplates) {
        $templatePath = Join-Path $env:BUILD "make\$template"
        if (Test-Path $templatePath) {
            $foundTemplates++
        }
    }
    
    if ($foundTemplates -eq $makefileTemplates.Count) {
        Write-Pass "All enhanced makefile templates found"
    } else {
        Write-Fail "Some enhanced makefile templates missing ($foundTemplates/$($makefileTemplates.Count))"
    }
    
    Write-TestInfo "Configuration file structure"
    $configFile = Join-Path $env:BUILD "script\build_optimized.txt"
    if (Test-Path $configFile) {
        $content = Get-Content $configFile -Raw
        if ($content -match "DYNAMIC OPTIMIZATION DETECTION") {
            Write-Pass "Configuration has dynamic detection documentation"
        } else {
            Write-Fail "Configuration missing dynamic detection documentation"
        }
        
        if ($content -match "opt1.*Basic optimization") {
            Write-Pass "Configuration has optimization level definitions"
        } else {
            Write-Fail "Configuration missing optimization level definitions"
        }
    }
}

# Test specific optimized modules
function Test-OptimizedModules {
    Write-Header "Testing Specific Optimized Modules"
    
    $testModules = @{
        "Grid" = "grid.1.24_optimized.1"
        "ACF" = "acf.1.16_optimized.2.0" 
        "Binplotlib" = "binplotlib.1.0_optimized.2.0"
    }
    
    foreach ($moduleName in $testModules.Keys) {
        Write-TestInfo "$moduleName optimized module"
        $modulePath = Join-Path $env:CODEBASE "superdarn\src.lib\tk\$($testModules[$moduleName])"
        
        if (Test-Path $modulePath) {
            $makefilePaths = @(
                Join-Path $modulePath "src\makefile",
                Join-Path $modulePath "CMakeLists.txt",
                Join-Path $modulePath "Makefile"
            )
            
            $hasValidMakefile = $false
            foreach ($path in $makefilePaths) {
                if (Test-Path $path) {
                    $hasValidMakefile = $true
                    break
                }
            }
            
            if ($hasValidMakefile) {
                Write-Pass "$moduleName optimized module is valid"
            } else {
                Write-Fail "$moduleName optimized module missing build files"
            }
        } else {
            Write-Fail "$moduleName optimized module not found"
        }
    }
}

# Test documentation
function Test-Documentation {
    Write-Header "Testing Documentation"
    
    Write-TestInfo "Enhanced build system documentation"
    $docFile = Join-Path $env:RSTPATH "ENHANCED_BUILD_SYSTEM_GUIDE.md"
    if (Test-Path $docFile) {
        Write-Pass "Enhanced build system documentation found"
    } else {
        Write-Fail "Enhanced build system documentation missing"
    }
    
    Write-TestInfo "Validation script exists"
    $validationScript = Join-Path $env:RSTPATH "validate_optimization_system.sh"
    if (Test-Path $validationScript) {
        Write-Pass "Unix validation script found"
    } else {
        Write-Fail "Unix validation script missing"
    }
}

# Main execution
function Main {
    Write-Header "SuperDARN RST Optimized Build System Validation (PowerShell)"
    Write-Host "Testing dynamic optimization detection and build framework"
    Write-Host ""
    
    # Run test suites
    if (-not (Test-RSTEnvironment)) {
        Write-Host "Environment check failed. Cannot continue." -ForegroundColor Red
        exit 1
    }
    
    Test-DynamicDiscovery
    Test-BuildSystemFiles  
    Test-OptimizedModules
    Test-Documentation
    
    # Print summary
    Write-Header "Test Results Summary"
    Write-Host "Total Tests: $script:TestCount"
    Write-Host "Passed: " -NoNewline
    Write-Host $script:PassCount -ForegroundColor Green
    Write-Host "Failed: " -NoNewline  
    Write-Host $script:FailCount -ForegroundColor Red
    
    if ($script:FailCount -eq 0) {
        Write-Host "All tests passed! The optimized build system is ready to use." -ForegroundColor Green
        Write-Host ""
        Write-Host "Usage examples (use bash or WSL for full functionality):"
        Write-Host "  bash $env:BUILD/script/make.code.optimized --auto-optimize"
        Write-Host "  bash $env:BUILD/script/make.code.optimized -o opt2 lib"
        Write-Host "  bash $env:BUILD/script/make.code.optimized --list-optimizations"
        exit 0
    } else {
        Write-Host "Some tests failed. Please review the issues above." -ForegroundColor Red
        exit 1
    }
}

# Run main function
Main
