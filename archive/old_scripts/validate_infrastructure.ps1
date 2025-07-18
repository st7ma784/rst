# SuperDARN Performance Testing Infrastructure Validation
# PowerShell version for Windows

Write-Host "🧪 SuperDARN Performance Testing Infrastructure Validation" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check Docker setup
Write-Host "`n📋 Checking Docker configuration..." -ForegroundColor Yellow
if (Test-Path "dockerfile.fitacf") {
    Write-Host "✅ dockerfile.fitacf found" -ForegroundColor Green
} else {
    Write-Host "❌ dockerfile.fitacf missing" -ForegroundColor Red
    exit 1
}

if (Test-Path "docker-compose.yml") {
    Write-Host "✅ docker-compose.yml found" -ForegroundColor Green
} else {
    Write-Host "❌ docker-compose.yml missing" -ForegroundColor Red
    exit 1
}

# Check GitHub Actions workflow
Write-Host "`n📋 Checking GitHub Actions workflow..." -ForegroundColor Yellow
if (Test-Path ".github\workflows\superdarn-performance-tests.yml") {
    Write-Host "✅ GitHub Actions workflow found" -ForegroundColor Green
} else {
    Write-Host "❌ GitHub Actions workflow missing" -ForegroundColor Red
    exit 1
}

# Check scripts directory
Write-Host "`n📋 Checking performance testing scripts..." -ForegroundColor Yellow
$scriptsDir = "scripts"
$requiredScripts = @(
    "generate_test_fitacf_data.sh",
    "generate_performance_dashboard.py",
    "compare_performance.py",
    "regression_check.py"
)

foreach ($script in $requiredScripts) {
    if (Test-Path "$scriptsDir\$script") {
        Write-Host "✅ $script found" -ForegroundColor Green
    } else {
        Write-Host "❌ $script missing" -ForegroundColor Red
        exit 1
    }
}

# Check key source directories
Write-Host "`n📋 Checking SuperDARN source code structure..." -ForegroundColor Yellow
$fitacfDir = "codebase\superdarn\src.lib\tk\fitacf_v3.0"
$speckDir = "codebase\superdarn\src.bin\tk\tool\fit_speck_removal.1.0"

if (Test-Path $fitacfDir) {
    Write-Host "✅ FitACF v3.0 source directory found" -ForegroundColor Green
    if (Test-Path "$fitacfDir\src\makefile_standalone") {
        Write-Host "✅ FitACF standalone makefile found" -ForegroundColor Green
    } else {
        Write-Host "⚠️ FitACF standalone makefile not found (may need to be created)" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ FitACF v3.0 source directory missing" -ForegroundColor Red
}

if (Test-Path $speckDir) {
    Write-Host "✅ fit_speck_removal source directory found" -ForegroundColor Green
    if (Test-Path "$speckDir\makefile") {
        Write-Host "✅ fit_speck_removal makefile found" -ForegroundColor Green
    } else {
        Write-Host "⚠️ fit_speck_removal makefile not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ fit_speck_removal source directory missing" -ForegroundColor Red
}

# Check test results directory
Write-Host "`n📋 Checking test infrastructure..." -ForegroundColor Yellow
if (Test-Path "test-results") {
    Write-Host "✅ test-results directory found" -ForegroundColor Green
} else {
    Write-Host "ℹ️ Creating test-results directory..." -ForegroundColor Blue
    New-Item -ItemType Directory -Path "test-results" | Out-Null
    Write-Host "✅ test-results directory created" -ForegroundColor Green
}

# Test Python environment
Write-Host "`n📋 Checking Python environment..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    
    # Check for required Python packages
    $pythonPackages = @("pandas", "matplotlib", "seaborn", "numpy", "jinja2")
    $missingPackages = @()
    
    foreach ($package in $pythonPackages) {
        $importTest = python -c "import $package" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Python package '$package' available" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Python package '$package' missing" -ForegroundColor Yellow
            $missingPackages += $package
        }
    }
    
    if ($missingPackages.Count -gt 0) {
        Write-Host "ℹ️ To install missing packages, run:" -ForegroundColor Blue
        Write-Host "   pip install $($missingPackages -join ' ')" -ForegroundColor Blue
    }
} catch {
    Write-Host "❌ Python not found" -ForegroundColor Red
    exit 1
}

Write-Host "`n🎉 Infrastructure validation completed!" -ForegroundColor Green

Write-Host "`n📊 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Build Docker image: docker build -f dockerfile.fitacf -t superdarn-fitacf ." -ForegroundColor White
Write-Host "   2. Run tests: docker-compose up superdarn-test" -ForegroundColor White  
Write-Host "   3. Generate dashboard: python scripts\generate_performance_dashboard.py" -ForegroundColor White
Write-Host "   4. Push to GitHub to trigger automated CI/CD pipeline" -ForegroundColor White

Write-Host "`n🔗 Performance dashboard will be available at:" -ForegroundColor Cyan
Write-Host "   - Local: test-results\performance_dashboard.html" -ForegroundColor White
Write-Host "   - GitHub Pages: (configure in repository settings)" -ForegroundColor White
