#!/bin/bash

# SuperDARN Performance Testing Infrastructure Validation
# This script validates that all components are properly configured

echo "🧪 SuperDARN Performance Testing Infrastructure Validation"
echo "=========================================================="

# Check Docker setup
echo "📋 Checking Docker configuration..."
if [ -f "dockerfile.fitacf" ]; then
    echo "✅ dockerfile.fitacf found"
else
    echo "❌ dockerfile.fitacf missing"
    exit 1
fi

if [ -f "docker-compose.yml" ]; then
    echo "✅ docker-compose.yml found"
else
    echo "❌ docker-compose.yml missing"  
    exit 1
fi

# Check GitHub Actions workflow
echo ""
echo "📋 Checking GitHub Actions workflow..."
if [ -f ".github/workflows/superdarn-performance-tests.yml" ]; then
    echo "✅ GitHub Actions workflow found"
else
    echo "❌ GitHub Actions workflow missing"
    exit 1
fi

# Check scripts directory
echo ""
echo "📋 Checking performance testing scripts..."
SCRIPTS_DIR="scripts"
REQUIRED_SCRIPTS=(
    "generate_test_fitacf_data.sh"
    "generate_performance_dashboard.py"
    "compare_performance.py"
    "regression_check.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ -f "$SCRIPTS_DIR/$script" ]; then
        echo "✅ $script found"
    else
        echo "❌ $script missing"
        exit 1
    fi
done

# Check key source directories
echo ""
echo "📋 Checking SuperDARN source code structure..."
FITACF_DIR="codebase/superdarn/src.lib/tk/fitacf_v3.0"
SPECK_DIR="codebase/superdarn/src.bin/tk/tool/fit_speck_removal.1.0"

if [ -d "$FITACF_DIR" ]; then
    echo "✅ FitACF v3.0 source directory found"
    if [ -f "$FITACF_DIR/src/makefile_standalone" ]; then
        echo "✅ FitACF standalone makefile found"
    else
        echo "⚠️  FitACF standalone makefile not found (may need to be created)"
    fi
else
    echo "❌ FitACF v3.0 source directory missing"
fi

if [ -d "$SPECK_DIR" ]; then
    echo "✅ fit_speck_removal source directory found"
    if [ -f "$SPECK_DIR/makefile" ]; then
        echo "✅ fit_speck_removal makefile found"
    else
        echo "⚠️  fit_speck_removal makefile not found"
    fi
else
    echo "❌ fit_speck_removal source directory missing"
fi

# Check test results directory
echo ""
echo "📋 Checking test infrastructure..."
if [ -d "test-results" ]; then
    echo "✅ test-results directory found"
else
    echo "ℹ️  Creating test-results directory..."
    mkdir -p test-results
    echo "✅ test-results directory created"
fi

# Validate Docker Compose syntax
echo ""
echo "📋 Validating Docker Compose configuration..."
if command -v docker-compose > /dev/null 2>&1; then
    if docker-compose config > /dev/null 2>&1; then
        echo "✅ docker-compose.yml syntax is valid"
    else
        echo "❌ docker-compose.yml has syntax errors"
        exit 1
    fi
else
    echo "⚠️  docker-compose not available, skipping syntax validation"
fi

# Test Python environment
echo ""
echo "📋 Checking Python environment..."
if command -v python > /dev/null 2>&1; then
    echo "✅ Python found: $(python --version)"
    
    # Check for required Python packages
    PYTHON_PACKAGES=("pandas" "matplotlib" "seaborn" "numpy" "jinja2")
    missing_packages=()
    
    for package in "${PYTHON_PACKAGES[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            echo "✅ Python package '$package' available"
        else
            echo "⚠️  Python package '$package' missing"
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo "ℹ️  To install missing packages, run:"
        echo "   pip install ${missing_packages[*]}"
    fi
else
    echo "❌ Python not found"
    exit 1
fi

echo ""
echo "🎉 Infrastructure validation completed!"
echo ""
echo "📊 Next steps:"
echo "   1. Build Docker image: docker build -f dockerfile.fitacf -t superdarn-fitacf ."
echo "   2. Run tests: docker-compose up superdarn-test"
echo "   3. Generate dashboard: python scripts/generate_performance_dashboard.py"
echo "   4. Push to GitHub to trigger automated CI/CD pipeline"
echo ""
echo "🔗 Performance dashboard will be available at:"
echo "   - Local: test-results/performance_dashboard.html"
echo "   - GitHub Pages: (configure in repository settings)"
