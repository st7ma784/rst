#!/usr/bin/env python3
"""
SuperDARN Infrastructure Validation Script
Tests that all components are properly set up for comprehensive testing
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def validate_file_structure():
    """Validate that all required files and directories exist"""
    
    print("🔍 Validating file structure...")
    
    required_files = [
        "docker-compose.yml",
        "dockerfile.fitacf", 
        ".github/workflows/superdarn-performance-tests.yml",
        "scripts/superdarn_test_suite.sh",
        "scripts/test_fitacf_comprehensive.sh",
        "scripts/generate_comprehensive_dashboard.py"
    ]
    
    required_dirs = [
        "codebase/superdarn/src.lib/tk",
        "codebase/superdarn/src.bin/tk",
        "test-results",
        "scripts"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ File structure validation passed")
    return True

def validate_docker_config():
    """Validate Docker configuration"""
    
    print("🐳 Validating Docker configuration...")
    
    try:
        with open("docker-compose.yml", "r") as f:
            content = f.read()
            
        required_services = ["superdarn-dev", "superdarn-test"]
        missing_services = []
        
        for service in required_services:
            if service not in content:
                missing_services.append(service)
        
        if missing_services:
            print(f"❌ Missing Docker services: {missing_services}")
            return False
        
        print("✅ Docker configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Docker validation failed: {e}")
        return False

def validate_github_actions():
    """Validate GitHub Actions workflow"""
    
    print("🔄 Validating GitHub Actions workflow...")
    
    try:
        workflow_path = ".github/workflows/superdarn-performance-tests.yml"
        with open(workflow_path, "r") as f:
            content = f.read()
        
        required_jobs = [
            "build-test-image",
            "fitacf-array-tests", 
            "speck-removal-tests",
            "comprehensive-tests"
        ]
        
        missing_jobs = []
        for job in required_jobs:
            if job not in content:
                missing_jobs.append(job)
        
        if missing_jobs:
            print(f"❌ Missing GitHub Actions jobs: {missing_jobs}")
            return False
        
        print("✅ GitHub Actions workflow validation passed")
        return True
        
    except Exception as e:
        print(f"❌ GitHub Actions validation failed: {e}")
        return False

def count_testable_components():
    """Count how many components we can test"""
    
    print("📊 Counting testable components...")
    
    lib_count = 0
    bin_count = 0
    
    lib_dir = Path("codebase/superdarn/src.lib/tk")
    if lib_dir.exists():
        lib_count = len([d for d in lib_dir.iterdir() if d.is_dir()])
    
    bin_dir = Path("codebase/superdarn/src.bin/tk/tool")
    if bin_dir.exists():
        bin_count = len([d for d in bin_dir.iterdir() if d.is_dir()])
    
    print(f"📚 Libraries found: {lib_count}")
    print(f"🔧 Binaries/tools found: {bin_count}")
    print(f"📈 Total testable components: {lib_count + bin_count}")
    
    return lib_count + bin_count

def test_python_dependencies():
    """Test if Python dependencies are available"""
    
    print("🐍 Testing Python dependencies...")
    
    required_modules = ["json", "os", "sys", "pathlib"]
    optional_modules = ["matplotlib", "pandas", "numpy", "seaborn", "jinja2"]
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if missing_required:
        print(f"❌ Missing required Python modules: {missing_required}")
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional Python modules (will be installed in Docker): {missing_optional}")
    
    print("✅ Python dependencies validation passed")
    return True

def generate_validation_report():
    """Generate a validation report"""
    
    print("\n" + "="*60)
    print("🚀 SUPERDARN TESTING INFRASTRUCTURE VALIDATION REPORT")
    print("="*60)
    
    all_passed = True
    
    # Run all validations
    validations = [
        ("File Structure", validate_file_structure),
        ("Docker Configuration", validate_docker_config),
        ("GitHub Actions", validate_github_actions),
        ("Python Dependencies", test_python_dependencies)
    ]
    
    results = {}
    
    for name, validation_func in validations:
        print(f"\n📋 {name}:")
        try:
            result = validation_func()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {name} validation failed with error: {e}")
            results[name] = False
            all_passed = False
    
    # Count components
    component_count = count_testable_components()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nTotal Components Available for Testing: {component_count}")
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("🚀 Your SuperDARN testing infrastructure is ready!")
        print("\nNext steps:")
        print("1. Run: docker-compose up superdarn-test")
        print("2. View results in: test-results/dashboards/")
        print("3. Push to GitHub to trigger CI/CD pipeline")
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED")
        print("Please fix the issues above before proceeding.")
        return False
    
    return True

def main():
    """Main validation function"""
    
    # Change to project root directory (script is already in project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = generate_validation_report()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
