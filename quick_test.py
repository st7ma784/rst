#!/usr/bin/env python3
"""
Quick test to validate SuperDARN testing infrastructure
"""

import os
import json
from pathlib import Path
import subprocess

def count_components():
    """Count SuperDARN components available for testing"""
    
    print("🔍 Scanning SuperDARN components...")
    
    lib_dir = Path("codebase/superdarn/src.lib/tk")
    bin_dir = Path("codebase/superdarn/src.bin/tk/tool")
    
    libraries = []
    binaries = []
    
    if lib_dir.exists():
        libraries = [d.name for d in lib_dir.iterdir() if d.is_dir()]
        print(f"📚 Found {len(libraries)} libraries:")
        for lib in sorted(libraries):
            print(f"   - {lib}")
    
    if bin_dir.exists():
        binaries = [d.name for d in bin_dir.iterdir() if d.is_dir()]
        print(f"🔧 Found {len(binaries)} binaries/tools:")
        for bin_tool in sorted(binaries):
            print(f"   - {bin_tool}")
    
    total = len(libraries) + len(binaries)
    print(f"\n📊 Total testable components: {total}")
    
    return {
        "libraries": libraries,
        "binaries": binaries,
        "total": total
    }

def test_key_components():
    """Test a few key components to verify the infrastructure"""
    
    print("\n🧪 Testing key components...")
    
    # Test FitACF v3.0 (most important)
    fitacf_dir = Path("codebase/superdarn/src.lib/tk/fitacf_v3.0")
    if fitacf_dir.exists():
        print("✅ FitACF v3.0 found")
        makefile = fitacf_dir / "makefile"
        if makefile.exists():
            print("✅ FitACF v3.0 makefile found")
        else:
            print("⚠️  FitACF v3.0 makefile not found")
    else:
        print("❌ FitACF v3.0 not found")
    
    # Test fit_speck_removal tools
    speck_orig = Path("codebase/superdarn/src.bin/tk/tool/fit_speck_removal.1.0")
    speck_opt = Path("codebase/superdarn/src.bin/tk/tool/fit_speck_removal_optimized.1.0")
    
    if speck_orig.exists():
        print("✅ Original speck removal tool found")
    if speck_opt.exists():
        print("✅ Optimized speck removal tool found")
    
    # Check test results directory
    results_dir = Path("test-results")
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)
        print("📁 Created test-results directory")
    else:
        print("✅ Test results directory exists")

def generate_component_summary():
    """Generate a summary of all components ready for testing"""
    
    components = count_components()
    test_key_components()
    
    # Create summary JSON
    summary = {
        "timestamp": "2025-05-30",
        "total_components": components["total"],
        "libraries": len(components["libraries"]),
        "binaries": len(components["binaries"]),
        "component_list": {
            "libraries": components["libraries"],
            "binaries": components["binaries"]
        },
        "infrastructure_status": "ready"
    }
    
    # Save summary
    summary_path = Path("test-results/component_summary.json")
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Component summary saved to: {summary_path}")
    
    return summary

if __name__ == "__main__":
    print("🚀 SuperDARN Quick Infrastructure Test")
    print("=" * 50)
    
    summary = generate_component_summary()
    
    print("\n🎉 Quick test completed!")
    print(f"Ready to test {summary['total_components']} SuperDARN components")
    print("\nNext steps:")
    print("1. Run comprehensive test suite: ./scripts/superdarn_test_suite.sh")
    print("2. Run detailed FitACF tests: ./scripts/test_fitacf_comprehensive.sh")
    print("3. Generate dashboard: python scripts/generate_comprehensive_dashboard.py")
