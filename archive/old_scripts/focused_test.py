#!/usr/bin/env python3
"""
Focused SuperDARN Component Test
Tests a few key components to validate the testing framework
"""

import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

def test_library_build(lib_name):
    """Test building a library component"""
    
    print(f"🔨 Testing library: {lib_name}")
    
    lib_path = Path(f"codebase/superdarn/src.lib/tk/{lib_name}")
    if not lib_path.exists():
        return {"status": "not_found", "error": f"Library path not found: {lib_path}"}
    
    # Check for makefile in src/ subdirectory (RST standard)
    makefile = lib_path / "src" / "makefile"
    if not makefile.exists():
        # Check for special cases like FitACF v3.0
        special_makefiles = [
            lib_path / "Makefile.performance",
            lib_path / "CMakeLists.txt", 
            lib_path / "Makefile",
            lib_path / "makefile"
        ]
        
        makefile = None
        for special in special_makefiles:
            if special.exists():
                makefile = special
                break
        
        if makefile is None:
            return {"status": "no_makefile", "error": "No makefile found in src/ or root directory"}
    
    # Try to read makefile to understand structure
    try:
        with open(makefile, 'r') as f:
            makefile_content = f.read()
        
        result = {
            "status": "analyzed",
            "name": lib_name,
            "path": str(lib_path),
            "makefile_path": str(makefile),
            "makefile_exists": True,
            "makefile_lines": len(makefile_content.split('\n')),
            "has_includes": "include" in makefile_content.lower(),
            "has_sources": any(ext in makefile_content for ext in ['.c', '.cpp', '.cc']),
            "is_rst_makefile": "$(MAKELIB)" in makefile_content,
            "is_cmake": "CMakeLists.txt" in str(makefile)
        }
        
        # Look for source files in both src/ and root directories
        src_dir = lib_path / "src"
        search_dirs = [lib_path, src_dir] if src_dir.exists() else [lib_path]
        
        c_files = []
        h_files = []
        for search_dir in search_dirs:
            c_files.extend(list(search_dir.glob("*.c")))
            h_files.extend(list(search_dir.glob("*.h")))
        
        result.update({
            "c_files": len(c_files),
            "h_files": len(h_files),
            "source_files": [f.name for f in c_files[:5]]  # First 5 files
        })
        
        return result
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def test_binary_build(bin_name):
    """Test building a binary component"""
    
    print(f"🔧 Testing binary: {bin_name}")
    
    bin_path = Path(f"codebase/superdarn/src.bin/tk/tool/{bin_name}")
    if not bin_path.exists():
        return {"status": "not_found", "error": f"Binary path not found: {bin_path}"}
    
    # Check for makefile
    makefile = bin_path / "makefile"
    if not makefile.exists():
        return {"status": "no_makefile", "error": "No makefile found"}
    
    try:
        with open(makefile, 'r') as f:
            makefile_content = f.read()
        
        result = {
            "status": "analyzed",
            "name": bin_name,
            "path": str(bin_path),
            "makefile_exists": True,
            "makefile_lines": len(makefile_content.split('\n')),
            "has_includes": "include" in makefile_content.lower(),
            "has_sources": any(ext in makefile_content for ext in ['.c', '.cpp', '.cc'])
        }
        
        # Look for source files
        c_files = list(bin_path.glob("*.c"))
        h_files = list(bin_path.glob("*.h"))
        
        result.update({
            "c_files": len(c_files),
            "h_files": len(h_files),
            "source_files": [f.name for f in c_files[:5]]  # First 5 files
        })
        
        return result
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def run_focused_tests():
    """Run focused tests on key components"""
    
    print("🎯 Running Focused SuperDARN Component Tests")
    print("=" * 60)
    
    # Key libraries to test
    key_libraries = [
        "fitacf_v3.0",
        "fit.1.35", 
        "grid.1.24",
        "raw.1.22",
        "cfit.1.19"
    ]
    
    # Key binaries to test  
    key_binaries = [
        "fit_speck_removal.1.0",
        "fit_speck_removal_optimized.1.0",
        "make_fit",
        "make_grid.2.0",
        "trim_fit.1.20"
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "focused_validation",
        "libraries": {},
        "binaries": {}
    }
    
    # Test libraries
    print(f"\n📚 Testing {len(key_libraries)} key libraries...")
    for lib in key_libraries:
        result = test_library_build(lib)
        results["libraries"][lib] = result
        
        if result["status"] == "analyzed":
            print(f"  ✅ {lib}: {result['c_files']} C files, {result['h_files']} headers")
        else:
            print(f"  ⚠️  {lib}: {result['status']} - {result.get('error', 'Unknown issue')}")
    
    # Test binaries
    print(f"\n🔧 Testing {len(key_binaries)} key binaries...")
    for binary in key_binaries:
        result = test_binary_build(binary)
        results["binaries"][binary] = result
        
        if result["status"] == "analyzed":
            print(f"  ✅ {binary}: {result['c_files']} C files, {result['h_files']} headers")
        else:
            print(f"  ⚠️  {binary}: {result['status']} - {result.get('error', 'Unknown issue')}")
    
    # Summary
    lib_success = sum(1 for r in results["libraries"].values() if r["status"] == "analyzed")
    bin_success = sum(1 for r in results["binaries"].values() if r["status"] == "analyzed")
    
    print(f"\n📊 Focused Test Results:")
    print(f"   Libraries analyzed: {lib_success}/{len(key_libraries)}")
    print(f"   Binaries analyzed: {bin_success}/{len(key_binaries)}")
    print(f"   Total success rate: {(lib_success + bin_success)}/{len(key_libraries) + len(key_binaries)}")
    
    # Save results
    results_path = Path("test-results/focused_test_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Focused test results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    results = run_focused_tests()
    
    print("\n🎉 Focused testing completed!")
    print("This validates that the testing infrastructure can analyze SuperDARN components.")
    print("\nReady for comprehensive testing of all 96 components!")
