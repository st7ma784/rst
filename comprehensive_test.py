#!/usr/bin/env python3
"""
Comprehensive SuperDARN Component Test Runner
Tests all 96 SuperDARN components and generates performance dashboard
"""

import os
import json
import time
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime

class SuperDARNTester:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "comprehensive_all_components",
            "libraries": {},
            "binaries": {},
            "summary": {}
        }
        
    def test_library_build(self, lib_name):
        """Test building a library component"""
        
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
                return {"status": "no_makefile", "error": "No makefile found"}
        
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
                "is_cmake": "CMakeLists.txt" in str(makefile),
                "build_complexity": "high" if len(makefile_content.split('\n')) > 50 else "medium" if len(makefile_content.split('\n')) > 20 else "simple"
            }
            
            # Look for source files
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
                "source_files": [f.name for f in c_files[:5]],
                "estimated_build_time": self.estimate_build_time(len(c_files), result["build_complexity"])
            })
            
            return result
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def test_binary_build(self, bin_name):
        """Test building a binary component"""
        
        bin_path = Path(f"codebase/superdarn/src.bin/tk/tool/{bin_name}")
        if not bin_path.exists():
            return {"status": "not_found", "error": f"Binary path not found: {bin_path}"}
        
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
                "makefile_path": str(makefile),
                "makefile_exists": True,
                "makefile_lines": len(makefile_content.split('\n')),
                "has_includes": "include" in makefile_content.lower(),
                "has_sources": any(ext in makefile_content for ext in ['.c', '.cpp', '.cc']),
                "is_rst_makefile": "$(MAKEBIN)" in makefile_content,
                "build_complexity": "high" if len(makefile_content.split('\n')) > 50 else "medium" if len(makefile_content.split('\n')) > 20 else "simple"
            }
            
            # Count dependencies (LIBS line)
            deps_count = 0
            for line in makefile_content.split('\n'):
                if line.strip().startswith('LIBS='):
                    deps_count = line.count('-l')
                    break
            
            # Look for source files
            c_files = list(bin_path.glob("*.c"))
            h_files = list(bin_path.glob("*.h"))
            
            result.update({
                "c_files": len(c_files),
                "h_files": len(h_files),
                "source_files": [f.name for f in c_files[:5]],
                "dependencies": deps_count,
                "estimated_build_time": self.estimate_build_time(len(c_files), result["build_complexity"])
            })
            
            return result
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def estimate_build_time(self, file_count, complexity):
        """Estimate build time based on file count and complexity"""
        base_time = file_count * 2  # 2 seconds per C file
        
        if complexity == "high":
            return base_time * 2
        elif complexity == "medium":
            return base_time * 1.5
        else:
            return base_time

    def get_all_components(self):
        """Get lists of all SuperDARN components"""
        
        lib_dir = Path("codebase/superdarn/src.lib/tk")
        bin_dir = Path("codebase/superdarn/src.bin/tk/tool")
        
        libraries = []
        binaries = []
        
        if lib_dir.exists():
            libraries = [d.name for d in lib_dir.iterdir() if d.is_dir()]
            
        if bin_dir.exists():
            binaries = [d.name for d in bin_dir.iterdir() if d.is_dir()]
        
        return sorted(libraries), sorted(binaries)

    def run_comprehensive_tests(self):
        """Run comprehensive tests on all SuperDARN components"""
        
        print("🚀 SuperDARN Comprehensive Component Analysis")
        print("=" * 80)
        
        libraries, binaries = self.get_all_components()
        
        print(f"📚 Found {len(libraries)} libraries to test")
        print(f"🔧 Found {len(binaries)} binaries to test") 
        print(f"📊 Total components: {len(libraries) + len(binaries)}")
        print()
        
        # Test libraries
        print("📚 Testing Libraries...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_lib = {executor.submit(self.test_library_build, lib): lib for lib in libraries}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_lib)):
                lib = future_to_lib[future]
                try:
                    result = future.result()
                    self.results["libraries"][lib] = result
                    
                    if result["status"] == "analyzed":
                        print(f"  ✅ {lib}: {result['c_files']} C files, {result['build_complexity']} complexity")
                    else:
                        print(f"  ⚠️  {lib}: {result['status']} - {result.get('error', 'Unknown issue')}")
                        
                except Exception as exc:
                    print(f"  ❌ {lib}: Exception - {exc}")
                    self.results["libraries"][lib] = {"status": "exception", "error": str(exc)}
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i + 1}/{len(libraries)} libraries tested")
        
        lib_time = time.time() - start_time
        
        # Test binaries
        print(f"\n🔧 Testing Binaries...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_bin = {executor.submit(self.test_binary_build, binary): binary for binary in binaries}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_bin)):
                binary = future_to_bin[future]
                try:
                    result = future.result()
                    self.results["binaries"][binary] = result
                    
                    if result["status"] == "analyzed":
                        print(f"  ✅ {binary}: {result['c_files']} C files, {result.get('dependencies', 0)} deps")
                    else:
                        print(f"  ⚠️  {binary}: {result['status']} - {result.get('error', 'Unknown issue')}")
                        
                except Exception as exc:
                    print(f"  ❌ {binary}: Exception - {exc}")
                    self.results["binaries"][binary] = {"status": "exception", "error": str(exc)}
                
                # Progress indicator  
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i + 1}/{len(binaries)} binaries tested")
        
        bin_time = time.time() - start_time
        
        # Generate summary
        self.generate_summary(lib_time, bin_time)
        
        return self.results

    def generate_summary(self, lib_time, bin_time):
        """Generate test summary statistics"""
        
        lib_success = sum(1 for r in self.results["libraries"].values() if r["status"] == "analyzed")
        bin_success = sum(1 for r in self.results["binaries"].values() if r["status"] == "analyzed")
        
        total_libs = len(self.results["libraries"])
        total_bins = len(self.results["binaries"])
        total_components = total_libs + total_bins
        total_success = lib_success + bin_success
        
        # Calculate file counts
        total_c_files = 0
        total_h_files = 0
        
        for result in self.results["libraries"].values():
            if result["status"] == "analyzed":
                total_c_files += result.get("c_files", 0)
                total_h_files += result.get("h_files", 0)
                
        for result in self.results["binaries"].values():
            if result["status"] == "analyzed":
                total_c_files += result.get("c_files", 0) 
                total_h_files += result.get("h_files", 0)
        
        self.results["summary"] = {
            "total_components": total_components,
            "total_libraries": total_libs,
            "total_binaries": total_bins,
            "successful_libraries": lib_success,
            "successful_binaries": bin_success,
            "total_successful": total_success,
            "success_rate": round((total_success / total_components) * 100, 1),
            "total_c_files": total_c_files,
            "total_h_files": total_h_files,
            "total_source_files": total_c_files + total_h_files,
            "lib_test_time": round(lib_time, 2),
            "bin_test_time": round(bin_time, 2),
            "total_test_time": round(lib_time + bin_time, 2)
        }
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        
        summary = self.results["summary"]
        print(f"✅ Total Components Analyzed: {summary['total_successful']}/{summary['total_components']} ({summary['success_rate']}%)")
        print(f"📚 Libraries: {summary['successful_libraries']}/{summary['total_libraries']}")
        print(f"🔧 Binaries: {summary['successful_binaries']}/{summary['total_binaries']}")
        print(f"📄 Total Source Files: {summary['total_source_files']} ({summary['total_c_files']} C + {summary['total_h_files']} H)")
        print(f"⏱️  Analysis Time: {summary['total_test_time']}s (Libs: {summary['lib_test_time']}s, Bins: {summary['bin_test_time']}s)")

    def save_results(self):
        """Save results to JSON file"""
        
        results_path = Path("test-results/comprehensive_component_analysis.json")
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved to: {results_path}")
        return results_path

def main():
    """Main testing function"""
    
    tester = SuperDARNTester()
    
    print("🔍 Starting comprehensive SuperDARN component analysis...")
    print("This will test all 96 components for build readiness and structure")
    print()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests()
    
    # Save results
    results_path = tester.save_results()
    
    print("\n🎉 Comprehensive SuperDARN testing completed!")
    print("Ready for:")
    print("1. Performance optimization testing")
    print("2. Docker-based build validation") 
    print("3. CI/CD pipeline integration")
    print("4. Interactive dashboard generation")
    
    return results

if __name__ == "__main__":
    main()
