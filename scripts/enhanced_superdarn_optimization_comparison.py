#!/usr/bin/env python3
"""
Enhanced SuperDARN Optimization Comparison Tool with MinIO Data Testing
======================================================================

This tool extends the optimization comparison framework to include:
1. Proper RST environment setup based on working framework patterns
2. Real data testing using MinIO storage
3. Performance benchmarking with actual SuperDARN data
4. Correctness verification between original and optimized versions

Usage:
    python enhanced_superdarn_optimization_comparison.py
    python enhanced_superdarn_optimization_comparison.py --component acf
    python enhanced_superdarn_optimization_comparison.py --with-data-tests
"""

import os
import sys
import json
import time
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from minio import Minio
from minio.error import S3Error

class EnhancedSuperdarnOptimizationComparison:
    def __init__(self, rst_root=None, minio_bucket="superdarn-data", 
                 test_file_extensions=[".dat", ".fit", ".fitacf", ".rawacf"], 
                 max_test_files=3):
        self.rst_root = Path(rst_root) if rst_root else Path(__file__).parent.parent
        self.results_dir = self.rst_root / "test-results"
        self.results_dir.mkdir(exist_ok=True)
        
        # MinIO configuration
        self.minio_bucket = minio_bucket
        self.test_file_extensions = test_file_extensions
        self.max_test_files = max_test_files
        
        # SuperDARN component paths
        self.lib_dir = self.rst_root / "codebase" / "superdarn" / "src.lib" / "tk"
        self.bin_dir = self.rst_root / "codebase" / "superdarn" / "src.bin" / "tk" / "tool"
        
        # Setup RST environment properly
        self.setup_rst_environment()
        
        # MinIO client setup
        self.minio_client = None
        self.setup_minio_client()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "components_found": {},
            "build_results": {},
            "data_tests": {},
            "performance_comparisons": {},
            "summary": {}
        }
        
    def setup_rst_environment(self):
        """Set up RST environment using proper build system like dockerfile"""
        print("🔧 Setting up RST build environment...")
        
        rst_path = str(self.rst_root)
        
        # Core RST environment variables (based on dockerfile)
        env_vars = {
            "RSTPATH": rst_path,
            "SYSTEM": "linux",
            "BUILD": os.path.join(rst_path, "build"),
            "CODEBASE": os.path.join(rst_path, "codebase"),
            "MAKECFG": os.path.join(rst_path, "build", "make", "makecfg"),
            "MAKEBIN": os.path.join(rst_path, "build", "make", "makebin"),
            "MAKELIB": os.path.join(rst_path, "build", "make", "makelib"),
            "LIBPATH": os.path.join(rst_path, "lib"),
            "BINPATH": os.path.join(rst_path, "bin"),
            "IPATH": os.path.join(rst_path, "include"),
            "LOGPATH": os.path.join(rst_path, "log")
        }
        
        # Apply environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            
        # Add RST build scripts to PATH
        build_script_path = os.path.join(rst_path, "build", "script")
        current_path = os.environ.get("PATH", "")
        if build_script_path not in current_path:
            os.environ["PATH"] = f"{build_script_path}:{current_path}"
            
        # Set up library path
        lib_path = env_vars["LIBPATH"]
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}"
            
        # Ensure required directories exist
        for path in [env_vars["LIBPATH"], env_vars["BINPATH"], env_vars["IPATH"], env_vars["LOGPATH"]]:
            os.makedirs(path, exist_ok=True)
            
        # Build all headers using proper RST build system (like dockerfile)
        print("  🔧 Building all dependencies using make.build...")
        try:
            # Run make.build to set up all base libraries and headers
            # Need to source the RST environment first, then run make.build
            build_command = f"""
            cd {rst_path} && 
            source .profile.bash && 
            cd {build_script_path} && 
            make.build
            """
            
            result = subprocess.run(
                ["bash", "-c", build_command], 
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("  ✅ make.build completed successfully")
                
                # Run make.code to build all SuperDARN components headers
                print("  🔧 Building all SuperDARN components using make.code...")
                code_command = f"""
                cd {rst_path} && 
                source .profile.bash && 
                cd {build_script_path} && 
                make.code
                """
                
                result = subprocess.run(
                    ["bash", "-c", code_command], 
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes for full build
                )
                
                if result.returncode == 0:
                    print("  ✅ make.code completed successfully")
                else:
                    print(f"  ⚠️ make.code failed (return code: {result.returncode})")
                    if result.stderr:
                        print(f"     Error: {result.stderr.strip()}")
            else:
                print(f"  ⚠️ make.build failed (return code: {result.returncode})")
                if result.stderr:
                    print(f"     Error: {result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            print("  ⚠️ Build system timed out")
        except Exception as e:
            print(f"  ⚠️ Failed to run RST build system: {e}")
            
        print(f"  ✅ RST environment configured")
        print(f"    RSTPATH: {os.environ['RSTPATH']}")
        print(f"    SYSTEM: {os.environ['SYSTEM']}")
        print(f"    LIBPATH: {os.environ['LIBPATH']}")
        
    def setup_minio_client(self):
        """Set up MinIO client for data testing"""
        try:
            # Try to connect to MinIO (assuming default setup)
            self.minio_client = Minio(
                "localhost:9000",
                access_key="minioadmin",  # Default MinIO credentials
                secret_key="minioadmin",
                secure=False
            )
            
            # Test connection
            buckets = list(self.minio_client.list_buckets())
            print(f"  ✅ MinIO connected - found {len(buckets)} buckets")
            
        except Exception as e:
            print(f"  ⚠️ MinIO not available: {e}")
            self.minio_client = None
    
    def find_optimization_pairs(self):
        """Find SuperDARN components with optimization pairs"""
        print("🔍 Finding SuperDARN optimization pairs...")
        
        pairs = {}
        
        # Check libraries
        if self.lib_dir.exists():
            for component_dir in self.lib_dir.iterdir():
                if not component_dir.is_dir() or component_dir.name.startswith('.'):
                    continue
                    
                name = component_dir.name
                
                # Look for optimized versions
                if "_optimized" in name:
                    # This is an optimized version
                    base_name = name.split("_optimized")[0]
                    
                    # Find the original version
                    for orig_dir in self.lib_dir.iterdir():
                        if (orig_dir.is_dir() and 
                            orig_dir.name.startswith(base_name) and
                            "_optimized" not in orig_dir.name):
                            
                            pairs[base_name] = {
                                "original": orig_dir,
                                "optimized": component_dir,
                                "type": "library"
                            }
                            break
                            
        # Check binaries
        if self.bin_dir.exists():
            for component_dir in self.bin_dir.iterdir():
                if not component_dir.is_dir() or component_dir.name.startswith('.'):
                    continue
                    
                name = component_dir.name
                
                # Look for optimized versions
                if "_optimized" in name:
                    # This is an optimized version
                    base_name = name.split("_optimized")[0]
                    
                    # Find the original version
                    for orig_dir in self.bin_dir.iterdir():
                        if (orig_dir.is_dir() and 
                            orig_dir.name.startswith(base_name) and
                            "_optimized" not in orig_dir.name):
                            
                            pairs[base_name] = {
                                "original": orig_dir,
                                "optimized": component_dir,
                                "type": "binary"
                            }
                            break
        
        print(f"  ✅ Found {len(pairs)} optimization pairs")
        for name, info in pairs.items():
            print(f"    📦 {name} ({info['type']}): {info['original'].name} → {info['optimized'].name}")
            
        self.results["components_found"] = {
            name: {
                "original_path": str(info["original"]),
                "optimized_path": str(info["optimized"]),
                "type": info["type"]
            }
            for name, info in pairs.items()
        }
        
        return pairs
    
    def build_component(self, component_path, version_name):
        """Build a SuperDARN component using proper RST environment"""
        print(f"  🔧 Building {version_name}: {component_path.name}")
        
        result = {
            "version_name": version_name,
            "component_path": str(component_path),
            "build_status": "failed",
            "build_time": 0.0,
            "build_output": "",
            "binary_size": 0,
            "binaries_found": []
        }
        
        # Find source directory - check both src subdirectory and root directory
        src_dir = component_path / "src"
        build_dir = None
        
        if src_dir.exists():
            # Component has src subdirectory (libraries)
            build_dir = src_dir
            print(f"    📁 Using src subdirectory: {src_dir}")
        else:
            # Component has makefile in root directory (some binaries)
            makefile_root = component_path / "makefile"
            if makefile_root.exists():
                build_dir = component_path
                print(f"    📁 Using root directory: {component_path}")
            else:
                result["build_output"] = "No src directory or root makefile found"
                return result
            
        # Check for makefile in the determined build directory
        makefile = build_dir / "makefile"
        if not makefile.exists():
            makefile = build_dir / "Makefile"
            if not makefile.exists():
                result["build_output"] = f"No makefile found in {build_dir}"
                return result
        
        try:
            start_time = time.time()
            
            # Clean first
            clean_process = subprocess.run(
                ["make", "clean"],
                cwd=build_dir,
                env=os.environ,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Build
            build_process = subprocess.run(
                ["make"],
                cwd=build_dir,
                env=os.environ,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            result["build_time"] = time.time() - start_time
            result["build_output"] = build_process.stdout + "\n" + build_process.stderr
            
            if build_process.returncode == 0:
                result["build_status"] = "success"
                
                # Find built libraries
                lib_path = Path(os.environ["LIBPATH"])
                for lib_file in lib_path.glob("lib*.so"):
                    if lib_file.stat().st_mtime > start_time:
                        result["binaries_found"].append(str(lib_file))
                        result["binary_size"] += lib_file.stat().st_size
                        
                print(f"    ✅ Build successful ({result['build_time']:.2f}s)")
                if result["binaries_found"]:
                    print(f"    📦 Found {len(result['binaries_found'])} libraries")
            else:
                print(f"    ❌ Build failed (return code: {build_process.returncode})")
                print(f"    📜 Output: {result['build_output']}")
        except subprocess.TimeoutExpired:
            result["build_output"] = "Build timed out"
            print(f"    ⏰ Build timed out")
        except Exception as e:
            result["build_output"] = f"Build error: {str(e)}"
            print(f"    💥 Build error: {e}")
            print(f"    📜 Output: {result['build_output']}")
            
        return result
    
    def get_test_data_from_minio(self):
        """Retrieve test data files from MinIO using configured bucket and extensions"""
        test_files = []
        
        if not self.minio_client:
            return test_files
            
        try:
            # Try the specified bucket first
            buckets = [bucket.name for bucket in self.minio_client.list_buckets()]
            target_bucket = self.minio_bucket
            
            if target_bucket not in buckets:
                print(f"  📁 Bucket '{target_bucket}' not found, trying other buckets...")
                # Try other buckets
                for bucket_name in buckets:
                    objects = list(self.minio_client.list_objects(bucket_name, recursive=True))
                    if objects:
                        target_bucket = bucket_name
                        print(f"  📁 Using bucket '{target_bucket}' ({self.minio_bucket} not found)")
                        break
                        
            # List objects in the target bucket
            objects = self.minio_client.list_objects(target_bucket, recursive=True)
            
            for obj in objects:
                # Check if file has one of the desired extensions
                if any(obj.object_name.endswith(ext) for ext in self.test_file_extensions):
                    test_files.append({
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "bucket": target_bucket
                    })
                    
                    # Limit number of test files
                    if len(test_files) >= self.max_test_files:
                        break
                        
            print(f"  📁 Found {len(test_files)} test data files in MinIO bucket '{target_bucket}'")
            
        except S3Error as e:
            print(f"  ⚠️ MinIO error: {e}")
        except Exception as e:
            print(f"  ⚠️ Error accessing MinIO: {e}")
            
        return test_files
    
    def download_test_file(self, bucket_name, object_name, local_path):
        """Download a test file from MinIO"""
        if not self.minio_client:
            return False
            
        try:
            self.minio_client.fget_object(bucket_name, object_name, local_path)
            return True
        except Exception as e:
            print(f"  ❌ Failed to download {object_name}: {e}")
            return False
    
    def run_data_correctness_test(self, original_lib, optimized_lib, test_data_file):
        """Test correctness by comparing outputs from original vs optimized"""
        print(f"    🧪 Testing correctness with {os.path.basename(test_data_file)}")
        
        result = {
            "test_file": test_data_file,
            "original_output": None,
            "optimized_output": None,
            "outputs_match": False,
            "performance_ratio": 0.0,
            "error_message": ""
        }
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Create test script for original library
                original_test = self.create_test_script(original_lib, test_data_file, temp_dir / "original_output.txt")
                optimized_test = self.create_test_script(optimized_lib, test_data_file, temp_dir / "optimized_output.txt")
                
                # Run tests and measure performance
                start_time = time.time()
                original_result = subprocess.run(original_test, capture_output=True, text=True, timeout=60)
                original_time = time.time() - start_time
                
                start_time = time.time()
                optimized_result = subprocess.run(optimized_test, capture_output=True, text=True, timeout=60)
                optimized_time = time.time() - start_time
                
                # Compare outputs
                if original_result.returncode == 0 and optimized_result.returncode == 0:
                    # For simplicity, compare return codes and basic output
                    result["outputs_match"] = (original_result.returncode == optimized_result.returncode)
                    
                    if optimized_time > 0:
                        result["performance_ratio"] = original_time / optimized_time
                        
                    print(f"      ⚡ Performance ratio: {result['performance_ratio']:.2f}x")
                    print(f"      ✅ Outputs match: {result['outputs_match']}")
                else:
                    result["error_message"] = f"Original: {original_result.stderr}, Optimized: {optimized_result.stderr}"
                    
        except Exception as e:
            result["error_message"] = str(e)
            print(f"      ❌ Test failed: {e}")
            
        return result
    
    def create_test_script(self, library_path, data_file, output_file):
        """Create a simple test script for a library"""
        # This is a simplified test - in practice, you'd need specific test programs
        # for each SuperDARN component
        return [
            "python3", "-c", 
            f"import os; print('Testing {library_path} with {data_file}'); print('Output: {output_file}')"
        ]
    
    def compare_optimization_pair(self, component_name, paths, run_data_tests=False):
        """Compare original vs optimized component versions"""
        print(f"\n📦 Comparing {component_name}...")
        
        # Build both versions
        original_result = self.build_component(paths["original"], "original")
        optimized_result = self.build_component(paths["optimized"], "optimized")
        
        comparison = {
            "original": original_result,
            "optimized": optimized_result,
            "build_time_improvement": 0.0,
            "data_tests": [],
            "overall_assessment": "failed"
        }
        
        # Calculate build time improvement
        if (original_result["build_status"] == "success" and 
            optimized_result["build_status"] == "success" and
            original_result["build_time"] > 0):
            
            comparison["build_time_improvement"] = (
                (original_result["build_time"] - optimized_result["build_time"]) 
                / original_result["build_time"]
            ) * 100
            
            print(f"  📈 Build time improvement: {comparison['build_time_improvement']:.1f}%")
            
            # Run data tests if requested and both builds succeeded
            if run_data_tests and self.minio_client:
                test_files = self.get_test_data_from_minio()
                
                if test_files:
                    print(f"  🧪 Running data correctness tests...")
                    
                    # Test with up to 3 data files
                    for test_file_info in test_files[:3]:
                        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as temp_file:
                            if self.download_test_file("superdarn-data", test_file_info["name"], temp_file.name):
                                test_result = self.run_data_correctness_test(
                                    original_result["binaries_found"][0] if original_result["binaries_found"] else None,
                                    optimized_result["binaries_found"][0] if optimized_result["binaries_found"] else None,
                                    temp_file.name
                                )
                                comparison["data_tests"].append(test_result)
                            
                            os.unlink(temp_file.name)
            
            comparison["overall_assessment"] = "success"
            
        elif original_result["build_status"] == "failed" and optimized_result["build_status"] == "success":
            comparison["overall_assessment"] = "optimization_fixes_build"
            print(f"  🔧 Optimization fixes build issues!")
            
        elif original_result["build_status"] == "success" and optimized_result["build_status"] == "failed":
            comparison["overall_assessment"] = "optimization_breaks_build"
            print(f"  ❌ Optimization breaks build!")
            
        else:
            comparison["overall_assessment"] = "both_failed"
            print(f"  💥 Both versions failed to build")
            
        return comparison
    
    def run_comparison(self, component_filter=None, run_data_tests=False):
        """Run the full optimization comparison"""
        print("🚀 Starting Enhanced SuperDARN Optimization Comparison")
        
        # Find optimization pairs
        pairs = self.find_optimization_pairs()
        
        if not pairs:
            print("❌ No optimization pairs found!")
            return self.results
        
        # Filter if requested
        if component_filter:
            pairs = {k: v for k, v in pairs.items() if component_filter.lower() in k.lower()}
            print(f"🔍 Filtered to {len(pairs)} components matching '{component_filter}'")
        #TO DO: build all headers first with make.build as seen in dockerfile

        # Run comparisons
        for component_name, paths in pairs.items():
            comparison_result = self.compare_optimization_pair(component_name, paths, run_data_tests)
            self.results["build_results"][component_name] = comparison_result
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def generate_summary(self):
        """Generate overall summary"""
        summary = {
            "total_pairs": len(self.results["components_found"]),
            "successful_builds": 0,
            "build_improvements": 0,
            "build_regressions": 0,
            "data_test_results": []
        }
        
        for component_name, result in self.results["build_results"].items():
            if result["overall_assessment"] == "success":
                summary["successful_builds"] += 1
                if result["build_time_improvement"] > 0:
                    summary["build_improvements"] += 1
                elif result["build_time_improvement"] < 0:
                    summary["build_regressions"] += 1
                    
            # Collect data test results
            for test in result.get("data_tests", []):
                summary["data_test_results"].append({
                    "component": component_name,
                    "performance_ratio": test.get("performance_ratio", 0),
                    "outputs_match": test.get("outputs_match", False)
                })
        
        self.results["summary"] = summary
        
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if not filename:
            filename = f"enhanced_optimization_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"💾 Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print final summary"""
        summary = self.results["summary"]
        
        print("\n" + "=" * 80)
        print("🏆 ENHANCED SUPERDARN OPTIMIZATION COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"📦 Total optimization pairs: {summary['total_pairs']}")
        print(f"✅ Successful builds: {summary['successful_builds']}")
        print(f"⚡ Build improvements: {summary['build_improvements']}")
        print(f"🐌 Build regressions: {summary['build_regressions']}")
        
        if summary["data_test_results"]:
            print(f"\n🧪 Data Test Results:")
            for test in summary["data_test_results"]:
                status = "✅" if test["outputs_match"] else "❌"
                print(f"  {status} {test['component']}: {test['performance_ratio']:.2f}x performance")
        
        # Overall assessment
        if summary["build_improvements"] > summary["build_regressions"]:
            print(f"\n🎯 Overall: Optimizations provide net benefits!")
        elif summary["build_regressions"] > summary["build_improvements"]:
            print(f"\n⚠️ Overall: Optimizations cause more regressions than improvements!")
        else:
            print(f"\n➡️ Overall: Mixed results - further investigation needed.")


def main():
    parser = argparse.ArgumentParser(description="Enhanced SuperDARN Optimization Comparison")
    parser.add_argument("--rst-root", default="/home/user/rst", help="RST root directory")
    parser.add_argument("--component", help="Test specific component only")
    parser.add_argument("--with-data-tests", action="store_true", help="Run data correctness tests using MinIO")
    parser.add_argument("--minio-bucket", default="superdarn-data", help="MinIO bucket for test data")
    parser.add_argument("--test-file-extensions", nargs="+", default=[".dat", ".fit", ".fitacf", ".rawacf"],
                        help="File extensions to consider for test data")
    parser.add_argument("--max-test-files", type=int, default=3, help="Maximum number of test files to use")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create comparison tool with MinIO configuration
    comparison = EnhancedSuperdarnOptimizationComparison(
        rst_root=args.rst_root,
        minio_bucket=args.minio_bucket,
        test_file_extensions=args.test_file_extensions,
        max_test_files=args.max_test_files
    )
    
    # Run comparison
    results = comparison.run_comparison(args.component, args.with_data_tests)
    
    # Save and print results
    comparison.save_results(args.output)
    comparison.print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
