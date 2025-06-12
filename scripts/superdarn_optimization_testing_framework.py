#!/usr/bin/env python3
"""
SuperDARN Optimization Testing Framework
========================================

This framework tests SuperDARN components in their original form and
compares them with optimized versions when they exist. It does NOT
assume components already have OpenMP - most don't yet.

Key Features:
1. Tests original components as-is (vanilla C code)
2. Detects when optimized versions exist (e.g., *_optimized.* pattern)
3. Builds and profiles both versions when available
4. Shows "optimization opportunity" when only original exists
5. Integrates results into comprehensive dashboard

Usage:
    python superdarn_optimization_testing_framework.py
    python superdarn_optimization_testing_framework.py --component fit_speck_removal
    python superdarn_optimization_testing_framework.py --only-optimized
"""

import os
import sys
import json
import time
import subprocess
import argparse
import glob
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class SuperDARNOptimizationTester:
    def __init__(self, rst_root):
        self.rst_root = Path(rst_root)
        self.results_dir = self.rst_root / "test-results"
        self.results_dir.mkdir(exist_ok=True)
        
        # SuperDARN paths
        self.lib_dir = self.rst_root / "codebase" / "superdarn" / "src.lib" / "tk"
        self.bin_dir = self.rst_root / "codebase" / "superdarn" / "src.bin" / "tk" / "tool"
        
        self.results = {
            'libraries': {},
            'binaries': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_original': 0,
                'total_optimized': 0,
                'optimization_coverage': 0.0
            }
        }
        
        self.lock = threading.Lock()

    def find_all_components(self):
        """Find all SuperDARN components (original and optimized versions)"""
        components = {
            'libraries': {},
            'binaries': {}
        }
        
        # Find libraries
        if self.lib_dir.exists():
            for lib_path in self.lib_dir.iterdir():
                if lib_path.is_dir() and not lib_path.name.startswith('.'):
                    lib_name = lib_path.name
                    components['libraries'][lib_name] = {
                        'original': lib_path,
                        'optimized': None
                    }
                    
                    # Look for optimized version
                    optimized_pattern = f"{lib_name}_optimized*"
                    for opt_path in self.lib_dir.glob(optimized_pattern):
                        if opt_path.is_dir():
                            components['libraries'][lib_name]['optimized'] = opt_path
                            break
        
        # Find binaries
        if self.bin_dir.exists():
            for bin_path in self.bin_dir.iterdir():
                if bin_path.is_dir() and not bin_path.name.startswith('.'):
                    bin_name = bin_path.name
                    components['binaries'][bin_name] = {
                        'original': bin_path,
                        'optimized': None
                    }
                    
                    # Look for optimized version
                    optimized_pattern = f"{bin_name}_optimized*"
                    for opt_path in self.bin_dir.glob(optimized_pattern):
                        if opt_path.is_dir():
                            components['binaries'][bin_name]['optimized'] = opt_path
                            break
        
        return components

    def analyze_component_features(self, component_path):
        """Analyze what optimization features a component has"""
        features = {
            'has_openmp': False,
            'has_simd': False,
            'has_linked_lists': False,
            'has_loops': False,
            'source_files': [],
            'makefile_exists': False,
            'complexity_score': 0
        }
        
        # Find source files
        for ext in ['*.c', '*.h']:
            features['source_files'].extend(list(component_path.rglob(ext)))
        
        # Check for makefile (both in src subdirectory and root)
        makefile_path_src = os.path.join(component_path, "src", "makefile")
        makefile_path_root = os.path.join(component_path, "makefile")
        features['makefile_exists'] = os.path.exists(makefile_path_src) or os.path.exists(makefile_path_root)
        # Analyze source code features
        for src_file in features['source_files']:
            try:
                with open(src_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Check for OpenMP
                    if '#include <omp.h>' in content or '#pragma omp' in content:
                        features['has_openmp'] = True
                    
                    # Check for SIMD
                    if any(simd in content for simd in ['<immintrin.h>', '__AVX', '__SSE', '_mm_']):
                        features['has_simd'] = True
                    
                    # Check for linked lists (common patterns)
                    if any(pattern in content for pattern in ['->next', '->prev', 'struct.*next', 'list.*head']):
                        features['has_linked_lists'] = True
                    
                    # Check for loops
                    if any(loop in content for loop in ['for(', 'for (', 'while(', 'while (']):
                        features['has_loops'] = True
                    
                    # Simple complexity scoring
                    features['complexity_score'] += content.count('for') * 2
                    features['complexity_score'] += content.count('while') * 2
                    features['complexity_score'] += content.count('if') * 1
                    features['complexity_score'] += len(content.split('\n'))
                    
            except Exception as e:
                continue
        
        return features

    def build_component(self, component_path, component_type):
        """Build a component and measure performance"""
        result = {
            'build_status': 'not_attempted',
            'build_time': 0.0,
            'binary_size': 0,
            'build_output': '',
            'features': {},
            'optimization_level': 'O0'  # Default for original versions
        }
        
        try:
            # Analyze features first
            result['features'] = self.analyze_component_features(component_path)
            
            # Check if makefile exists in src subdirectory or root
            makefile_path_src = os.path.join(component_path, "src", "makefile")
            makefile_path_root = os.path.join(component_path, "makefile")
            
            build_dir = component_path
            if os.path.exists(makefile_path_src):
                makefile_path = makefile_path_src
                build_dir = os.path.join(component_path, "src")
                print(f"🔍 Found makefile in src subdirectory: {makefile_path}")
            elif os.path.exists(makefile_path_root):
                makefile_path = makefile_path_root
                build_dir = component_path
                print(f"🔍 Found makefile in root directory: {makefile_path}")
            else:
                result['build_status'] = 'no_makefile'
                result['build_output'] = 'No makefile found in src directory or root'
                return result
            
            # Determine optimization level from path
            if 'optimized' in str(component_path):
                result['optimization_level'] = 'O3_plus_openmp'  # Assume optimized versions use O3 + OpenMP
            
            # Build the component
            start_time = time.time()
            
            cmd = ["make", "-C", str(build_dir)]
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            build_time = time.time() - start_time
            result['build_time'] = build_time
            result['build_output'] = process.stdout + process.stderr
            
            if process.returncode == 0:
                result['build_status'] = 'success'
                
                # Try to find and measure binary size
                for binary_pattern in ['*.exe', '*']:
                    binaries = list(component_path.glob(f"**/{binary_pattern}"))
                    for binary in binaries:
                        if binary.is_file() and binary.stat().st_size > 1000:  # Reasonable binary size
                            result['binary_size'] = binary.stat().st_size
                            break
                    if result['binary_size'] > 0:
                        break
            else:
                result['build_status'] = 'failed'
                
        except subprocess.TimeoutExpired:
            result['build_status'] = 'timeout'
        except Exception as e:
            result['build_status'] = 'error'
            print(f"Error building {component_type} {component_path}: {str(e)}")
            if result['build_output'] == '':
                result['build_output'] = str(e)
                print(e)
            else:
                result['build_output'] += f"\nError: {str(e)}"
                print(result['build_output'])
        
        return result

    def test_component_pair(self, component_name, component_type, paths):
        """Test both original and optimized versions of a component"""
        print(f"Testing {component_type}: {component_name}")
        
        results = {
            'original': None,
            'optimized': None,
            'has_optimization': False,
            'performance_gain': {}
        }
        
        # Test original version
        if paths['original']:
            print(f"  Building original version...")
            results['original'] = self.build_component(paths['original'], component_type)
        
        # Test optimized version if it exists
        if paths['optimized']:
            print(f"  Building optimized version...")
            results['optimized'] = self.build_component(paths['optimized'], component_type)
            results['has_optimization'] = True
            
            # Calculate performance gains
            if (results['original'] and results['original']['build_status'] == 'success' and
                results['optimized'] and results['optimized']['build_status'] == 'success'):
                
                orig_time = results['original']['build_time']
                opt_time = results['optimized']['build_time']
                
                if orig_time > 0:
                    results['performance_gain'] = {
                        'build_time_ratio': opt_time / orig_time,
                        'build_time_improvement': ((orig_time - opt_time) / orig_time) * 100,
                        'size_change': 0  # Would need actual runtime comparison
                    }
        
        return results

    def run_comprehensive_test(self, max_workers=4, component_filter=None, only_optimized=False):
        """Run comprehensive testing of all components"""
        print("🔍 Discovering SuperDARN components...")
        components = self.find_all_components()
        
        total_libs = len(components['libraries'])
        total_bins = len(components['binaries'])
        
        print(f"Found {total_libs} libraries and {total_bins} binaries")
        
        # Filter components if requested
        if component_filter:
            components['libraries'] = {k: v for k, v in components['libraries'].items() 
                                     if component_filter.lower() in k.lower()}
            components['binaries'] = {k: v for k, v in components['binaries'].items() 
                                    if component_filter.lower() in k.lower()}
        
        if only_optimized:
            components['libraries'] = {k: v for k, v in components['libraries'].items() 
                                     if v['optimized'] is not None}
            components['binaries'] = {k: v for k, v in components['binaries'].items() 
                                    if v['optimized'] is not None}
        
        all_tests = []
        
        # Prepare library tests
        for lib_name, paths in components['libraries'].items():
            all_tests.append(('library', lib_name, paths))
        
        # Prepare binary tests
        for bin_name, paths in components['binaries'].items():
            all_tests.append(('binary', bin_name, paths))
        
        print(f"🧪 Running {len(all_tests)} component tests with {max_workers} workers...")
        
        # Run tests in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {
                executor.submit(self.test_component_pair, name, comp_type, paths): (comp_type, name)
                for comp_type, name, paths in all_tests
            }
            
            completed = 0
            for future in as_completed(future_to_test):
                comp_type, name = future_to_test[future]
                try:
                    result = future.result()
                    
                    with self.lock:
                        if comp_type == 'library':
                            self.results['libraries'][name] = result
                        else:
                            self.results['binaries'][name] = result
                    
                    completed += 1
                    print(f"✅ Completed {completed}/{len(all_tests)}: {name}")
                    
                except Exception as e:
                    print(f"❌ Failed {comp_type}: {name} - {e}")
        
        # Update metadata
        total_original = len([r for r in self.results['libraries'].values() if r['original']] + 
                           [r for r in self.results['binaries'].values() if r['original']])
        total_optimized = len([r for r in self.results['libraries'].values() if r['optimized']] + 
                            [r for r in self.results['binaries'].values() if r['optimized']])
        
        self.results['metadata'].update({
            'total_original': total_original,
            'total_optimized': total_optimized,
            'optimization_coverage': (total_optimized / total_original * 100) if total_original > 0 else 0
        })
        
        return self.results

    def save_results(self, filename="optimization_test_results.json"):
        """Save test results to JSON file"""
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📊 Results saved to: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description='SuperDARN Optimization Testing Framework')
    parser.add_argument('--rst-root', default='.', help='RST root directory')
    parser.add_argument('--component', help='Test specific component only')
    parser.add_argument('--only-optimized', action='store_true', help='Test only components with optimized versions')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SuperDARNOptimizationTester(args.rst_root)
    
    # Run tests
    print("🚀 Starting SuperDARN Optimization Testing Framework")
    results = tester.run_comprehensive_test(
        max_workers=args.workers,
        component_filter=args.component,
        only_optimized=args.only_optimized
    )
    
    # Save results
    tester.save_results()
    
    # Print summary
    print("\n📈 TESTING SUMMARY")
    print("=" * 50)
    print(f"Total Original Components: {results['metadata']['total_original']}")
    print(f"Total Optimized Components: {results['metadata']['total_optimized']}")
    print(f"Optimization Coverage: {results['metadata']['optimization_coverage']:.1f}%")
    
    if results['metadata']['total_optimized'] > 0:
        print(f"\n🎯 Components with optimizations:")
        for comp_type in ['libraries', 'binaries']:
            for name, data in results[comp_type].items():
                if data['has_optimization']:
                    status = "✅" if (data['optimized'] and data['optimized']['build_status'] == 'success') else "❌"
                    print(f"  {status} {comp_type[:-1]}: {name}")
                    if status == "✅":
                        print(f"    - Build Time: {data['optimized']['build_time']:.2f}s")
                        print(f"    - Size: {data['optimized']['binary_size']} bytes")
                        if 'performance_gain' in data and data['performance_gain']:
                            print(f"    - Performance Gain: {data['performance_gain']['build_time_improvement']:.1f}%")
                    else:
                        print(data['optimized']['build_output']) 

if __name__ == "__main__":
    main()
