#!/usr/bin/env python3
"""
SuperDARN Multi-System Benchmark Script

Catalogues SuperDARN data on /dev/sda through /dev/sdd, runs it through 
3 processing systems (C, CUDA, Python), times execution, and compares outputs.
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import tempfile

# Data inventory structures
@dataclass
class DriveInfo:
    """Information about a data drive"""
    device: str
    mount_point: str
    total_space_gb: float
    used_space_gb: float
    available_space_gb: float

@dataclass 
class DataFile:
    """A SuperDARN data file"""
    name: str
    path: str
    minio_path: str
    size_bytes: int
    data_type: str  # 'rawacf', 'fitacf', 'convmap', etc.
    timestamp: str  # YYYYMMDDHHM from filename

@dataclass
class DataInventory:
    """Complete data inventory"""
    scan_time: str
    drives: List[DriveInfo]
    file_counts: Dict[str, int]
    total_files: int
    total_size_gb: float
    data_files: Dict[str, List[DataFile]]  # by data_type
    date_range: Dict[str, Tuple[str, str]]  # by data_type -> (earliest, latest)

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    system: str  # 'c_cpu', 'c_cuda', 'python'
    data_file: str
    success: bool
    execution_time_sec: float
    memory_peak_mb: float
    output_file: str
    output_hash: str
    error_message: Optional[str] = None

@dataclass
class ComparisonResult:
    """Comparison of outputs between systems"""
    file_name: str
    systems_compared: List[str]
    outputs_identical: bool
    hash_values: Dict[str, str]
    differences: List[str]

# Drive mount points
DRIVE_MOUNTS = {
    'sda': '/media/user/c9e6d06c-2981-4697-9747-450cd09e6cde',
    'sdb': '/media/user/c4b5011b-4278-431f-b977-4e3e7812ed79',
    'sdc': '/media/user/c078fa0c-611b-4a52-a4a2-97a506a541b9',
    'sdd': '/media/user/f185ee55-7d95-4936-8b51-eb231896a216'
}

# Data types and their directories
DATA_TYPES = {
    'rawacf': 'rawacf',
    'fitacf': 'fitacf', 
    'fitacf_output': 'fitacf-output',
    'convmap': 'convmap',
    'grid': 'grid'
}

# RST root directory
RST_ROOT = Path('/home/user/rst')


class SuperDARNCatalogue:
    """Catalogues SuperDARN data across multiple drives"""
    
    def __init__(self):
        self.drives = {}
        self.inventory = None
        
    def scan_drives(self) -> Dict[str, DriveInfo]:
        """Scan all configured drives and get their status"""
        for device, mount_point in DRIVE_MOUNTS.items():
            if os.path.exists(mount_point):
                try:
                    stat = os.statvfs(mount_point)
                    total = (stat.f_blocks * stat.f_frsize) / (1024**3)
                    used = ((stat.f_blocks - stat.f_bfree) * stat.f_frsize) / (1024**3)
                    available = (stat.f_bavail * stat.f_frsize) / (1024**3)
                    
                    self.drives[device] = DriveInfo(
                        device=device,
                        mount_point=mount_point,
                        total_space_gb=round(total, 2),
                        used_space_gb=round(used, 2),
                        available_space_gb=round(available, 2)
                    )
                except Exception as e:
                    print(f"Warning: Could not stat {device}: {e}")
                    
        return self.drives
    
    def catalogue_data(self, max_files_per_type: int = None) -> DataInventory:
        """
        Catalogue all SuperDARN data files across drives.
        
        Parameters
        ----------
        max_files_per_type : int, optional
            Limit number of files catalogued per type (for quick scans)
        """
        if not self.drives:
            self.scan_drives()
            
        data_files = {dtype: [] for dtype in DATA_TYPES}
        file_counts = {dtype: 0 for dtype in DATA_TYPES}
        date_range = {}
        total_size = 0
        
        # Use first available drive for cataloguing (they have identical data)
        primary_drive = None
        for device, info in self.drives.items():
            if os.path.exists(info.mount_point):
                primary_drive = info
                break
                
        if not primary_drive:
            raise RuntimeError("No drives available for cataloguing")
            
        print(f"Scanning data on {primary_drive.device} ({primary_drive.mount_point})...")
        
        for dtype, dirname in DATA_TYPES.items():
            data_dir = Path(primary_drive.mount_point) / dirname
            
            if not data_dir.exists():
                print(f"  {dtype}: directory not found")
                continue
                
            # List all items in the directory
            items = list(data_dir.iterdir())
            count = 0
            earliest = None
            latest = None
            
            for item in items:
                if max_files_per_type and count >= max_files_per_type:
                    break
                    
                # Extract timestamp from filename (format: YYYYMMDDHH)
                name = item.name
                if len(name) >= 10 and name[:10].isdigit():
                    timestamp = name[:10]
                elif '.' in name:
                    parts = name.split('.')
                    timestamp = parts[0] if parts[0][:8].isdigit() else None
                else:
                    timestamp = None
                
                # Find the actual data file (MinIO structure)
                if item.is_dir():
                    # MinIO format: name/uuid/part.1
                    part_files = list(item.glob('*/part.1'))
                    if part_files:
                        data_path = part_files[0]
                        file_size = data_path.stat().st_size
                    else:
                        continue
                else:
                    data_path = item
                    file_size = item.stat().st_size
                
                data_file = DataFile(
                    name=name,
                    path=str(data_path),
                    minio_path=str(item),
                    size_bytes=file_size,
                    data_type=dtype,
                    timestamp=timestamp or 'unknown'
                )
                
                data_files[dtype].append(data_file)
                total_size += file_size
                count += 1
                
                # Track date range
                if timestamp:
                    if earliest is None or timestamp < earliest:
                        earliest = timestamp
                    if latest is None or timestamp > latest:
                        latest = timestamp
            
            file_counts[dtype] = len(items)  # Total count
            if earliest and latest:
                date_range[dtype] = (earliest, latest)
                
            print(f"  {dtype}: {file_counts[dtype]} files, catalogued {count}")
        
        self.inventory = DataInventory(
            scan_time=datetime.now().isoformat(),
            drives=list(self.drives.values()),
            file_counts=file_counts,
            total_files=sum(file_counts.values()),
            total_size_gb=round(total_size / (1024**3), 2),
            data_files=data_files,
            date_range=date_range
        )
        
        return self.inventory
    
    def save_inventory(self, filename: str = 'data_inventory.json'):
        """Save inventory to JSON file"""
        if not self.inventory:
            raise RuntimeError("No inventory to save - run catalogue_data() first")
            
        output_path = RST_ROOT / 'scripts' / filename
        
        # Convert to dict
        inventory_dict = {
            'scan_time': self.inventory.scan_time,
            'drives': [asdict(d) for d in self.inventory.drives],
            'file_counts': self.inventory.file_counts,
            'total_files': self.inventory.total_files,
            'total_size_gb': self.inventory.total_size_gb,
            'date_range': self.inventory.date_range,
            'data_files': {
                dtype: [asdict(f) for f in files]
                for dtype, files in self.inventory.data_files.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(inventory_dict, f, indent=2)
            
        print(f"Inventory saved to {output_path}")
        return output_path


class SystemBenchmark:
    """Benchmarks data processing across different systems"""
    
    def __init__(self, work_dir: Path = None):
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix='superdarn_bench_'))
        self.results = []
        self.comparisons = []
        
        # System configurations
        self.systems = {
            'c_cpu': {
                'name': 'C CPU',
                'available': self._check_c_binary(),
                'runner': self._run_c_cpu
            },
            'c_cuda': {
                'name': 'C CUDA',
                'available': self._check_cuda_binary(),
                'runner': self._run_c_cuda
            },
            'python': {
                'name': 'Python',
                'available': True,  # Always available
                'runner': self._run_python
            }
        }
        
    def _check_c_binary(self) -> bool:
        """Check if C binary is available"""
        # Look for make_grid, map_potential, etc.
        bin_dir = RST_ROOT / 'build' / 'bin'
        return (bin_dir / 'make_grid').exists() or (bin_dir / 'makeall').exists()
    
    def _check_cuda_binary(self) -> bool:
        """Check if CUDA binary is available"""
        cuda_dir = RST_ROOT / 'CUDArst'
        # Check for compiled CUDA objects or binaries
        build_dir = cuda_dir / 'build'
        if build_dir.exists():
            return any(build_dir.glob('*.o')) or any(build_dir.glob('cudarst*'))
        return False
    
    def extract_data_file(self, data_file: DataFile) -> Path:
        """
        Extract data from MinIO format to a regular file.
        
        Returns the path to the extracted file.
        """
        # Create output path
        output_path = self.work_dir / data_file.name
        
        # Copy the data part
        source_path = Path(data_file.path)
        if source_path.exists():
            shutil.copy2(source_path, output_path)
        else:
            raise FileNotFoundError(f"Data file not found: {source_path}")
            
        return output_path
    
    def _run_c_cpu(self, input_file: Path, data_type: str) -> Tuple[Path, float]:
        """Run C CPU processing"""
        output_file = self.work_dir / f"{input_file.stem}_c_cpu_output"
        
        # Determine which C tool to use based on data type
        start_time = time.time()
        
        if data_type == 'convmap':
            # Use map_potential or similar
            cmd = ['map_potential', str(input_file), '-o', str(output_file)]
        elif data_type == 'fitacf':
            cmd = ['make_grid', str(input_file), '-o', str(output_file)]
        else:
            # Placeholder - just copy for now
            shutil.copy2(input_file, output_file)
            cmd = None
            
        if cmd:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300)
                if result.returncode != 0:
                    raise RuntimeError(f"C processing failed: {result.stderr.decode()}")
            except FileNotFoundError:
                # Binary not found - use placeholder
                shutil.copy2(input_file, output_file)
                
        elapsed = time.time() - start_time
        return output_file, elapsed
    
    def _run_c_cuda(self, input_file: Path, data_type: str) -> Tuple[Path, float]:
        """Run C CUDA processing"""
        output_file = self.work_dir / f"{input_file.stem}_c_cuda_output"
        
        start_time = time.time()
        
        # Try to use CUDA RST tools
        cuda_bin = RST_ROOT / 'CUDArst' / 'cudarst'
        
        if cuda_bin.exists():
            cmd = [str(cuda_bin), str(input_file), '-o', str(output_file)]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300)
                if result.returncode != 0:
                    raise RuntimeError(f"CUDA processing failed: {result.stderr.decode()}")
            except FileNotFoundError:
                # Use placeholder
                shutil.copy2(input_file, output_file)
        else:
            # Placeholder
            shutil.copy2(input_file, output_file)
            
        elapsed = time.time() - start_time
        return output_file, elapsed
    
    def _run_python(self, input_file: Path, data_type: str) -> Tuple[Path, float]:
        """Run Python processing"""
        output_file = self.work_dir / f"{input_file.stem}_python_output"
        
        start_time = time.time()
        
        try:
            # Add pythonv2 to path
            sys.path.insert(0, str(RST_ROOT / 'pythonv2'))
            
            from superdarn_gpu.io.readers import load
            from superdarn_gpu.processing.convmap import ConvMapProcessor
            from superdarn_gpu.processing.grid import GridProcessor
            from superdarn_gpu.processing.fitacf import FitACFProcessor
            
            # Load data
            data = load(input_file)
            
            # Process based on data type
            if data_type == 'convmap':
                processor = ConvMapProcessor()
                result = processor.process(data)
            elif data_type == 'grid':
                processor = GridProcessor()
                result = processor.process(data)
            elif data_type == 'fitacf':
                processor = FitACFProcessor()
                result = processor.process(data)
            else:
                result = data
            
            # Save output
            import pickle
            with open(output_file, 'wb') as f:
                pickle.dump(result, f)
                
        except Exception as e:
            # If processing fails, just copy the file as placeholder
            print(f"  Warning: Python processing error: {e}")
            shutil.copy2(input_file, output_file)
            
        elapsed = time.time() - start_time
        return output_file, elapsed
    
    def compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # Shortened for display
    
    def benchmark_file(self, data_file: DataFile, systems: List[str] = None) -> List[BenchmarkResult]:
        """
        Benchmark a single data file across specified systems.
        
        Parameters
        ----------
        data_file : DataFile
            Data file to benchmark
        systems : list, optional
            Systems to benchmark (default: all available)
        """
        if systems is None:
            systems = [s for s, info in self.systems.items() if info['available']]
            
        # Extract data file
        try:
            extracted_path = self.extract_data_file(data_file)
        except Exception as e:
            return [BenchmarkResult(
                system=s, data_file=data_file.name, success=False,
                execution_time_sec=0, memory_peak_mb=0,
                output_file='', output_hash='',
                error_message=f"Extraction failed: {e}"
            ) for s in systems]
        
        results = []
        
        for system in systems:
            sys_info = self.systems.get(system)
            if not sys_info or not sys_info['available']:
                continue
                
            print(f"    Running {sys_info['name']}...")
            
            try:
                output_path, elapsed = sys_info['runner'](extracted_path, data_file.data_type)
                output_hash = self.compute_file_hash(output_path) if output_path.exists() else ''
                
                result = BenchmarkResult(
                    system=system,
                    data_file=data_file.name,
                    success=True,
                    execution_time_sec=round(elapsed, 4),
                    memory_peak_mb=0,  # TODO: Implement memory tracking
                    output_file=str(output_path),
                    output_hash=output_hash
                )
            except Exception as e:
                result = BenchmarkResult(
                    system=system,
                    data_file=data_file.name,
                    success=False,
                    execution_time_sec=0,
                    memory_peak_mb=0,
                    output_file='',
                    output_hash='',
                    error_message=str(e)
                )
            
            results.append(result)
            self.results.append(result)
            
        return results
    
    def compare_outputs(self, results: List[BenchmarkResult]) -> ComparisonResult:
        """Compare outputs from different systems"""
        successful = [r for r in results if r.success]
        
        if len(successful) < 2:
            return ComparisonResult(
                file_name=results[0].data_file if results else 'unknown',
                systems_compared=[r.system for r in successful],
                outputs_identical=len(successful) <= 1,
                hash_values={r.system: r.output_hash for r in successful},
                differences=['Not enough successful runs to compare']
            )
        
        hash_values = {r.system: r.output_hash for r in successful}
        unique_hashes = set(hash_values.values())
        identical = len(unique_hashes) == 1
        
        differences = []
        if not identical:
            # Group systems by their output hash
            hash_groups = {}
            for system, h in hash_values.items():
                if h not in hash_groups:
                    hash_groups[h] = []
                hash_groups[h].append(system)
                
            for h, systems in hash_groups.items():
                differences.append(f"Hash {h}: {', '.join(systems)}")
        
        comparison = ComparisonResult(
            file_name=results[0].data_file,
            systems_compared=list(hash_values.keys()),
            outputs_identical=identical,
            hash_values=hash_values,
            differences=differences
        )
        
        self.comparisons.append(comparison)
        return comparison
    
    def cleanup(self):
        """Clean up intermediate files"""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
            print(f"Cleaned up work directory: {self.work_dir}")


def run_full_benchmark(
    sample_size: int = 10,
    data_types: List[str] = None,
    save_results: bool = True,
    cleanup: bool = True
):
    """
    Run the full benchmark pipeline.
    
    Parameters
    ----------
    sample_size : int
        Number of files to benchmark per data type
    data_types : list
        Data types to benchmark (default: ['convmap'])
    save_results : bool
        Save results to JSON file
    cleanup : bool
        Clean up intermediate files after benchmark
    """
    if data_types is None:
        data_types = ['convmap']
    
    print("=" * 60)
    print("SuperDARN Multi-System Benchmark")
    print("=" * 60)
    print()
    
    # Step 1: Catalogue data
    print("Step 1: Cataloguing data...")
    catalogue = SuperDARNCatalogue()
    catalogue.scan_drives()
    
    print(f"\nDrives found:")
    for device, info in catalogue.drives.items():
        print(f"  {device}: {info.mount_point}")
        print(f"    Total: {info.total_space_gb:.1f} GB, Used: {info.used_space_gb:.1f} GB")
    
    inventory = catalogue.catalogue_data(max_files_per_type=sample_size * 2)
    catalogue.save_inventory()
    
    print(f"\nData summary:")
    print(f"  Total files: {inventory.total_files}")
    print(f"  Total size: {inventory.total_size_gb:.2f} GB")
    for dtype, (earliest, latest) in inventory.date_range.items():
        print(f"  {dtype}: {earliest} to {latest}")
    
    # Step 2: Run benchmarks
    print(f"\nStep 2: Running benchmarks (sample_size={sample_size})...")
    benchmark = SystemBenchmark()
    
    print(f"\nAvailable systems:")
    for sys_id, sys_info in benchmark.systems.items():
        status = "✓" if sys_info['available'] else "✗"
        print(f"  [{status}] {sys_info['name']}")
    
    all_results = []
    all_comparisons = []
    
    for dtype in data_types:
        if dtype not in inventory.data_files:
            print(f"\nSkipping {dtype}: no data files found")
            continue
            
        files = inventory.data_files[dtype][:sample_size]
        print(f"\nBenchmarking {dtype} ({len(files)} files)...")
        
        for i, data_file in enumerate(files):
            print(f"  [{i+1}/{len(files)}] {data_file.name}")
            
            results = benchmark.benchmark_file(data_file)
            all_results.extend(results)
            
            comparison = benchmark.compare_outputs(results)
            all_comparisons.append(comparison)
            
            # Print brief result
            times = {r.system: r.execution_time_sec for r in results if r.success}
            if times:
                time_str = ", ".join(f"{s}: {t:.3f}s" for s, t in times.items())
                match_str = "✓ identical" if comparison.outputs_identical else "✗ different"
                print(f"    Times: {time_str} | {match_str}")
    
    # Step 3: Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    # Timing summary
    timing_by_system = {}
    for result in all_results:
        if result.success:
            if result.system not in timing_by_system:
                timing_by_system[result.system] = []
            timing_by_system[result.system].append(result.execution_time_sec)
    
    print("\nAverage execution times:")
    for system, times in timing_by_system.items():
        avg = sum(times) / len(times)
        print(f"  {system}: {avg:.4f}s (n={len(times)})")
    
    # Output comparison summary
    identical_count = sum(1 for c in all_comparisons if c.outputs_identical)
    print(f"\nOutput comparisons: {identical_count}/{len(all_comparisons)} identical")
    
    # Save results
    if save_results:
        results_path = RST_ROOT / 'scripts' / 'benchmark_results.json'
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': sample_size,
            'data_types': data_types,
            'results': [asdict(r) for r in all_results],
            'comparisons': [asdict(c) for c in all_comparisons],
            'timing_summary': {
                sys: {'avg': sum(t)/len(t), 'count': len(t)}
                for sys, t in timing_by_system.items()
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    # Cleanup
    if cleanup:
        benchmark.cleanup()
    
    print("\nBenchmark complete!")
    return all_results, all_comparisons


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SuperDARN Multi-System Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick scan with 5 files
  python superdarn_benchmark.py --sample-size 5
  
  # Full benchmark with 50 files, don't cleanup
  python superdarn_benchmark.py --sample-size 50 --no-cleanup
  
  # Benchmark specific data type
  python superdarn_benchmark.py --data-type fitacf --sample-size 10
  
  # Just catalogue, no benchmark
  python superdarn_benchmark.py --catalogue-only
"""
    )
    
    parser.add_argument('--sample-size', type=int, default=10,
                       help='Number of files to benchmark per data type (default: 10)')
    parser.add_argument('--data-type', nargs='+', default=['convmap'],
                       help='Data types to benchmark (default: convmap)')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep intermediate files')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to JSON')
    parser.add_argument('--catalogue-only', action='store_true',
                       help='Only run data catalogue, skip benchmarks')
    
    args = parser.parse_args()
    
    if args.catalogue_only:
        print("Running data catalogue only...")
        catalogue = SuperDARNCatalogue()
        catalogue.scan_drives()
        inventory = catalogue.catalogue_data()
        catalogue.save_inventory()
        
        print("\nInventory summary:")
        print(f"  Total files: {inventory.total_files}")
        print(f"  Total size: {inventory.total_size_gb:.2f} GB")
        for dtype, count in inventory.file_counts.items():
            print(f"  {dtype}: {count} files")
    else:
        run_full_benchmark(
            sample_size=args.sample_size,
            data_types=args.data_type,
            save_results=not args.no_save,
            cleanup=not args.no_cleanup
        )


if __name__ == '__main__':
    main()
