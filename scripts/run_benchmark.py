#!/usr/bin/env python3
"""
SuperDARN Data Benchmark Script

Runs extracted convmap data through Python processing pipeline and measures timing.
Compares Python (NumPy) processing performance and validates outputs.

Usage:
    python3 run_benchmark.py [--sample N] [--output DIR]
"""

import sys
import os
import time
import json
import hashlib
import argparse
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime

# Add pythonv2 to path
sys.path.insert(0, '/home/user/rst/pythonv2')

# Suppress CuPy warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
from superdarn_gpu.io.dmap import DmapReader, read_dmap


@dataclass
class BenchmarkResult:
    """Results from processing a single file"""
    filename: str
    num_records: int
    processing_time_sec: float
    read_time_sec: float
    compute_time_sec: float
    memory_mb: float
    output_hash: str
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark run"""
    total_files: int
    successful_files: int
    failed_files: int
    total_records: int
    total_read_time: float
    total_compute_time: float
    total_processing_time: float
    avg_time_per_file: float
    avg_time_per_record: float
    throughput_files_per_sec: float
    throughput_records_per_sec: float
    timestamp: str


def compute_convmap_statistics(records: List[Any]) -> Dict[str, Any]:
    """
    Compute basic statistics from convmap records.
    This simulates what a real processing pipeline would do.
    """
    stats = {
        'num_records': len(records),
        'total_potential_drop': 0.0,
        'avg_chi_sqr': 0.0,
        'avg_rms_err': 0.0,
        'vectors_processed': 0,
        'lat_range': [90.0, 0.0],
        'mlt_range': [24.0, 0.0],
    }
    
    potential_drops = []
    chi_sqrs = []
    rms_errs = []
    
    for rec in records:
        # Extract potential drop
        pot_drop = rec.get('pot.drop', 0.0)
        if pot_drop:
            potential_drops.append(pot_drop)
            stats['total_potential_drop'] += pot_drop
            
        # Extract chi-square
        chi_sqr = rec.get('chi.sqr', 0.0)
        if chi_sqr:
            chi_sqrs.append(chi_sqr)
            
        # Extract RMS error
        rms_err = rec.get('rms.err', 0.0)
        if rms_err:
            rms_errs.append(rms_err)
            
        # Process velocity vectors if present
        if 'vector.mlat' in rec.arrays:
            mlats = rec.arrays['vector.mlat']
            stats['vectors_processed'] += len(mlats)
            if len(mlats) > 0:
                stats['lat_range'][0] = min(stats['lat_range'][0], np.min(mlats))
                stats['lat_range'][1] = max(stats['lat_range'][1], np.max(mlats))
                
        if 'vector.mlon' in rec.arrays:
            mlons = rec.arrays['vector.mlon']
            # Convert to MLT
            mlts = (mlons / 15.0) % 24.0
            if len(mlts) > 0:
                stats['mlt_range'][0] = min(stats['mlt_range'][0], np.min(mlts))
                stats['mlt_range'][1] = max(stats['mlt_range'][1], np.max(mlts))
                
    if chi_sqrs:
        stats['avg_chi_sqr'] = np.mean(chi_sqrs)
    if rms_errs:
        stats['avg_rms_err'] = np.mean(rms_errs)
        
    return stats


def hash_stats(stats: Dict[str, Any]) -> str:
    """Create a hash of the statistics for comparison"""
    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    # Serialize stats deterministically
    converted = convert(stats)
    serialized = json.dumps(converted, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()[:16]


def process_single_file(filepath: Path) -> BenchmarkResult:
    """Process a single convmap file and measure timing"""
    filename = filepath.name
    
    try:
        # Measure read time
        read_start = time.perf_counter()
        with DmapReader(str(filepath)) as reader:
            records = reader.read_all()
        read_time = time.perf_counter() - read_start
        
        # Measure compute time
        compute_start = time.perf_counter()
        stats = compute_convmap_statistics(records)
        compute_time = time.perf_counter() - compute_start
        
        total_time = read_time + compute_time
        
        # Create output hash
        output_hash = hash_stats(stats)
        
        return BenchmarkResult(
            filename=filename,
            num_records=len(records),
            processing_time_sec=total_time,
            read_time_sec=read_time,
            compute_time_sec=compute_time,
            memory_mb=0.0,  # Would need psutil for accurate measurement
            output_hash=output_hash,
            success=True
        )
        
    except Exception as e:
        return BenchmarkResult(
            filename=filename,
            num_records=0,
            processing_time_sec=0.0,
            read_time_sec=0.0,
            compute_time_sec=0.0,
            memory_mb=0.0,
            output_hash="",
            success=False,
            error=str(e)
        )


def run_benchmark(
    data_dir: Path,
    sample_size: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> BenchmarkSummary:
    """
    Run benchmark on convmap files.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing extracted convmap files
    sample_size : int, optional
        Number of files to process (None = all)
    output_dir : Path, optional
        Directory to save results
        
    Returns
    -------
    BenchmarkSummary
        Summary statistics
    """
    # Find all convmap files
    files = sorted(data_dir.glob('*.cnvmap'))
    total_available = len(files)
    
    if sample_size:
        files = files[:sample_size]
        
    print(f"\n{'='*60}")
    print(f"SuperDARN Python Benchmark")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Files available: {total_available}")
    print(f"Files to process: {len(files)}")
    print(f"{'='*60}\n")
    
    results: List[BenchmarkResult] = []
    
    start_time = time.perf_counter()
    
    for i, filepath in enumerate(files):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(files) - i - 1) / rate if rate > 0 else 0
            print(f"Processing file {i+1}/{len(files)} ({rate:.1f} files/s, ETA: {eta:.0f}s)...")
            
        result = process_single_file(filepath)
        results.append(result)
        
    total_time = time.perf_counter() - start_time
    
    # Calculate summary statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    total_records = sum(r.num_records for r in successful)
    total_read = sum(r.read_time_sec for r in successful)
    total_compute = sum(r.compute_time_sec for r in successful)
    total_proc = sum(r.processing_time_sec for r in successful)
    
    summary = BenchmarkSummary(
        total_files=len(files),
        successful_files=len(successful),
        failed_files=len(failed),
        total_records=total_records,
        total_read_time=total_read,
        total_compute_time=total_compute,
        total_processing_time=total_proc,
        avg_time_per_file=total_proc / len(successful) if successful else 0,
        avg_time_per_record=total_proc / total_records if total_records > 0 else 0,
        throughput_files_per_sec=len(successful) / total_time if total_time > 0 else 0,
        throughput_records_per_sec=total_records / total_time if total_time > 0 else 0,
        timestamp=datetime.now().isoformat()
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Files processed:      {summary.total_files}")
    print(f"  - Successful:       {summary.successful_files}")
    print(f"  - Failed:           {summary.failed_files}")
    print(f"Total records:        {summary.total_records}")
    print(f"\nTiming:")
    print(f"  Total wall time:    {total_time:.2f}s")
    print(f"  Total read time:    {summary.total_read_time:.2f}s")
    print(f"  Total compute time: {summary.total_compute_time:.2f}s")
    print(f"  Avg per file:       {summary.avg_time_per_file*1000:.2f}ms")
    print(f"  Avg per record:     {summary.avg_time_per_record*1000:.3f}ms")
    print(f"\nThroughput:")
    print(f"  Files/sec:          {summary.throughput_files_per_sec:.1f}")
    print(f"  Records/sec:        {summary.throughput_records_per_sec:.1f}")
    print(f"{'='*60}\n")
    
    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_file = output_dir / 'benchmark_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"Summary saved to: {summary_file}")
        
        # Save detailed results
        results_file = output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Detailed results saved to: {results_file}")
        
        # Save failures
        if failed:
            failures_file = output_dir / 'benchmark_failures.txt'
            with open(failures_file, 'w') as f:
                for r in failed:
                    f.write(f"{r.filename}: {r.error}\n")
            print(f"Failures saved to: {failures_file}")
            
    return summary


def main():
    parser = argparse.ArgumentParser(description='SuperDARN Data Benchmark')
    parser.add_argument('--data-dir', type=Path, 
                        default=Path('/home/user/rst/extracted_data'),
                        help='Directory containing convmap files')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of files to sample (default: all)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
        
    summary = run_benchmark(
        data_dir=args.data_dir,
        sample_size=args.sample,
        output_dir=args.output
    )
    
    return summary


if __name__ == '__main__':
    main()
