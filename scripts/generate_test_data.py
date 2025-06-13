#!/usr/bin/env python3
"""
SuperDARN RST Performance Test Data Generator
============================================

This script generates synthetic radar data files for performance testing
of the SuperDARN RST optimization framework.

Usage:
    python generate_test_data.py [options]

Features:
- Generates realistic RAWACF format files
- Configurable file sizes and complexity
- Multiple dataset configurations
- Metadata generation for tracking
"""

import os
import sys
import json
import random
import struct
import argparse
from datetime import datetime, timedelta
from pathlib import Path

class RAWACFGenerator:
    """Generator for synthetic RAWACF files that mimic real radar data structure"""
    
    def __init__(self):
        self.header_size = 512
        self.record_size = 1024
        self.beam_count = 16
        self.range_gate_count = 75
        self.frequency_count = 8
        
    def generate_header(self, radar_id=1, scan_id=1):
        """Generate RAWACF file header"""
        header = bytearray(self.header_size)
        
        # Magic number for RAWACF format
        struct.pack_into('<I', header, 0, 0x41574152)  # 'RAWA'
        
        # Version and basic info
        struct.pack_into('<H', header, 4, 1)           # Version
        struct.pack_into('<H', header, 6, radar_id)    # Radar ID
        struct.pack_into('<I', header, 8, scan_id)     # Scan ID
        
        # Timestamp
        now = datetime.utcnow()
        struct.pack_into('<Q', header, 12, int(now.timestamp()))
        
        # Configuration
        struct.pack_into('<H', header, 20, self.beam_count)
        struct.pack_into('<H', header, 22, self.range_gate_count)
        struct.pack_into('<H', header, 24, self.frequency_count)
        
        return bytes(header)
    
    def generate_record(self, beam_num, complexity_factor=1.0):
        """Generate a single data record"""
        record = bytearray(self.record_size)
        
        # Record header
        struct.pack_into('<H', record, 0, beam_num)
        struct.pack_into('<H', record, 2, self.range_gate_count)
        
        # Generate synthetic radar data
        offset = 16
        for gate in range(self.range_gate_count):
            for freq in range(self.frequency_count):
                # Generate complex IQ data with some realistic characteristics
                if random.random() < 0.3 * complexity_factor:  # Signal present
                    i_val = random.randint(-32768, 32767)
                    q_val = random.randint(-32768, 32767)
                else:  # Noise
                    i_val = random.randint(-1000, 1000)
                    q_val = random.randint(-1000, 1000)
                
                struct.pack_into('<hh', record, offset, i_val, q_val)
                offset += 4
                
                if offset >= self.record_size - 4:
                    break
            if offset >= self.record_size - 4:
                break
        
        return bytes(record)
    
    def generate_file(self, output_path, target_size_mb, complexity_factor=1.0):
        """Generate a complete RAWACF file"""
        target_bytes = target_size_mb * 1024 * 1024
        
        with open(output_path, 'wb') as f:
            # Write header
            header = self.generate_header()
            f.write(header)
            bytes_written = len(header)
            
            # Write records until we reach target size
            beam_num = 0
            while bytes_written < target_bytes:
                record = self.generate_record(beam_num % self.beam_count, complexity_factor)
                f.write(record)
                bytes_written += len(record)
                beam_num += 1
                
                # Add some variation in record sizes
                if random.random() < 0.1:
                    padding = random.randint(0, 100)
                    f.write(b'\x00' * padding)
                    bytes_written += padding
        
        return output_path

def create_dataset_metadata(dataset_dir, config):
    """Create metadata file for a dataset"""
    files = list(Path(dataset_dir).glob('*.rawacf'))
    
    metadata = {
        "dataset_name": config['name'],
        "description": config['description'],
        "file_count": len(files),
        "target_size_mb": config['target_size_mb'],
        "complexity_factor": config.get('complexity_factor', 1.0),
        "created_at": datetime.utcnow().isoformat(),
        "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
        "files": [
            {
                "name": f.name,
                "size_mb": f.stat().st_size / (1024 * 1024),
                "path": str(f.relative_to(dataset_dir))
            }
            for f in files
        ],
        "performance_expectations": {
            "estimated_processing_time_seconds": len(files) * config['target_size_mb'] * 0.1,
            "expected_memory_usage_mb": config['target_size_mb'] * 2,
            "complexity_level": config.get('complexity_level', 'medium')
        }
    }
    
    metadata_path = Path(dataset_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path

def generate_test_datasets(output_dir, datasets_config=None):
    """Generate all test datasets"""
    
    if datasets_config is None:
        datasets_config = {
            'small': {
                'name': 'small',
                'description': 'Quick validation tests - small files for rapid testing',
                'file_count': 3,
                'target_size_mb': 2,
                'complexity_factor': 0.8,
                'complexity_level': 'low'
            },
            'medium': {
                'name': 'medium',
                'description': 'Standard performance tests - medium complexity files',
                'file_count': 5,
                'target_size_mb': 8,
                'complexity_factor': 1.0,
                'complexity_level': 'medium'
            },
            'large': {
                'name': 'large',
                'description': 'Comprehensive tests - large files with high complexity',
                'file_count': 8,
                'target_size_mb': 25,
                'complexity_factor': 1.3,
                'complexity_level': 'high'
            },
            'benchmark': {
                'name': 'benchmark',
                'description': 'Intensive benchmark tests - very large complex files',
                'file_count': 12,
                'target_size_mb': 50,
                'complexity_factor': 1.5,
                'complexity_level': 'extreme'
            }
        }
    
    generator = RAWACFGenerator()
    output_path = Path(output_dir)
    
    print(f"Generating test datasets in {output_path}")
    
    for dataset_name, config in datasets_config.items():
        print(f"\nGenerating {dataset_name} dataset...")
        
        # Create dataset directory
        dataset_dir = output_path / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate files
        for i in range(config['file_count']):
            filename = f"radar_{i:03d}.rawacf"
            filepath = dataset_dir / filename
            
            print(f"  Creating {filename} ({config['target_size_mb']}MB)...")
            
            generator.generate_file(
                str(filepath),
                config['target_size_mb'],
                config['complexity_factor']
            )
        
        # Create metadata
        metadata_path = create_dataset_metadata(dataset_dir, config)
        print(f"  Created metadata: {metadata_path}")
        
        # Print dataset summary
        total_size = sum(f.stat().st_size for f in dataset_dir.glob('*.rawacf'))
        print(f"  Dataset complete: {config['file_count']} files, {total_size / (1024*1024):.1f}MB total")

def create_readme(output_dir):
    """Create README file for test data"""
    readme_content = """# SuperDARN RST Performance Test Data
    
This directory contains synthetic radar data files for performance testing
of the SuperDARN RST optimization framework.

## Dataset Structure

```
test-data/
├── small/           # Quick validation (2MB files, 3 files)
├── medium/          # Standard tests (8MB files, 5 files)  
├── large/           # Comprehensive tests (25MB files, 8 files)
├── benchmark/       # Intensive benchmarks (50MB files, 12 files)
└── README.md        # This file
```

## Dataset Descriptions

### Small Dataset
- **Purpose**: Quick validation and CI testing
- **File Count**: 3 files
- **File Size**: ~2MB each
- **Complexity**: Low complexity radar patterns
- **Expected Runtime**: < 30 seconds per build type

### Medium Dataset  
- **Purpose**: Standard performance testing
- **File Count**: 5 files
- **File Size**: ~8MB each
- **Complexity**: Realistic radar complexity
- **Expected Runtime**: 1-3 minutes per build type

### Large Dataset
- **Purpose**: Comprehensive performance analysis
- **File Count**: 8 files
- **File Size**: ~25MB each
- **Complexity**: High complexity radar patterns
- **Expected Runtime**: 5-10 minutes per build type

### Benchmark Dataset
- **Purpose**: Intensive performance benchmarking
- **File Count**: 12 files
- **File Size**: ~50MB each
- **Complexity**: Maximum complexity patterns
- **Expected Runtime**: 15-30 minutes per build type

## File Format

Files are in RAWACF format with synthetic radar data that includes:
- Realistic IQ data patterns
- Multiple beam configurations
- Variable range gate data
- Frequency diversity patterns

## Usage in Performance Tests

These datasets are automatically used by the GitHub Actions performance testing
workflow. Each dataset includes a `metadata.json` file with:
- File descriptions and sizes
- Performance expectations
- Complexity metrics
- Creation timestamps

## Regenerating Test Data

To regenerate this test data:

```bash
python scripts/generate_test_data.py --output test-data/
```

## Data Characteristics

The synthetic data mimics real SuperDARN radar characteristics:
- Multi-beam scanning patterns
- Range-Doppler data structures  
- Frequency diversity
- Realistic noise and signal patterns
- Variable data complexity

Generated: {timestamp}
"""
    
    readme_path = Path(output_dir) / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content.format(timestamp=datetime.utcnow().isoformat()))
    
    return readme_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic test data for SuperDARN RST performance testing'
    )
    parser.add_argument(
        '--output', '-o',
        default='test-data',
        help='Output directory for test datasets (default: test-data)'
    )
    parser.add_argument(
        '--dataset', '-d',
        choices=['small', 'medium', 'large', 'benchmark', 'all'],
        default='all',
        help='Generate specific dataset (default: all)'
    )
    parser.add_argument(
        '--custom-config',
        help='Path to custom dataset configuration JSON file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without creating files'
    )
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    datasets_config = None
    if args.custom_config:
        with open(args.custom_config, 'r') as f:
            datasets_config = json.load(f)
    
    # Filter datasets if specific one requested
    if args.dataset != 'all' and datasets_config is None:
        # Load default config and filter
        default_config = {
            'small': {
                'name': 'small',
                'description': 'Quick validation tests',
                'file_count': 3,
                'target_size_mb': 2,
                'complexity_factor': 0.8,
                'complexity_level': 'low'
            },
            'medium': {
                'name': 'medium', 
                'description': 'Standard performance tests',
                'file_count': 5,
                'target_size_mb': 8,
                'complexity_factor': 1.0,
                'complexity_level': 'medium'
            },
            'large': {
                'name': 'large',
                'description': 'Comprehensive tests',
                'file_count': 8,
                'target_size_mb': 25,
                'complexity_factor': 1.3,
                'complexity_level': 'high'
            },
            'benchmark': {
                'name': 'benchmark',
                'description': 'Intensive benchmark tests',
                'file_count': 12,
                'target_size_mb': 50,
                'complexity_factor': 1.5,
                'complexity_level': 'extreme'
            }
        }
        datasets_config = {args.dataset: default_config[args.dataset]}
    
    if args.dry_run:
        print("DRY RUN - Would generate:")
        if datasets_config:
            for name, config in datasets_config.items():
                print(f"  {name}: {config['file_count']} files × {config['target_size_mb']}MB")
        else:
            print("  All default datasets")
        return
    
    # Generate datasets
    generate_test_datasets(args.output, datasets_config)
    
    # Create README
    readme_path = create_readme(args.output)
    print(f"\nCreated README: {readme_path}")
    
    print(f"\nTest data generation complete!")
    print(f"Output directory: {Path(args.output).absolute()}")

if __name__ == '__main__':
    main()
