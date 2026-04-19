#!/usr/bin/env python3
"""
Simple SuperDARN test data generator without external dependencies

This script creates basic test data files for testing the CUDA implementation
using only Python standard library modules.

Author: CUDA Conversion Project
Date: 2025
"""

import struct
import math
import random
import os
from datetime import datetime, timedelta

def generate_acf_data(num_ranges=75, num_lags=17):
    """Generate realistic ACF data using pure Python"""
    
    acf_real = []
    acf_imag = []
    power = []
    velocity = []
    width = []
    
    for r in range(num_ranges):
        # Range-dependent parameters
        range_km = 180 + r * 45  # Starting at 180 km, 45 km gates
        
        # Power decreases with range
        base_power = 10000 * math.exp(-range_km / 1000.0) * (1 + 0.1 * random.gauss(0, 1))
        range_power = max(base_power, 1000)
        power.append(range_power)
        
        # Realistic velocity patterns
        range_velocity = 200 * math.sin(2 * math.pi * r / 20) + 50 * random.gauss(0, 1)
        velocity.append(range_velocity)
        
        # Spectral width
        range_width = 50 + abs(range_velocity) * 0.3 + 20 * random.gauss(0, 1)
        range_width = max(range_width, 20)
        width.append(range_width)
        
        # Generate ACF for this range
        range_acf_real = []
        range_acf_imag = []
        
        for lag in range(num_lags):
            lag_time = lag * 2.4e-3  # 2.4 ms lag separation
            
            # Theoretical ACF model
            decay = math.exp(-(range_width * lag_time) ** 2)
            phase = 2 * math.pi * range_velocity * lag_time / 300.0
            
            amplitude = range_power * decay
            
            # Add noise
            noise_real = random.gauss(0, 100)
            noise_imag = random.gauss(0, 100)
            
            real_val = amplitude * math.cos(phase) + noise_real
            imag_val = amplitude * math.sin(phase) + noise_imag
            
            # Lag 0 is purely real
            if lag == 0:
                imag_val = 0
            
            range_acf_real.append(real_val)
            range_acf_imag.append(imag_val)
        
        acf_real.append(range_acf_real)
        acf_imag.append(range_acf_imag)
    
    return {
        'acf_real': acf_real,
        'acf_imag': acf_imag,
        'power': power,
        'velocity': velocity,
        'width': width
    }

def write_binary_test_file(data, filename):
    """Write test data in a simple binary format"""
    
    num_ranges = len(data['power'])
    num_lags = len(data['acf_real'][0])
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(b'TSTD')  # Test data magic number
        f.write(struct.pack('<II', num_ranges, num_lags))
        
        # Write power array
        for p in data['power']:
            f.write(struct.pack('<f', p))
        
        # Write velocity array
        for v in data['velocity']:
            f.write(struct.pack('<f', v))
        
        # Write width array
        for w in data['width']:
            f.write(struct.pack('<f', w))
        
        # Write ACF data (interleaved real/imaginary)
        for r in range(num_ranges):
            for lag in range(num_lags):
                f.write(struct.pack('<f', data['acf_real'][r][lag]))
                f.write(struct.pack('<f', data['acf_imag'][r][lag]))

def write_text_test_file(data, filename):
    """Write test data in text format for easy inspection"""
    
    num_ranges = len(data['power'])
    num_lags = len(data['acf_real'][0])
    
    with open(filename, 'w') as f:
        f.write("# SuperDARN Test Data File\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Ranges: {num_ranges}, Lags: {num_lags}\n")
        f.write("#\n")
        f.write("# Format: Range Power Velocity Width ACF_Real[0..16] ACF_Imag[0..16]\n")
        f.write("#\n")
        
        for r in range(num_ranges):
            f.write(f"{r:3d} {data['power'][r]:8.1f} {data['velocity'][r]:7.1f} {data['width'][r]:6.1f}")
            
            # Write ACF real parts
            for lag in range(num_lags):
                f.write(f" {data['acf_real'][r][lag]:8.2f}")
            
            # Write ACF imaginary parts
            for lag in range(num_lags):
                f.write(f" {data['acf_imag'][r][lag]:8.2f}")
            
            f.write("\n")

def write_c_test_file(data, filename):
    """Write test data as C arrays for direct inclusion in test programs"""
    
    num_ranges = len(data['power'])
    num_lags = len(data['acf_real'][0])
    
    with open(filename, 'w') as f:
        f.write("/*\n")
        f.write(" * SuperDARN Test Data Arrays\n")
        f.write(f" * Generated: {datetime.now().isoformat()}\n")
        f.write(f" * Ranges: {num_ranges}, Lags: {num_lags}\n")
        f.write(" */\n\n")
        
        f.write(f"#define TEST_NUM_RANGES {num_ranges}\n")
        f.write(f"#define TEST_NUM_LAGS {num_lags}\n\n")
        
        # Power array
        f.write("static float test_power[TEST_NUM_RANGES] = {\n    ")
        for i, p in enumerate(data['power']):
            f.write(f"{p:8.1f}f")
            if i < len(data['power']) - 1:
                f.write(", ")
                if (i + 1) % 8 == 0:
                    f.write("\n    ")
        f.write("\n};\n\n")
        
        # Velocity array
        f.write("static float test_velocity[TEST_NUM_RANGES] = {\n    ")
        for i, v in enumerate(data['velocity']):
            f.write(f"{v:7.1f}f")
            if i < len(data['velocity']) - 1:
                f.write(", ")
                if (i + 1) % 8 == 0:
                    f.write("\n    ")
        f.write("\n};\n\n")
        
        # Width array
        f.write("static float test_width[TEST_NUM_RANGES] = {\n    ")
        for i, w in enumerate(data['width']):
            f.write(f"{w:6.1f}f")
            if i < len(data['width']) - 1:
                f.write(", ")
                if (i + 1) % 8 == 0:
                    f.write("\n    ")
        f.write("\n};\n\n")
        
        # ACF arrays
        f.write("static float test_acf_real[TEST_NUM_RANGES][TEST_NUM_LAGS] = {\n")
        for r in range(num_ranges):
            f.write("    {")
            for lag in range(num_lags):
                f.write(f"{data['acf_real'][r][lag]:8.2f}f")
                if lag < num_lags - 1:
                    f.write(", ")
            f.write("}")
            if r < num_ranges - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        f.write("static float test_acf_imag[TEST_NUM_RANGES][TEST_NUM_LAGS] = {\n")
        for r in range(num_ranges):
            f.write("    {")
            for lag in range(num_lags):
                f.write(f"{data['acf_imag'][r][lag]:8.2f}f")
                if lag < num_lags - 1:
                    f.write(", ")
            f.write("}")
            if r < num_ranges - 1:
                f.write(",")
            f.write("\n")
        f.write("};\n")

def main():
    print("SuperDARN Simple Test Data Generator")
    print("=" * 40)
    
    # Create output directory
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate different sized test datasets
    test_configs = [
        {'ranges': 25, 'lags': 17, 'name': 'small'},
        {'ranges': 75, 'lags': 17, 'name': 'medium'},
        {'ranges': 150, 'lags': 17, 'name': 'large'}
    ]
    
    for config in test_configs:
        print(f"Generating {config['name']} dataset ({config['ranges']} ranges, {config['lags']} lags)...")
        
        # Generate data
        data = generate_acf_data(config['ranges'], config['lags'])
        
        # Write in different formats
        base_name = f"test_{config['name']}"
        
        write_binary_test_file(data, os.path.join(output_dir, f"{base_name}.bin"))
        write_text_test_file(data, os.path.join(output_dir, f"{base_name}.txt"))
        write_c_test_file(data, os.path.join(output_dir, f"{base_name}.h"))
        
        print(f"  Created: {base_name}.bin, {base_name}.txt, {base_name}.h")
    
    # Write test documentation
    doc_file = os.path.join(output_dir, "README.md")
    with open(doc_file, 'w') as f:
        f.write("# SuperDARN Test Data\n\n")
        f.write("This directory contains synthetic SuperDARN test data for validating the CUDA implementation.\n\n")
        f.write("## File Formats\n\n")
        f.write("- `.bin` files: Binary format for efficient loading in C/CUDA programs\n")
        f.write("- `.txt` files: Human-readable text format for inspection\n")
        f.write("- `.h` files: C header files with static arrays for direct inclusion\n\n")
        f.write("## Data Sets\n\n")
        f.write("- `small`: 25 ranges, 17 lags - for quick testing\n")
        f.write("- `medium`: 75 ranges, 17 lags - typical SuperDARN size\n")
        f.write("- `large`: 150 ranges, 17 lags - stress testing\n\n")
        f.write("## Usage\n\n")
        f.write("Use these files to test CPU vs CUDA implementations and validate numerical accuracy.\n")
        f.write("The data includes realistic physics-based ACF patterns with appropriate noise levels.\n")
    
    print(f"\nTest data generation complete!")
    print(f"Files saved to: {output_dir}/")
    print(f"Documentation: {output_dir}/README.md")

if __name__ == '__main__':
    main()