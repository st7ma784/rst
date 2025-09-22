#!/usr/bin/env python3
"""
Simple SuperDARN FITACF test data generator (no numpy dependency)
Generates realistic test data for CUDA vs CPU pipeline comparison
"""

import struct
import math
import random
import os
from datetime import datetime

def generate_simple_fitacf_data():
    """Generate simple FITACF test data in binary format"""
    
    # SuperDARN parameters
    nrang = 75  # Number of range gates
    mplgs = 17  # Number of lags
    beams = 16  # Number of beams
    
    # Create test data directory
    os.makedirs('test_data', exist_ok=True)
    
    # Generate one test file
    filename = 'test_data/20250115.1200.00.sas.fitacf'
    
    print(f"Generating test FITACF data: {filename}")
    
    with open(filename, 'wb') as f:
        # Write a simple header
        f.write(b'FITACF_TEST_DATA')
        f.write(struct.pack('<I', beams))  # Number of beams
        f.write(struct.pack('<I', nrang))  # Number of ranges
        f.write(struct.pack('<I', mplgs))  # Number of lags
        
        # Generate data for each beam
        for beam in range(beams):
            print(f"  Generating beam {beam + 1}/{beams}")
            
            # Beam header
            f.write(struct.pack('<I', beam))  # Beam number
            f.write(struct.pack('<I', 20250115))  # Date
            f.write(struct.pack('<I', 120000))   # Time
            
            # Generate ACF data for each range gate
            for rng in range(nrang):
                # Power (decreasing with range)
                power = 10000.0 * math.exp(-rng / 20.0) * (1 + 0.1 * random.gauss(0, 1))
                power = max(power, 100.0)  # Minimum noise level
                f.write(struct.pack('<f', power))
                
                # Generate ACF real and imaginary parts
                for lag in range(mplgs):
                    # Realistic ACF with exponential decay and phase rotation
                    lag_time = lag * 0.0024  # 2.4 ms
                    velocity = 200 * math.sin(2 * math.pi * rng / 20)  # Spatial pattern
                    width = 50 + abs(velocity) * 0.3  # Spectral width
                    
                    # ACF calculation
                    decay = math.exp(-(width * lag_time) ** 2)
                    phase = 2 * math.pi * velocity * lag_time / 300.0
                    amplitude = power * decay
                    
                    # Add noise
                    noise_level = 100.0
                    real_part = amplitude * math.cos(phase) + random.gauss(0, noise_level * 0.1)
                    imag_part = amplitude * math.sin(phase) + random.gauss(0, noise_level * 0.1)
                    
                    if lag == 0:
                        imag_part = 0.0  # Lag 0 is purely real
                    
                    f.write(struct.pack('<f', real_part))
                    f.write(struct.pack('<f', imag_part))
    
    print(f"Test data generated: {filename}")
    print(f"File size: {os.path.getsize(filename)} bytes")
    
    return filename

def create_test_summary():
    """Create a summary of the test data"""
    summary_file = 'test_data/test_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("SuperDARN FITACF Test Data Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Purpose: Testing CUDA vs CPU FITACF processing\n\n")
        f.write("File Format:\n")
        f.write("- Binary format with simple header\n")
        f.write("- 16 beams × 75 ranges × 17 lags\n")
        f.write("- Realistic ACF data with noise\n")
        f.write("- Exponential range dependence\n")
        f.write("- Spatial velocity patterns\n\n")
        f.write("Data Structure:\n")
        f.write("- Header: 'FITACF_TEST_DATA' + parameters\n")
        f.write("- For each beam: beam_num, date, time\n")
        f.write("- For each range: power + 17 complex ACF values\n")
        f.write("- All values in IEEE 754 32-bit float format\n")
    
    print(f"Summary written: {summary_file}")

if __name__ == '__main__':
    print("Simple SuperDARN FITACF Test Data Generator")
    print("=" * 45)
    
    # Generate test data
    test_file = generate_simple_fitacf_data()
    create_test_summary()
    
    print("\nTest data generation complete!")
    print("Ready for CPU vs CUDA pipeline testing.")