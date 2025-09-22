#!/usr/bin/env python3
"""
Generate synthetic SuperDARN FITACF test data for CUDA implementation testing

This script creates realistic FITACF test files that follow the SuperDARN
data format specifications for testing the CUDA-accelerated processing pipeline.

Author: CUDA Conversion Project
Date: 2025
"""

import numpy as np
import struct
import gzip
import os
from datetime import datetime, timedelta
import argparse

class FITACFDataGenerator:
    """Generate synthetic FITACF data files"""
    
    def __init__(self):
        # SuperDARN radar parameters
        self.radars = {
            'sas': {'name': 'Saskatoon', 'id': 5, 'lat': 52.16, 'lon': -106.53},
            'pgr': {'name': 'Prince George', 'id': 6, 'lat': 53.98, 'lon': -122.59},
            'rkn': {'name': 'Rankin Inlet', 'id': 64, 'lat': 62.82, 'lon': -93.11},
            'cly': {'name': 'Clyde River', 'id': 66, 'lat': 70.49, 'lon': -68.52},
            'inv': {'name': 'Inuvik', 'id': 65, 'lat': 68.41, 'lon': -133.77}
        }
        
        # Standard SuperDARN parameters
        self.max_range = 75
        self.max_lags = 17
        self.frequencies = [8000, 10000, 12000, 14000, 16000, 18000]  # kHz
        self.beam_count = 16
        self.pulse_sequences = [
            [0, 14, 22, 24, 27, 31, 42, 43],  # Standard 8-pulse
            [0, 15, 16, 23, 27, 29, 32, 47],  # Alternative sequence
        ]
        
    def generate_acf_data(self, num_ranges, num_lags, noise_level=1000):
        """Generate realistic ACF data with physics-based characteristics"""
        
        # Initialize arrays
        acf_real = np.zeros((num_ranges, num_lags))
        acf_imag = np.zeros((num_ranges, num_lags))
        power = np.zeros(num_ranges)
        velocity = np.zeros(num_ranges)
        width = np.zeros(num_ranges)
        
        for r in range(num_ranges):
            # Range-dependent parameters
            range_km = 180 + r * 45  # Starting at 180 km, 45 km gates
            
            # Power decreases with range (1/R^2 law with atmospheric effects)
            base_power = 10000 * np.exp(-range_km / 1000.0) * (1 + 0.1 * np.random.randn())
            power[r] = max(base_power, noise_level)
            
            # Realistic velocity patterns (ionospheric flow)
            # Create spatial coherence patterns
            velocity[r] = 200 * np.sin(2 * np.pi * r / 20) + 50 * np.random.randn()
            
            # Spectral width (turbulence increases with velocity)
            width[r] = 50 + abs(velocity[r]) * 0.3 + 20 * np.random.randn()
            width[r] = max(width[r], 20)  # Minimum width
            
            # Generate ACF based on theoretical model
            for lag in range(num_lags):
                lag_time = lag * 2.4e-3  # 2.4 ms lag separation
                
                # Theoretical ACF: P * exp(-width^2 * t^2) * exp(i * velocity * t)
                decay = np.exp(-(width[r] * lag_time) ** 2)
                phase = 2 * np.pi * velocity[r] * lag_time / 300.0  # Convert to phase
                
                amplitude = power[r] * decay
                
                # Add realistic noise
                noise_real = np.random.normal(0, noise_level * 0.1)
                noise_imag = np.random.normal(0, noise_level * 0.1)
                
                acf_real[r, lag] = amplitude * np.cos(phase) + noise_real
                acf_imag[r, lag] = amplitude * np.sin(phase) + noise_imag
                
                # Lag 0 is purely real
                if lag == 0:
                    acf_imag[r, lag] = 0
        
        return {
            'acf_real': acf_real,
            'acf_imag': acf_imag, 
            'power': power,
            'velocity': velocity,
            'width': width
        }
    
    def generate_fitacf_record(self, radar_code, timestamp, beam_num, scan_flag=1):
        """Generate a single FITACF record in DMAP format"""
        
        radar_info = self.radars[radar_code]
        num_ranges = self.max_range
        num_lags = self.max_lags
        
        # Generate ACF data
        acf_data = self.generate_acf_data(num_ranges, num_lags)
        
        # Create FITACF record structure
        record = {
            # Time information
            'time.yr': timestamp.year,
            'time.mo': timestamp.month,
            'time.dy': timestamp.day,
            'time.hr': timestamp.hour,
            'time.mt': timestamp.minute,
            'time.sc': timestamp.second,
            'time.us': timestamp.microsecond,
            
            # Radar information
            'radar.revision.major': 5,
            'radar.revision.minor': 0,
            'cp': 153,  # Control program ID
            'stid': radar_info['id'],  # Station ID
            'tfreq': np.random.choice(self.frequencies),  # Transmit frequency
            'scan': scan_flag,
            'rxrise': 100,  # Receiver rise time
            'bmnum': beam_num,
            'nrang': num_ranges,
            'frang': 180,  # First range gate (km)
            'rsep': 45,    # Range separation (km)
            'nave': 20,    # Number of averages
            'lagfr': 1200, # Lag to first range (μs)
            'smsep': 300,  # Sample separation (μs)
            'noise.search': 1000,  # Noise level
            'noise.mean': 1200,
            'mpinc': 2400,  # Multi-pulse increment (μs)
            'mppul': 8,     # Number of pulses
            'mplgs': num_lags,  # Number of lags
            'mplgexs': 0,   # Number of excluded lags
            'pulse': np.array(self.pulse_sequences[0], dtype=np.int16),
            'lag': np.array([[i, i+1] for i in range(num_lags)], dtype=np.int16),
            
            # Data arrays
            'pwr0': acf_data['power'].astype(np.float32),
            'slist': np.arange(num_ranges, dtype=np.int16),  # Range gate list
            'nump': np.ones(num_ranges, dtype=np.int16) * num_lags,  # Points per range
            
            # ACF data (real and imaginary)
            'acfd': np.stack([acf_data['acf_real'], acf_data['acf_imag']], axis=2).flatten().astype(np.float32),
            
            # Fitted parameters (what FITACF algorithm produces)
            'v': acf_data['velocity'].astype(np.float32),
            'w_l': acf_data['width'].astype(np.float32),
            'p_l': acf_data['power'].astype(np.float32),
            
            # Error estimates
            'v_e': np.abs(acf_data['velocity'] * 0.1).astype(np.float32),
            'w_l_e': np.abs(acf_data['width'] * 0.1).astype(np.float32),
            'p_l_e': np.abs(acf_data['power'] * 0.1).astype(np.float32),
            
            # Quality flags
            'gflg': np.ones(num_ranges, dtype=np.int8),  # Ground scatter flag
            'qflg': np.ones(num_ranges, dtype=np.int8),  # Quality flag
        }
        
        return record
    
    def write_dmap_file(self, records, filename):
        """Write records to a DMAP format file"""
        
        print(f"Writing {len(records)} records to {filename}")
        
        # For this demonstration, we'll create a simplified binary format
        # In a real implementation, this would follow the exact DMAP specification
        
        with open(filename, 'wb') as f:
            # Write header
            f.write(b'DMAP')  # Magic number
            f.write(struct.pack('<I', len(records)))  # Number of records
            
            for record in records:
                # Write record header
                f.write(struct.pack('<I', len(record)))  # Number of fields
                
                for key, value in record.items():
                    # Write field name
                    key_bytes = key.encode('utf-8')
                    f.write(struct.pack('<H', len(key_bytes)))
                    f.write(key_bytes)
                    
                    # Write data type and data
                    if isinstance(value, int):
                        f.write(struct.pack('<c', b'i'))  # Integer type
                        f.write(struct.pack('<i', value))
                    elif isinstance(value, float):
                        f.write(struct.pack('<c', b'f'))  # Float type
                        f.write(struct.pack('<f', value))
                    elif isinstance(value, np.ndarray):
                        if value.dtype == np.int16:
                            f.write(struct.pack('<c', b's'))  # Short array
                        elif value.dtype == np.int8:
                            f.write(struct.pack('<c', b'c'))  # Char array
                        else:
                            f.write(struct.pack('<c', b'F'))  # Float array
                        
                        f.write(struct.pack('<I', len(value)))  # Array length
                        f.write(value.tobytes())
    
    def generate_test_files(self, output_dir, radar_code='sas', num_files=3, beams_per_file=16):
        """Generate multiple test FITACF files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate files for different times
        start_time = datetime(2025, 1, 15, 12, 0, 0)  # Example date
        
        for file_idx in range(num_files):
            timestamp = start_time + timedelta(hours=file_idx * 2)
            
            records = []
            
            # Generate records for each beam in a scan
            for beam in range(beams_per_file):
                beam_time = timestamp + timedelta(seconds=beam * 3)  # 3 seconds per beam
                record = self.generate_fitacf_record(radar_code, beam_time, beam)
                records.append(record)
            
            # Create filename following SuperDARN convention
            filename = f"{timestamp.strftime('%Y%m%d.%H%M.%S')}.{radar_code}.fitacf"
            filepath = os.path.join(output_dir, filename)
            
            self.write_dmap_file(records, filepath)
            
            # Compress the file
            with open(filepath, 'rb') as f_in:
                with gzip.open(filepath + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)
            
            os.remove(filepath)  # Remove uncompressed version
            
            print(f"Generated test file: {filename}.gz")
    
    def generate_summary_report(self, output_dir):
        """Generate a summary report of the test data"""
        
        report_path = os.path.join(output_dir, 'test_data_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("SuperDARN FITACF Test Data Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write("Generated: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write("Purpose: Testing CUDA-accelerated FITACF processing\n\n")
            
            f.write("File Format:\n")
            f.write("- DMAP binary format (simplified for testing)\n")
            f.write("- Compressed with gzip\n")
            f.write("- Follows SuperDARN naming convention\n\n")
            
            f.write("Data Characteristics:\n")
            f.write(f"- Range gates: {self.max_range}\n")
            f.write(f"- Lags per range: {self.max_lags}\n")
            f.write(f"- Beams per scan: 16\n")
            f.write("- Realistic noise levels and physics\n")
            f.write("- Spatial coherence patterns\n\n")
            
            f.write("Processing Notes:\n")
            f.write("- Use these files to test CPU vs CUDA performance\n")
            f.write("- Validate numerical accuracy between implementations\n")
            f.write("- Benchmark processing speed improvements\n")
        
        print(f"Summary report written to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate SuperDARN FITACF test data')
    parser.add_argument('--output-dir', '-o', default='test_data', help='Output directory')
    parser.add_argument('--radar', '-r', default='sas', choices=['sas', 'pgr', 'rkn', 'cly', 'inv'], 
                       help='Radar code')
    parser.add_argument('--num-files', '-n', type=int, default=3, help='Number of files to generate')
    parser.add_argument('--beams', '-b', type=int, default=16, help='Beams per file')
    
    args = parser.parse_args()
    
    print("SuperDARN FITACF Test Data Generator")
    print("=" * 40)
    
    generator = FITACFDataGenerator()
    
    # Generate test files
    generator.generate_test_files(
        output_dir=args.output_dir,
        radar_code=args.radar,
        num_files=args.num_files,
        beams_per_file=args.beams
    )
    
    # Generate summary report
    generator.generate_summary_report(args.output_dir)
    
    print(f"\nTest data generation complete!")
    print(f"Files saved to: {args.output_dir}")
    print(f"Use these files to test the CUDA-accelerated FITACF processing pipeline.")

if __name__ == '__main__':
    main()