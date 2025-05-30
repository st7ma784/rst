#!/bin/bash
# Generate test FITACF data for performance benchmarking
# Usage: ./generate_test_fitacf_data.sh [small|medium|large]

set -e

SIZE=${1:-medium}
OUTPUT_DIR="/workspace/test-data"

mkdir -p $OUTPUT_DIR

case $SIZE in
    "small")
        BEAMS=8
        RANGES=75
        RECORDS=100
        ;;
    "medium")
        BEAMS=16
        RANGES=100
        RECORDS=500
        ;;
    "large")
        BEAMS=24
        RANGES=150
        RECORDS=1000
        ;;
    *)
        echo "Usage: $0 [small|medium|large]"
        exit 1
        ;;
esac

echo "Generating $SIZE test data: $BEAMS beams, $RANGES ranges, $RECORDS records"

# Create synthetic FITACF data for testing
python3 << EOF
import struct
import random
import time
import os

def generate_fitacf_data(filename, beams, ranges, records):
    """Generate synthetic FITACF data for testing"""
    with open(filename, 'wb') as f:
        for record in range(records):
            # Generate synthetic radar parameters
            beam = record % beams
            
            # Write basic header (simplified)
            f.write(struct.pack('<I', 0x12345678))  # Magic number
            f.write(struct.pack('<H', beam))        # Beam number
            f.write(struct.pack('<H', ranges))      # Number of ranges
            
            # Generate synthetic ACF data
            for r in range(ranges):
                # Quality flag (random, biased toward valid data)
                qflg = 1 if random.random() > 0.3 else 0
                f.write(struct.pack('<B', qflg))
                
                if qflg:
                    # Velocity (m/s)
                    velocity = random.gauss(0, 500)
                    f.write(struct.pack('<f', velocity))
                    
                    # Power (dB)
                    power = random.gauss(10, 5)
                    f.write(struct.pack('<f', power))
                    
                    # Width (m/s)
                    width = random.gauss(100, 50)
                    f.write(struct.pack('<f', width))
                else:
                    # Invalid data
                    f.write(struct.pack('<f', 0.0))
                    f.write(struct.pack('<f', 0.0))
                    f.write(struct.pack('<f', 0.0))

# Generate test data
generate_fitacf_data(
    '/workspace/test_data_${SIZE}.fitacf',
    ${BEAMS},
    ${RANGES}, 
    ${RECORDS}
)

print(f"Generated test data: ${SIZE}")
print(f"  File: /workspace/test_data_${SIZE}.fitacf")
print(f"  Beams: ${BEAMS}")
print(f"  Ranges: ${RANGES}")
print(f"  Records: ${RECORDS}")
EOF

echo "Test data generation complete!"
