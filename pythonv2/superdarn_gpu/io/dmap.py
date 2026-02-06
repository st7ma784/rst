"""
DMAP (DataMap) file reader for SuperDARN data

This module implements reading of RST DMAP format files, which is the
native binary format used by SuperDARN radar data products.

The format consists of:
- A code (65537) and size header
- Number of scalars and arrays
- Scalar definitions and data
- Array definitions and data

References:
- RST source: codebase/general/src.lib/dmap.1.25/src/dmap.c
"""

import struct
import numpy as np
from typing import Dict, Any, List, Tuple, BinaryIO, Optional
from pathlib import Path
from dataclasses import dataclass


# DMAP data types (from dmap.h)
DMAP_TYPES = {
    1: ('char', 'b', 1),       # DATACHAR
    2: ('short', 'h', 2),      # DATASHORT  
    3: ('int', 'i', 4),        # DATAINT
    4: ('float', 'f', 4),      # DATAFLOAT
    8: ('double', 'd', 8),     # DATADOUBLE
    9: ('string', None, None), # DATASTRING
    10: ('long', 'q', 8),      # DATALONG
    16: ('uchar', 'B', 1),     # DATAUCHAR
    17: ('ushort', 'H', 2),    # DATAUSHORT
    18: ('uint', 'I', 4),      # DATAUINT
    19: ('ulong', 'Q', 8),     # DATAULONG
}

# NumPy dtype mapping
NUMPY_DTYPES = {
    1: np.int8,      # char
    2: np.int16,     # short
    3: np.int32,     # int
    4: np.float32,   # float
    8: np.float64,   # double
    10: np.int64,    # long
    16: np.uint8,    # uchar
    17: np.uint16,   # ushort
    18: np.uint32,   # uint
    19: np.uint64,   # ulong
}


@dataclass
class DmapRecord:
    """A single DMAP record containing scalars and arrays"""
    scalars: Dict[str, Any]
    arrays: Dict[str, np.ndarray]
    
    def __getitem__(self, key):
        if key in self.scalars:
            return self.scalars[key]
        if key in self.arrays:
            return self.arrays[key]
        raise KeyError(key)
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        return list(self.scalars.keys()) + list(self.arrays.keys())


class DmapReader:
    """
    Reader for DMAP (DataMap) format files.
    
    Example usage:
        reader = DmapReader('file.cnvmap')
        for record in reader:
            print(record['start.year'], record['start.month'])
    """
    
    def __init__(self, filename: str):
        self.filename = Path(filename)
        self.file = None
        self._records = None
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, *args):
        self.close()
        
    def open(self):
        """Open the file for reading"""
        self.file = open(self.filename, 'rb')
        
    def close(self):
        """Close the file"""
        if self.file:
            self.file.close()
            self.file = None
            
    def __iter__(self):
        """Iterate over all records in the file"""
        if self.file is None:
            self.open()
        
        self.file.seek(0)
        while True:
            try:
                record = self._read_record()
                if record is None:
                    break
                yield record
            except EOFError:
                break
                
    def read_all(self) -> List[DmapRecord]:
        """Read all records from the file"""
        if self._records is not None:
            return self._records
            
        self._records = list(self)
        return self._records
        
    def _read_record(self) -> Optional[DmapRecord]:
        """Read a single DMAP record"""
        # Read code and size
        header = self.file.read(8)
        if len(header) < 8:
            return None
            
        code, size = struct.unpack('<Ii', header)
        
        # Verify code (65537 = valid DMAP record)
        if code != 65537:
            raise ValueError(f"Invalid DMAP code: {code} (expected 65537)")
            
        # Read number of scalars and arrays
        counts = self.file.read(8)
        if len(counts) < 8:
            raise EOFError("Unexpected end of file reading record header")
            
        n_scalars, n_arrays = struct.unpack('<ii', counts)
        
        # Read scalars
        scalars = {}
        for _ in range(n_scalars):
            name, value = self._read_scalar()
            scalars[name] = value
            
        # Read arrays  
        arrays = {}
        for _ in range(n_arrays):
            name, value = self._read_array()
            arrays[name] = value
            
        return DmapRecord(scalars=scalars, arrays=arrays)
        
    def _read_scalar(self) -> Tuple[str, Any]:
        """Read a scalar value"""
        # Read name (null-terminated string)
        name = self._read_string()
        
        # Read type (single byte in DMAP format)
        type_byte = self.file.read(1)
        if not type_byte:
            raise EOFError("Unexpected end of file reading scalar type")
        data_type = type_byte[0]
        
        # Read value based on type
        if data_type == 9:  # String
            value = self._read_string()
        elif data_type in DMAP_TYPES:
            type_name, fmt, size = DMAP_TYPES[data_type]
            if fmt and size:
                data = self.file.read(size)
                value = struct.unpack(f'<{fmt}', data)[0]
            else:
                value = None
        else:
            raise ValueError(f"Unknown DMAP type: {data_type}")
            
        return name, value
        
    def _read_array(self) -> Tuple[str, np.ndarray]:
        """Read an array value"""
        # Read name (null-terminated string)
        name = self._read_string()
        
        # Read type (single byte)
        type_byte = self.file.read(1)
        if not type_byte:
            raise EOFError("Unexpected end of file reading array type")
        data_type = type_byte[0]
        
        # Read dimensions
        dim_data = self.file.read(4)
        n_dims = struct.unpack('<i', dim_data)[0]
        
        dims = []
        for _ in range(n_dims):
            dim_size = struct.unpack('<i', self.file.read(4))[0]
            dims.append(dim_size)
            
        # Calculate total elements
        n_elements = 1
        for d in dims:
            n_elements *= d
            
        # Read array data
        if data_type == 9:  # String array
            values = []
            for _ in range(n_elements):
                values.append(self._read_string())
            arr = np.array(values, dtype=object).reshape(dims)
        elif data_type in NUMPY_DTYPES:
            dtype = NUMPY_DTYPES[data_type]
            _, fmt, size = DMAP_TYPES[data_type]
            data = self.file.read(n_elements * size)
            arr = np.frombuffer(data, dtype=dtype).reshape(dims)
        else:
            raise ValueError(f"Unknown DMAP array type: {data_type}")
            
        return name, arr
        
    def _read_string(self) -> str:
        """Read a null-terminated string"""
        chars = []
        while True:
            c = self.file.read(1)
            if not c or c == b'\x00':
                break
            chars.append(c.decode('latin-1'))
        return ''.join(chars)


def read_dmap(filename: str) -> List[DmapRecord]:
    """
    Read all records from a DMAP file.
    
    Parameters
    ----------
    filename : str
        Path to DMAP file
        
    Returns
    -------
    list
        List of DmapRecord objects
    """
    with DmapReader(filename) as reader:
        return reader.read_all()


def read_convmap(filename: str) -> Dict[str, Any]:
    """
    Read a convection map file and return structured data.
    
    Parameters
    ----------
    filename : str
        Path to convmap file
        
    Returns
    -------
    dict
        Dictionary with convection map data including:
        - start_time, end_time
        - IMF parameters (Bx, By, Bz)
        - Model info
        - Potential data
        - Grid vectors
    """
    records = read_dmap(filename)
    
    if not records:
        raise ValueError("No records found in file")
        
    # For convmap, we usually want the first (or only) record
    # that contains the map data
    result = {
        'records': [],
        'n_records': len(records)
    }
    
    for rec in records:
        record_data = {
            'scalars': dict(rec.scalars),
            'arrays': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in rec.arrays.items()}
        }
        
        # Extract key fields if present
        if 'start.year' in rec.scalars:
            record_data['start_time'] = {
                'year': rec.get('start.year'),
                'month': rec.get('start.month'),
                'day': rec.get('start.day'),
                'hour': rec.get('start.hour'),
                'minute': rec.get('start.minute'),
                'second': rec.get('start.second', 0)
            }
            
        if 'IMF.Bx' in rec.scalars:
            record_data['IMF'] = {
                'Bx': rec.get('IMF.Bx'),
                'By': rec.get('IMF.By'),
                'Bz': rec.get('IMF.Bz'),
                'Vx': rec.get('IMF.Vx'),
                'tilt': rec.get('IMF.tilt'),
                'Kp': rec.get('IMF.Kp')
            }
            
        if 'model.name' in rec.scalars:
            record_data['model'] = {
                'name': rec.get('model.name'),
                'angle': rec.get('model.angle'),
                'level': rec.get('model.level'),
                'tilt': rec.get('model.tilt')
            }
            
        if 'pot.drop' in rec.scalars:
            record_data['potential'] = {
                'drop': rec.get('pot.drop'),
                'drop_err': rec.get('pot.drop.err'),
                'max': rec.get('pot.max'),
                'max_err': rec.get('pot.max.err'),
                'min': rec.get('pot.min'),
                'min_err': rec.get('pot.min.err')
            }
            
        result['records'].append(record_data)
        
    return result


# Test function
def test_reader():
    """Test the DMAP reader with a sample file"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dmap.py <filename>")
        sys.exit(1)
        
    filename = sys.argv[1]
    print(f"Reading: {filename}")
    
    try:
        records = read_dmap(filename)
        print(f"Found {len(records)} records")
        
        for i, rec in enumerate(records[:3]):  # Show first 3 records
            print(f"\nRecord {i}:")
            print(f"  Scalars ({len(rec.scalars)}): {list(rec.scalars.keys())[:10]}...")
            print(f"  Arrays ({len(rec.arrays)}): {list(rec.arrays.keys())[:10]}...")
            
            # Show some common fields
            for field in ['start.year', 'start.month', 'start.day', 
                         'IMF.By', 'IMF.Bz', 'pot.drop']:
                if field in rec.scalars:
                    print(f"    {field}: {rec.scalars[field]}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_reader()
