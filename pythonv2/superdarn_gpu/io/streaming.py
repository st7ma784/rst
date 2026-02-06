"""
Streaming data loaders for SuperDARN files

Provides efficient batch loading and streaming for large datasets
that don't fit in memory.
"""

from typing import Any, Union, Iterator, Optional, List
from pathlib import Path
import numpy as np


class DataStreamer:
    """
    Stream data records from SuperDARN files
    
    Provides memory-efficient iteration over large files by reading
    records on demand rather than loading all at once.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the data file
    chunk_size : int, optional
        Number of records to buffer (default: 100)
    preload : bool, optional
        Whether to preload first chunk (default: True)
        
    Examples
    --------
    >>> streamer = DataStreamer('data.rawacf')
    >>> for record in streamer:
    ...     process(record)
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        chunk_size: int = 100,
        preload: bool = True
    ):
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self._buffer: List[Any] = []
        self._position = 0
        self._exhausted = False
        self._file_handle = None
        
        if preload:
            self._load_chunk()
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over all records"""
        self._position = 0
        self._exhausted = False
        return self
    
    def __next__(self) -> Any:
        """Get next record"""
        if self._position >= len(self._buffer):
            if self._exhausted:
                raise StopIteration
            self._load_chunk()
            if not self._buffer:
                raise StopIteration
            self._position = 0
            
        record = self._buffer[self._position]
        self._position += 1
        return record
    
    def __enter__(self) -> 'DataStreamer':
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.close()
    
    def _load_chunk(self) -> None:
        """Load next chunk of records into buffer"""
        # Placeholder implementation
        # In production, this would read from actual file
        self._buffer = []
        self._exhausted = True
    
    def close(self) -> None:
        """Close file handle and release resources"""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        self._buffer = []
    
    def skip(self, n: int) -> None:
        """Skip n records"""
        for _ in range(n):
            try:
                next(self)
            except StopIteration:
                break
    
    def peek(self) -> Optional[Any]:
        """Peek at next record without advancing"""
        if self._position >= len(self._buffer):
            if self._exhausted:
                return None
            self._load_chunk()
            self._position = 0
        
        if self._buffer:
            return self._buffer[self._position]
        return None


class BatchLoader:
    """
    Load data in batches for parallel processing
    
    Designed for GPU processing where loading data in larger batches
    is more efficient than single records.
    
    Parameters
    ----------
    filepaths : list
        List of file paths to load
    batch_size : int, optional
        Number of records per batch (default: 1000)
    shuffle : bool, optional
        Whether to shuffle records (default: False)
    backend : str, optional
        Backend for arrays ('numpy' or 'cupy')
        
    Examples
    --------
    >>> loader = BatchLoader(['file1.rawacf', 'file2.rawacf'], batch_size=500)
    >>> for batch in loader:
    ...     results = processor.process_batch(batch)
    """
    
    def __init__(
        self,
        filepaths: List[Union[str, Path]],
        batch_size: int = 1000,
        shuffle: bool = False,
        backend: str = 'numpy'
    ):
        self.filepaths = [Path(f) for f in filepaths]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.backend = backend
        
        self._data: List[Any] = []
        self._current_batch = 0
        self._total_batches = 0
        
    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches"""
        self._load_all_data()
        self._current_batch = 0
        return self
    
    def __next__(self) -> Any:
        """Get next batch"""
        start = self._current_batch * self.batch_size
        end = start + self.batch_size
        
        if start >= len(self._data):
            raise StopIteration
        
        batch = self._data[start:end]
        self._current_batch += 1
        return self._convert_batch(batch)
    
    def __len__(self) -> int:
        """Number of batches"""
        if not self._data:
            self._load_all_data()
        return (len(self._data) + self.batch_size - 1) // self.batch_size
    
    def _load_all_data(self) -> None:
        """Load all data from files"""
        from .readers import load
        
        self._data = []
        for filepath in self.filepaths:
            try:
                data = load(filepath)
                if hasattr(data, '__iter__') and not isinstance(data, dict):
                    self._data.extend(data)
                else:
                    self._data.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
        
        if self.shuffle:
            np.random.shuffle(self._data)
        
        self._total_batches = len(self)
    
    def _convert_batch(self, batch: List[Any]) -> Any:
        """Convert batch to appropriate backend format"""
        if self.backend == 'cupy':
            try:
                import cupy as cp
                return [self._to_cupy(item) for item in batch]
            except ImportError:
                pass
        return batch
    
    @staticmethod
    def _to_cupy(item: Any) -> Any:
        """Convert numpy arrays in item to cupy"""
        import cupy as cp
        
        if isinstance(item, np.ndarray):
            return cp.asarray(item)
        elif hasattr(item, '__dict__'):
            for key, value in item.__dict__.items():
                if isinstance(value, np.ndarray):
                    setattr(item, key, cp.asarray(value))
        return item


def stream_files(
    filepaths: List[Union[str, Path]],
    chunk_size: int = 100
) -> Iterator[Any]:
    """
    Stream records from multiple files
    
    Convenience function for iterating over multiple files
    without loading all into memory.
    
    Parameters
    ----------
    filepaths : list
        List of file paths
    chunk_size : int, optional
        Records to buffer per file
        
    Yields
    ------
    record
        Individual data records
    """
    for filepath in filepaths:
        with DataStreamer(filepath, chunk_size=chunk_size) as streamer:
            yield from streamer
