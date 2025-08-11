"""
GPU memory management utilities for efficient SuperDARN processing
"""

from typing import Optional, Dict, Any, Tuple, Union
import gc
import warnings
from contextlib import contextmanager
from .backends import get_backend, Backend, get_array_module

class GPUMemoryManager:
    """
    Manages GPU memory allocation and pools for efficient processing
    """
    
    def __init__(self):
        self.pools = {}
        self.stats = {
            'peak_usage': 0,
            'current_usage': 0,
            'allocations': 0,
            'deallocations': 0
        }
        
    def get_memory_info(self) -> Dict[str, int]:
        """
        Get current GPU memory usage information
        
        Returns
        -------
        dict
            Memory usage statistics in bytes
        """
        if get_backend() != Backend.CUPY:
            return {'total': 0, 'free': 0, 'used': 0}
        
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            device = cp.cuda.Device()
            
            total_bytes = device.mem_info[1]  # Total GPU memory
            used_pool = mempool.used_bytes()
            free_pool = mempool.free_bytes()
            
            return {
                'total': total_bytes,
                'used_pool': used_pool,
                'free_pool': free_pool,
                'available': total_bytes - used_pool,
                'pool_limit': mempool.get_limit() if hasattr(mempool, 'get_limit') else -1
            }
        except Exception as e:
            warnings.warn(f"Could not get GPU memory info: {e}")
            return {'total': 0, 'free': 0, 'used': 0}
    
    def get_memory_usage_percent(self) -> float:
        """Get current GPU memory usage as percentage"""
        info = self.get_memory_info()
        if info['total'] == 0:
            return 0.0
        return (info['used_pool'] / info['total']) * 100.0
    
    def optimize_memory(self):
        """Optimize GPU memory usage by cleaning up pools"""
        if get_backend() != Backend.CUPY:
            return
        
        try:
            import cupy as cp
            
            # Force garbage collection
            gc.collect()
            
            # Free all unused memory from pool
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            
            # Clear pinned memory as well
            if hasattr(cp.cuda, 'PinnedMemoryPool'):
                pinned_pool = cp.get_default_pinned_memory_pool()
                pinned_pool.free_all_blocks()
                
        except Exception as e:
            warnings.warn(f"Could not optimize GPU memory: {e}")
    
    def set_memory_limit(self, limit_bytes: Optional[int] = None, limit_fraction: Optional[float] = None):
        """
        Set GPU memory pool limit
        
        Parameters
        ----------
        limit_bytes : int, optional
            Absolute limit in bytes
        limit_fraction : float, optional
            Limit as fraction of total GPU memory (0.0 to 1.0)
        """
        if get_backend() != Backend.CUPY:
            return
        
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            
            if limit_bytes is not None:
                mempool.set_limit(size=limit_bytes)
            elif limit_fraction is not None:
                if not 0.0 <= limit_fraction <= 1.0:
                    raise ValueError("limit_fraction must be between 0.0 and 1.0")
                device = cp.cuda.Device()
                total_memory = device.mem_info[1]
                limit_bytes = int(total_memory * limit_fraction)
                mempool.set_limit(size=limit_bytes)
            else:
                # Remove limit
                mempool.set_limit()
                
        except Exception as e:
            warnings.warn(f"Could not set memory limit: {e}")

# Global memory manager instance
memory_pool = GPUMemoryManager()

@contextmanager
def gpu_memory_context(limit_fraction: float = 0.8):
    """
    Context manager for GPU memory management
    
    Parameters
    ----------
    limit_fraction : float
        Maximum fraction of GPU memory to use
    """
    original_limit = None
    
    try:
        if get_backend() == Backend.CUPY:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            if hasattr(mempool, 'get_limit'):
                original_limit = mempool.get_limit()
            
            # Set new limit
            memory_pool.set_memory_limit(limit_fraction=limit_fraction)
        
        yield memory_pool
        
    finally:
        # Restore original limit and cleanup
        if get_backend() == Backend.CUPY and original_limit is not None:
            try:
                import cupy as cp
                mempool = cp.get_default_memory_pool()
                if original_limit == -1:
                    mempool.set_limit()  # Remove limit
                else:
                    mempool.set_limit(size=original_limit)
            except:
                pass
        
        memory_pool.optimize_memory()

def estimate_memory_requirement(nrang: int, mplgs: int, nave: int = 1, 
                               num_beams: int = 16, dtype_size: int = 4) -> int:
    """
    Estimate memory requirements for SuperDARN processing
    
    Parameters
    ----------
    nrang : int
        Number of range gates
    mplgs : int  
        Number of lags
    nave : int
        Number of averages
    num_beams : int
        Number of beams
    dtype_size : int
        Size of data type in bytes (4 for float32/complex64)
        
    Returns
    -------
    int
        Estimated memory requirement in bytes
    """
    # RawACF data
    rawacf_size = nrang * mplgs * num_beams * dtype_size * 2  # Complex data
    rawacf_size += nrang * num_beams * dtype_size * 3  # Power, noise, flags
    
    # FitACF data (fitted parameters)
    fitacf_size = nrang * num_beams * dtype_size * 8  # Main parameters + errors
    
    # Intermediate processing arrays (estimate 3x data size)
    intermediate_size = (rawacf_size + fitacf_size) * 3
    
    # Total with safety margin
    total_size = (rawacf_size + fitacf_size + intermediate_size) * 1.5
    
    return int(total_size)

def check_memory_availability(required_bytes: int, safety_factor: float = 0.8) -> bool:
    """
    Check if enough GPU memory is available for processing
    
    Parameters
    ----------
    required_bytes : int
        Required memory in bytes
    safety_factor : float
        Safety factor (0.0 to 1.0) to avoid OOM
        
    Returns
    -------
    bool
        True if enough memory is available
    """
    if get_backend() != Backend.CUPY:
        return True  # CPU processing - assume sufficient RAM
    
    info = memory_pool.get_memory_info()
    available = info.get('available', 0) * safety_factor
    
    return available >= required_bytes

def suggest_batch_size(total_records: int, record_memory: int, 
                      target_memory_usage: float = 0.6) -> int:
    """
    Suggest optimal batch size for processing
    
    Parameters
    ----------
    total_records : int
        Total number of records to process
    record_memory : int
        Memory requirement per record in bytes
    target_memory_usage : float
        Target GPU memory usage fraction
        
    Returns
    -------
    int
        Suggested batch size
    """
    if get_backend() != Backend.CUPY:
        return min(total_records, 100)  # Conservative batch size for CPU
    
    info = memory_pool.get_memory_info()
    available = info.get('total', 0) * target_memory_usage
    
    if record_memory == 0:
        return total_records
    
    batch_size = max(1, int(available // record_memory))
    return min(batch_size, total_records)

class MemoryMonitor:
    """Monitor GPU memory usage during processing"""
    
    def __init__(self, name: str = "Processing"):
        self.name = name
        self.start_memory = 0
        self.peak_memory = 0
        
    def __enter__(self):
        if get_backend() == Backend.CUPY:
            self.start_memory = memory_pool.get_memory_info()['used_pool']
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if get_backend() == Backend.CUPY:
            current_memory = memory_pool.get_memory_info()['used_pool']
            memory_used = current_memory - self.start_memory
            
            if memory_used > 1024**2:  # > 1 MB
                print(f"{self.name} used {memory_used / (1024**2):.1f} MB GPU memory")

def gpu_array_cache(maxsize: int = 128):
    """
    Decorator to cache GPU arrays and avoid repeated allocations
    
    Parameters
    ----------
    maxsize : int
        Maximum cache size
    """
    from functools import lru_cache
    
    def decorator(func):
        if get_backend() == Backend.CUPY:
            return lru_cache(maxsize=maxsize)(func)
        else:
            return func  # No caching for CPU
    
    return decorator