"""
Backend management for CUPy/NumPy abstraction
"""

from typing import Union, Optional, Any
from enum import Enum
import warnings

# Global backend state
_current_backend = None
_backend_module = None

class Backend(Enum):
    """Available computation backends"""
    CUPY = "cupy"
    NUMPY = "numpy"

def get_backend() -> Backend:
    """Get current computation backend"""
    global _current_backend
    if _current_backend is None:
        set_backend("cupy")  # Try GPU first by default
    return _current_backend

def set_backend(backend: Union[str, Backend]) -> Backend:
    """
    Set computation backend
    
    Parameters
    ----------
    backend : str or Backend
        Backend to use ('cupy' or 'numpy')
        
    Returns
    -------
    Backend
        The actual backend that was set
    """
    global _current_backend, _backend_module
    
    if isinstance(backend, str):
        backend = Backend(backend.lower())
    
    if backend == Backend.CUPY:
        try:
            import cupy as cp
            if cp.cuda.is_available():
                _backend_module = cp
                _current_backend = Backend.CUPY
                return Backend.CUPY
            else:
                warnings.warn("CUDA not available, falling back to NumPy")
                return set_backend(Backend.NUMPY)
        except ImportError:
            warnings.warn("CuPy not installed, falling back to NumPy")
            return set_backend(Backend.NUMPY)
    
    elif backend == Backend.NUMPY:
        import numpy as np
        _backend_module = np
        _current_backend = Backend.NUMPY
        return Backend.NUMPY
    
    else:
        raise ValueError(f"Unknown backend: {backend}")

def get_array_module() -> Any:
    """Get the current array module (cupy or numpy)"""
    global _backend_module
    if _backend_module is None:
        get_backend()  # Initialize backend
    return _backend_module

def is_gpu_backend() -> bool:
    """Check if current backend uses GPU"""
    return get_backend() == Backend.CUPY

def ensure_array(data: Any, backend: Optional[Backend] = None) -> Any:
    """
    Ensure data is an array on the specified backend
    
    Parameters
    ----------
    data : array_like
        Input data
    backend : Backend, optional
        Target backend. If None, uses current backend
        
    Returns
    -------
    array
        Array on the specified backend
    """
    if backend is None:
        backend = get_backend()
    
    xp = get_array_module() if backend == get_backend() else (
        __import__('cupy') if backend == Backend.CUPY else __import__('numpy')
    )
    
    if backend == Backend.CUPY:
        # Convert to GPU array
        if hasattr(data, '__cuda_array_interface__'):
            return data  # Already on GPU
        else:
            return xp.asarray(data)
    else:
        # Convert to CPU array
        if hasattr(data, '__cuda_array_interface__'):
            return xp.asnumpy(data)  # GPU to CPU
        else:
            return xp.asarray(data)  # Ensure NumPy array

def to_cpu(data: Any) -> Any:
    """Transfer data to CPU (NumPy array)"""
    if hasattr(data, '__cuda_array_interface__'):
        import cupy as cp
        return cp.asnumpy(data)
    return data

def to_gpu(data: Any) -> Any:
    """Transfer data to GPU (CuPy array)"""
    if get_backend() != Backend.CUPY:
        warnings.warn("GPU backend not available, returning original data")
        return data
    
    import cupy as cp
    if hasattr(data, '__cuda_array_interface__'):
        return data  # Already on GPU
    return cp.asarray(data)

def synchronize():
    """Synchronize GPU operations"""
    if is_gpu_backend():
        import cupy as cp
        cp.cuda.Stream.null.synchronize()

class BackendContext:
    """Context manager for temporary backend switching"""
    
    def __init__(self, backend: Union[str, Backend]):
        self.target_backend = Backend(backend) if isinstance(backend, str) else backend
        self.original_backend = None
    
    def __enter__(self):
        self.original_backend = get_backend()
        set_backend(self.target_backend)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_backend is not None:
            set_backend(self.original_backend)