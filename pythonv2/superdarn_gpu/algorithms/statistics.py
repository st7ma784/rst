"""
Statistical processing algorithms for SuperDARN data

Provides GPU-accelerated statistical operations including
error estimation, confidence intervals, and noise analysis.
"""

from typing import Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from ..core.backends import get_array_module, ensure_array


@dataclass
class StatisticalResult:
    """Container for statistical analysis results"""
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray
    skewness: Optional[np.ndarray] = None
    kurtosis: Optional[np.ndarray] = None
    confidence_low: Optional[np.ndarray] = None
    confidence_high: Optional[np.ndarray] = None


class StatisticalProcessor:
    """
    GPU-accelerated statistical processing
    
    Provides efficient calculation of statistics over large datasets
    with optional CUDA acceleration.
    
    Parameters
    ----------
    use_gpu : bool, optional
        Use GPU acceleration if available
    confidence_level : float, optional
        Confidence level for interval calculations (default: 0.95)
        
    Examples
    --------
    >>> processor = StatisticalProcessor(use_gpu=True)
    >>> result = processor.compute_statistics(data)
    >>> print(f"Mean: {result.mean}, Std: {result.std}")
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        confidence_level: float = 0.95
    ):
        self.use_gpu = use_gpu
        self.confidence_level = confidence_level
        self.xp = get_array_module()
    
    def compute_statistics(
        self,
        data: np.ndarray,
        axis: Optional[int] = None,
        ddof: int = 1
    ) -> StatisticalResult:
        """
        Compute basic statistics over data
        
        Parameters
        ----------
        data : array
            Input data array
        axis : int, optional
            Axis along which to compute (None for all)
        ddof : int
            Delta degrees of freedom for variance
            
        Returns
        -------
        StatisticalResult
            Container with mean, std, variance, etc.
        """
        xp = self.xp
        data = ensure_array(data)
        
        mean = xp.mean(data, axis=axis)
        variance = xp.var(data, axis=axis, ddof=ddof)
        std = xp.sqrt(variance)
        
        # Higher-order moments (optional)
        if data.size > 100:  # Only for sufficient data
            centered = data - xp.expand_dims(mean, axis=axis) if axis is not None else data - mean
            normalized = centered / (xp.expand_dims(std, axis=axis) if axis is not None else std)
            
            skewness = xp.mean(normalized ** 3, axis=axis)
            kurtosis = xp.mean(normalized ** 4, axis=axis) - 3
        else:
            skewness = None
            kurtosis = None
        
        # Confidence intervals
        n = data.shape[axis] if axis is not None else data.size
        z_value = self._get_z_value()
        margin = z_value * std / xp.sqrt(n)
        
        return StatisticalResult(
            mean=self._to_numpy(mean),
            std=self._to_numpy(std),
            variance=self._to_numpy(variance),
            skewness=self._to_numpy(skewness) if skewness is not None else None,
            kurtosis=self._to_numpy(kurtosis) if kurtosis is not None else None,
            confidence_low=self._to_numpy(mean - margin),
            confidence_high=self._to_numpy(mean + margin)
        )
    
    def weighted_mean(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        axis: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute weighted mean
        
        Parameters
        ----------
        data : array
            Input data
        weights : array
            Weights (same shape as data)
        axis : int, optional
            Axis along which to compute
            
        Returns
        -------
        array
            Weighted mean
        """
        xp = self.xp
        data = ensure_array(data)
        weights = ensure_array(weights)
        
        weighted_sum = xp.sum(data * weights, axis=axis)
        weight_sum = xp.sum(weights, axis=axis)
        
        return self._to_numpy(weighted_sum / weight_sum)
    
    def weighted_std(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        axis: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute weighted standard deviation
        
        Parameters
        ----------
        data : array
            Input data
        weights : array
            Weights
        axis : int, optional
            Axis along which to compute
            
        Returns
        -------
        array
            Weighted standard deviation
        """
        xp = self.xp
        data = ensure_array(data)
        weights = ensure_array(weights)
        
        # Weighted mean
        w_mean = self.weighted_mean(data, weights, axis)
        if axis is not None:
            w_mean = xp.expand_dims(xp.asarray(w_mean), axis=axis)
        
        # Weighted variance
        squared_diff = (data - w_mean) ** 2
        weighted_var = xp.sum(weights * squared_diff, axis=axis) / xp.sum(weights, axis=axis)
        
        return self._to_numpy(xp.sqrt(weighted_var))
    
    def estimate_noise(
        self,
        data: np.ndarray,
        method: str = 'mad'
    ) -> float:
        """
        Estimate noise level in data
        
        Parameters
        ----------
        data : array
            Input data
        method : str
            Method: 'mad' (median absolute deviation), 'std', 'iqr'
            
        Returns
        -------
        float
            Estimated noise level
        """
        xp = self.xp
        data = ensure_array(data).ravel()
        
        if method == 'mad':
            # Median absolute deviation (robust)
            median = xp.median(data)
            mad = xp.median(xp.abs(data - median))
            # Scale factor for normal distribution
            noise = float(1.4826 * self._to_numpy(mad))
        elif method == 'std':
            noise = float(self._to_numpy(xp.std(data)))
        elif method == 'iqr':
            # Interquartile range
            q75, q25 = xp.percentile(data, [75, 25])
            noise = float(self._to_numpy((q75 - q25) / 1.349))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return noise
    
    def compute_error_bars(
        self,
        data: np.ndarray,
        errors: np.ndarray,
        method: str = 'propagation'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute error bars for derived quantities
        
        Parameters
        ----------
        data : array
            Data values
        errors : array
            Uncertainty estimates
        method : str
            Method: 'propagation', 'bootstrap', 'analytical'
            
        Returns
        -------
        tuple
            (lower_error, upper_error) arrays
        """
        xp = self.xp
        data = ensure_array(data)
        errors = ensure_array(errors)
        
        if method == 'propagation':
            # Simple Gaussian error propagation
            lower = self._to_numpy(errors)
            upper = self._to_numpy(errors)
        elif method == 'bootstrap':
            # Bootstrap resampling
            n_bootstrap = 1000
            n = data.size
            
            bootstrap_means = xp.zeros(n_bootstrap)
            for i in range(n_bootstrap):
                indices = xp.random.randint(0, n, n)
                bootstrap_means[i] = xp.mean(data.ravel()[indices])
            
            lower = self._to_numpy(xp.abs(xp.mean(data) - xp.percentile(bootstrap_means, 16)))
            upper = self._to_numpy(xp.abs(xp.percentile(bootstrap_means, 84) - xp.mean(data)))
        else:
            lower = self._to_numpy(errors)
            upper = self._to_numpy(errors)
        
        return lower, upper
    
    def _get_z_value(self) -> float:
        """Get z-value for confidence level"""
        # Common confidence levels
        z_table = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }
        return z_table.get(self.confidence_level, 1.960)
    
    def _to_numpy(self, arr):
        """Convert array to numpy if needed"""
        if arr is None:
            return None
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
