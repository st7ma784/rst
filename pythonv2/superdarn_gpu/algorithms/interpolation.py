"""
GPU-accelerated spatial interpolation algorithms for SuperDARN processing
"""

from typing import Optional, Dict, Any, Tuple, Union
import warnings
from enum import Enum

import numpy as np

from ..core.backends import get_array_module, get_backend, Backend

class InterpolationMethod(Enum):
    """Available interpolation methods"""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    IDW = "inverse_distance_weighted"
    KRIGING = "kriging"
    RBF = "radial_basis_function"

class SpatialInterpolator:
    """
    GPU-accelerated spatial interpolation for SuperDARN data
    """
    
    def __init__(self):
        self.xp = get_array_module()
        
        # Initialize GPU resources
        if get_backend() == Backend.CUPY:
            self._init_gpu_kernels()
    
    def _init_gpu_kernels(self):
        """Initialize CUDA kernels for interpolation"""
        try:
            import cupy as cp
            
            # Inverse Distance Weighting kernel
            self.idw_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void idw_interpolation(const float* x_points, const float* y_points, const float* values,
                                 const float* x_grid, const float* y_grid, float* result,
                                 int n_points, int n_grid, float power, float radius) {
                
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= n_grid) return;
                
                float target_x = x_grid[idx];
                float target_y = y_grid[idx];
                
                float weighted_sum = 0.0f;
                float weight_sum = 0.0f;
                int valid_points = 0;
                
                for (int i = 0; i < n_points; i++) {
                    float dx = x_points[i] - target_x;
                    float dy = y_points[i] - target_y;
                    float distance = sqrtf(dx*dx + dy*dy);
                    
                    if (distance <= radius) {
                        if (distance < 1e-6f) {
                            // Point coincides with grid point
                            result[idx] = values[i];
                            return;
                        }
                        
                        float weight = powf(1.0f / distance, power);
                        weighted_sum += values[i] * weight;
                        weight_sum += weight;
                        valid_points++;
                    }
                }
                
                if (valid_points > 0 && weight_sum > 0) {
                    result[idx] = weighted_sum / weight_sum;
                } else {
                    result[idx] = NAN;
                }
            }
            ''', 'idw_interpolation')
            
            # Radial Basis Function kernel (thin plate spline)
            self.rbf_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void rbf_interpolation(const float* x_points, const float* y_points, 
                                 const float* coefficients,
                                 const float* x_grid, const float* y_grid, float* result,
                                 int n_points, int n_grid) {
                
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= n_grid) return;
                
                float target_x = x_grid[idx];
                float target_y = y_grid[idx];
                
                float sum = coefficients[n_points]; // Linear term for x
                sum += coefficients[n_points + 1] * target_x; // Linear term for y  
                sum += coefficients[n_points + 2] * target_y; // Constant term
                
                // RBF terms
                for (int i = 0; i < n_points; i++) {
                    float dx = x_points[i] - target_x;
                    float dy = y_points[i] - target_y;
                    float r = sqrtf(dx*dx + dy*dy);
                    
                    if (r > 1e-6f) {
                        // Thin plate spline: r^2 * log(r)
                        float rbf_value = r * r * logf(r);
                        sum += coefficients[i] * rbf_value;
                    }
                }
                
                result[idx] = sum;
            }
            ''', 'rbf_interpolation')
            
            print("Spatial interpolation GPU kernels initialized")
            
        except Exception as e:
            warnings.warn(f"Could not initialize interpolation GPU kernels: {e}")
            self.idw_kernel = None
            self.rbf_kernel = None
    
    def interpolate(self, 
                   points: Tuple[Any, Any], 
                   values: Any,
                   grid_points: Tuple[Any, Any],
                   method: InterpolationMethod = InterpolationMethod.IDW,
                   **kwargs) -> Any:
        """
        Perform spatial interpolation
        
        Parameters
        ----------
        points : Tuple[array, array]
            (x, y) coordinates of data points
        values : array
            Values at data points
        grid_points : Tuple[array, array]
            (x, y) coordinates of grid points
        method : InterpolationMethod
            Interpolation method to use
        **kwargs
            Method-specific parameters
            
        Returns
        -------
        array
            Interpolated values at grid points
        """
        
        if method == InterpolationMethod.IDW:
            return self._idw_interpolation(points, values, grid_points, **kwargs)
        elif method == InterpolationMethod.RBF:
            return self._rbf_interpolation(points, values, grid_points, **kwargs)
        elif method == InterpolationMethod.LINEAR:
            return self._linear_interpolation(points, values, grid_points, **kwargs)
        elif method == InterpolationMethod.CUBIC:
            return self._cubic_interpolation(points, values, grid_points, **kwargs)
        elif method == InterpolationMethod.NEAREST:
            return self._nearest_interpolation(points, values, grid_points, **kwargs)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
    
    def _idw_interpolation(self, 
                          points: Tuple[Any, Any], 
                          values: Any,
                          grid_points: Tuple[Any, Any],
                          power: float = 2.0,
                          radius: float = float('inf'),
                          **kwargs) -> Any:
        """Inverse Distance Weighting interpolation"""
        
        x_points, y_points = points
        x_grid, y_grid = grid_points
        
        if (get_backend() == Backend.CUPY and 
            self.idw_kernel is not None):
            return self._idw_gpu(x_points, y_points, values, x_grid, y_grid, power, radius)
        else:
            return self._idw_cpu(x_points, y_points, values, x_grid, y_grid, power, radius)
    
    def _idw_gpu(self, x_points, y_points, values, x_grid, y_grid, power, radius):
        """GPU implementation of IDW"""
        import cupy as cp
        
        n_points = len(x_points)
        n_grid = len(x_grid)
        
        result = self.xp.zeros(n_grid, dtype=self.xp.float32)
        
        # Flatten grid coordinates if needed
        x_grid_flat = x_grid.ravel() if x_grid.ndim > 1 else x_grid
        y_grid_flat = y_grid.ravel() if y_grid.ndim > 1 else y_grid
        
        block_size = 256
        grid_size = (n_grid + block_size - 1) // block_size
        
        self.idw_kernel(
            (grid_size,), (block_size,),
            (x_points, y_points, values, x_grid_flat, y_grid_flat, result,
             n_points, n_grid, power, radius)
        )
        
        # Reshape result if grid was multi-dimensional
        if x_grid.ndim > 1:
            result = result.reshape(x_grid.shape)
        
        return result
    
    def _idw_cpu(self, x_points, y_points, values, x_grid, y_grid, power, radius):
        """CPU implementation of IDW"""
        
        result = self.xp.full(x_grid.shape, self.xp.nan, dtype=self.xp.float32)
        
        x_grid_flat = x_grid.ravel()
        y_grid_flat = y_grid.ravel()
        result_flat = result.ravel()
        
        for i in range(len(x_grid_flat)):
            target_x = x_grid_flat[i]
            target_y = y_grid_flat[i]
            
            # Calculate distances
            dx = x_points - target_x
            dy = y_points - target_y
            distances = self.xp.sqrt(dx**2 + dy**2)
            
            # Find points within radius
            within_radius = distances <= radius
            nearby_distances = distances[within_radius]
            nearby_values = values[within_radius]
            
            if len(nearby_distances) > 0:
                # Check for exact match
                exact_match = nearby_distances < 1e-6
                if self.xp.any(exact_match):
                    result_flat[i] = nearby_values[exact_match][0]
                else:
                    # IDW calculation
                    weights = 1.0 / (nearby_distances ** power)
                    result_flat[i] = self.xp.sum(nearby_values * weights) / self.xp.sum(weights)
        
        return result.reshape(x_grid.shape)
    
    def _rbf_interpolation(self, 
                          points: Tuple[Any, Any], 
                          values: Any,
                          grid_points: Tuple[Any, Any],
                          function: str = 'thin_plate',
                          **kwargs) -> Any:
        """Radial Basis Function interpolation"""
        
        x_points, y_points = points
        x_grid, y_grid = grid_points
        
        # Solve for RBF coefficients
        coefficients = self._solve_rbf_system(x_points, y_points, values, function)
        
        if (get_backend() == Backend.CUPY and 
            self.rbf_kernel is not None and 
            function == 'thin_plate'):
            return self._rbf_gpu(x_points, y_points, coefficients, x_grid, y_grid)
        else:
            return self._rbf_cpu(x_points, y_points, coefficients, x_grid, y_grid, function)
    
    def _solve_rbf_system(self, x_points, y_points, values, function):
        """Solve linear system for RBF coefficients"""
        
        n_points = len(x_points)
        
        # Build coefficient matrix
        A = self.xp.zeros((n_points + 3, n_points + 3), dtype=self.xp.float32)
        
        # Fill RBF matrix
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    dx = x_points[i] - x_points[j]
                    dy = y_points[i] - y_points[j]
                    r = self.xp.sqrt(dx**2 + dy**2)
                    
                    if function == 'thin_plate' and r > 1e-6:
                        A[i, j] = r**2 * self.xp.log(r)
                    elif function == 'multiquadric':
                        A[i, j] = self.xp.sqrt(1 + r**2)
                    elif function == 'gaussian':
                        A[i, j] = self.xp.exp(-r**2)
        
        # Add polynomial terms
        A[:n_points, n_points] = 1.0  # Constant
        A[:n_points, n_points + 1] = x_points  # Linear x
        A[:n_points, n_points + 2] = y_points  # Linear y
        
        A[n_points:, :n_points] = A[:n_points, n_points:].T  # Symmetry
        
        # Right hand side
        b = self.xp.zeros(n_points + 3, dtype=self.xp.float32)
        b[:n_points] = values
        
        # Solve system
        try:
            coefficients = self.xp.linalg.solve(A, b)
        except:
            # Fallback to least squares if singular
            coefficients = self.xp.linalg.lstsq(A, b, rcond=None)[0]
        
        return coefficients
    
    def _rbf_gpu(self, x_points, y_points, coefficients, x_grid, y_grid):
        """GPU implementation of RBF evaluation"""
        import cupy as cp
        
        n_points = len(x_points)
        n_grid = x_grid.size
        
        result = self.xp.zeros(n_grid, dtype=self.xp.float32)
        
        x_grid_flat = x_grid.ravel()
        y_grid_flat = y_grid.ravel()
        
        block_size = 256
        grid_size = (n_grid + block_size - 1) // block_size
        
        self.rbf_kernel(
            (grid_size,), (block_size,),
            (x_points, y_points, coefficients, x_grid_flat, y_grid_flat, result,
             n_points, n_grid)
        )
        
        return result.reshape(x_grid.shape)
    
    def _rbf_cpu(self, x_points, y_points, coefficients, x_grid, y_grid, function):
        """CPU implementation of RBF evaluation"""
        
        result = self.xp.zeros(x_grid.shape, dtype=self.xp.float32)
        
        x_grid_flat = x_grid.ravel()
        y_grid_flat = y_grid.ravel()
        result_flat = result.ravel()
        
        n_points = len(x_points)
        
        for i in range(len(x_grid_flat)):
            target_x = x_grid_flat[i]
            target_y = y_grid_flat[i]
            
            # Polynomial terms
            value = coefficients[n_points]  # Constant
            value += coefficients[n_points + 1] * target_x  # Linear x
            value += coefficients[n_points + 2] * target_y  # Linear y
            
            # RBF terms
            for j in range(n_points):
                dx = x_points[j] - target_x
                dy = y_points[j] - target_y
                r = self.xp.sqrt(dx**2 + dy**2)
                
                if r > 1e-6:
                    if function == 'thin_plate':
                        rbf_value = r**2 * self.xp.log(r)
                    elif function == 'multiquadric':
                        rbf_value = self.xp.sqrt(1 + r**2)
                    elif function == 'gaussian':
                        rbf_value = self.xp.exp(-r**2)
                    else:
                        rbf_value = 0.0
                    
                    value += coefficients[j] * rbf_value
            
            result_flat[i] = value
        
        return result.reshape(x_grid.shape)
    
    def _linear_interpolation(self, points, values, grid_points, **kwargs):
        """Linear interpolation using triangulation"""
        # Would use scipy.spatial.Delaunay for triangulation
        # This is a placeholder implementation
        return self._idw_interpolation(points, values, grid_points, power=1.0, **kwargs)
    
    def _cubic_interpolation(self, points, values, grid_points, **kwargs):
        """Cubic spline interpolation"""
        # Would implement cubic spline interpolation
        # This is a placeholder implementation
        return self._idw_interpolation(points, values, grid_points, power=3.0, **kwargs)
    
    def _nearest_interpolation(self, points, values, grid_points, **kwargs):
        """Nearest neighbor interpolation"""
        
        x_points, y_points = points
        x_grid, y_grid = grid_points
        
        result = self.xp.zeros(x_grid.shape, dtype=values.dtype)
        
        x_grid_flat = x_grid.ravel()
        y_grid_flat = y_grid.ravel()
        result_flat = result.ravel()
        
        for i in range(len(x_grid_flat)):
            target_x = x_grid_flat[i]
            target_y = y_grid_flat[i]
            
            # Find nearest point
            dx = x_points - target_x
            dy = y_points - target_y
            distances = self.xp.sqrt(dx**2 + dy**2)
            
            nearest_idx = self.xp.argmin(distances)
            result_flat[i] = values[nearest_idx]
        
        return result.reshape(x_grid.shape)