"""
GPU-accelerated fitting algorithms for SuperDARN processing
"""

from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np

from ..core.backends import get_array_module, get_backend, Backend

class LeastSquaresFitter:
    """
    GPU-accelerated least squares fitting for SuperDARN ACF data
    """
    
    def __init__(self):
        self.xp = get_array_module()
        
        # Initialize GPU-specific resources
        if get_backend() == Backend.CUPY:
            self._init_gpu_solver()
    
    def _init_gpu_solver(self):
        """Initialize GPU linear algebra solver"""
        try:
            import cupy as cp
            from cupy.linalg import solve
            from cupyx.scipy.linalg import lstsq
            
            self.gpu_solve = solve
            self.gpu_lstsq = lstsq
            
        except ImportError:
            warnings.warn("Could not initialize GPU solver, falling back to CPU")
            self.gpu_solve = None
            self.gpu_lstsq = None
    
    def fit_lorentzian_batch(self, 
                           acf_data: Any,
                           power_data: Any,
                           lag_time_step_sec: float = 1.0,
                           max_velocity: float = 2000.0,
                           max_width: float = 1000.0) -> Dict[str, Any]:
        """
        Batch fitting of Lorentzian curves to ACF data
        
        Parameters
        ----------
        acf_data : array
            ACF data with shape (n_ranges, n_lags)
        power_data : array
            Power data with shape (n_ranges,)
        max_velocity : float
            Maximum allowed velocity (m/s)
        max_width : float
            Maximum allowed spectral width (m/s)
            
        Returns
        -------
        Dict[str, array]
            Fitted parameters for all ranges
        """
        xp = self.xp
        acf_data = xp.asarray(acf_data)
        power_data = xp.asarray(power_data)
        n_ranges, n_lags = acf_data.shape
        lag_time_step_sec = max(float(lag_time_step_sec), 1e-9)

        # Initialize output arrays
        velocity = xp.zeros(n_ranges, dtype=xp.float32)
        velocity_error = xp.zeros(n_ranges, dtype=xp.float32)
        spectral_width = xp.zeros(n_ranges, dtype=xp.float32)
        spectral_width_error = xp.zeros(n_ranges, dtype=xp.float32)
        power_fitted = xp.real(acf_data[:, 0]).astype(xp.float32)
        power_error = xp.abs(power_fitted).astype(xp.float32) * xp.float32(0.05)

        # Keep CPU path on the validated scalar fitter to preserve baseline behavior.
        if get_backend() != Backend.CUPY:
            for i in range(n_ranges):
                if power_data[i] > 0:
                    result = self._fit_single_range(
                        acf_data[i, :], power_data[i], lag_time_step_sec, max_velocity, max_width
                    )
                    velocity[i] = result['velocity']
                    velocity_error[i] = result['velocity_error']
                    spectral_width[i] = result['spectral_width']
                    spectral_width_error[i] = result['spectral_width_error']
                    power_fitted[i] = result['power']
                    power_error[i] = result['power_error']

            return {
                'velocity': velocity,
                'velocity_error': velocity_error,
                'spectral_width': spectral_width,
                'spectral_width_error': spectral_width_error,
                'power': power_fitted,
                'power_error': power_error
            }

        if n_lags < 3:
            return {
                'velocity': velocity,
                'velocity_error': velocity_error,
                'spectral_width': spectral_width,
                'spectral_width_error': spectral_width_error,
                'power': power_fitted,
                'power_error': power_error
            }

        valid_range = power_data > 0

        # ------------------------------------------------------------------
        # Width from masked linear fit of log(|acf|) vs lag index.
        # ------------------------------------------------------------------
        amp0 = xp.abs(acf_data[:, 0]) + xp.float32(1e-10)
        amps = xp.abs(acf_data[:, 1:])
        y = xp.log(xp.clip(amps, xp.float32(1e-10), None))
        x = xp.arange(1, n_lags, dtype=xp.float32)

        w = (amps > (xp.float32(0.01) * amp0[:, None])) & valid_range[:, None]
        wf = w.astype(xp.float32)

        n = xp.sum(wf, axis=1)
        sx = xp.sum(wf * x[None, :], axis=1)
        sy = xp.sum(wf * y, axis=1)
        sxx = xp.sum(wf * (x[None, :] * x[None, :]), axis=1)
        sxy = xp.sum(wf * (x[None, :] * y), axis=1)

        den = n * sxx - sx * sx
        safe = (n >= 2) & (xp.abs(den) > xp.float32(1e-8))

        b1 = xp.zeros(n_ranges, dtype=xp.float32)
        b1 = xp.where(safe, (n * sxy - sx * sy) / (den + xp.float32(1e-12)), b1)
        decay_constant = -b1
        width_est = xp.abs(decay_constant) * xp.float32(100.0 / lag_time_step_sec)
        width_est = xp.clip(width_est, xp.float32(0.0), xp.float32(max_width))
        # Preserve prior fallback semantics for underconstrained rows.
        width_est = xp.where(valid_range & ~safe, xp.float32(50.0), width_est)
        spectral_width = width_est.astype(xp.float32)

        # ------------------------------------------------------------------
        # Velocity from phase increments between adjacent lags.
        # ------------------------------------------------------------------
        c_prev = acf_data[:, :-1]
        c_next = acf_data[:, 1:]
        inc = xp.angle(c_next * xp.conj(c_prev))

        winc = xp.sqrt(xp.abs(c_prev) * xp.abs(c_next))
        m_prev = xp.abs(c_prev) > (xp.float32(0.01) * amp0[:, None])
        m_next = xp.abs(c_next) > (xp.float32(0.01) * amp0[:, None])
        m = (m_prev & m_next & valid_range[:, None]).astype(xp.float32)

        winc = winc * m
        wsum = xp.sum(winc, axis=1)
        has_phase = wsum > xp.float32(1e-10)

        mean_inc = xp.where(
            has_phase,
            xp.sum(winc * inc, axis=1) / (wsum + xp.float32(1e-12)),
            xp.float32(0.0),
        )

        vel_est = mean_inc * xp.float32(200.0 / lag_time_step_sec)
        vel_est = xp.clip(vel_est, xp.float32(-max_velocity), xp.float32(max_velocity))
        velocity = xp.where(valid_range & has_phase, vel_est, xp.float32(0.0)).astype(xp.float32)

        # ------------------------------------------------------------------
        # Error estimates from weighted spread.
        # ------------------------------------------------------------------
        vel_step = inc * xp.float32(200.0 / lag_time_step_sec)
        vel_var = xp.where(
            has_phase,
            xp.sum(winc * (vel_step - vel_est[:, None]) ** 2, axis=1) / (wsum + xp.float32(1e-12)),
            xp.float32(0.0),
        )
        vel_err = xp.sqrt(xp.maximum(vel_var, xp.float32(0.0)))
        velocity_error = xp.maximum(vel_err, xp.float32(1e-6)).astype(xp.float32)

        # Width error: bounded to remain positive and below width for finite widths.
        width_err = spectral_width * xp.float32(0.1)
        tiny_width = spectral_width <= xp.float32(1e-6)
        width_err = xp.where(tiny_width, xp.float32(1e-7), width_err)
        width_err = xp.where(~tiny_width, xp.minimum(xp.maximum(width_err, xp.float32(1e-7)), spectral_width * xp.float32(0.99)), width_err)
        spectral_width_error = width_err.astype(xp.float32)

        power_error = xp.maximum(power_error, xp.float32(1e-7)).astype(xp.float32)
        
        return {
            'velocity': velocity,
            'velocity_error': velocity_error,
            'spectral_width': spectral_width,
            'spectral_width_error': spectral_width_error,
            'power': power_fitted,
            'power_error': power_error
        }
    
    def _fit_single_range(self, 
                         acf: Any,
                         power: float,
                         lag_time_step_sec: float,
                         max_velocity: float,
                         max_width: float) -> Dict[str, float]:
        """
        Fit Lorentzian curve to single range ACF
        
        This is a simplified implementation. The full FITACF algorithm
        includes more sophisticated fitting procedures.
        """
        n_lags = len(acf)
        
        if n_lags < 3:
            return self._zero_result()
        
        # Calculate phases and amplitudes
        phases = self.xp.angle(acf[1:])  # Skip lag-0
        amplitudes = self.xp.abs(acf[1:])
        
        # Simple linear fit to log(amplitude) vs lag
        lags = self.xp.arange(1, n_lags, dtype=self.xp.float32)
        
        # Filter out bad lags (very low amplitude)
        valid_mask = amplitudes > 0.01 * self.xp.abs(acf[0])
        
        if self.xp.sum(valid_mask) < 2:
            return self._zero_result()
        
        valid_lags = lags[valid_mask]
        valid_phases = phases[valid_mask]
        valid_amps = amplitudes[valid_mask]
        
        # Fit decay constant (spectral width) using short-lag amplitude ratios.
        try:
            lag_time_step_sec = max(float(lag_time_step_sec), 1e-9)

            # Backend-deterministic fit path.
            amps = np.asarray(self.xp.abs(acf[1:]), dtype=np.float64)
            lags_np = np.arange(1, n_lags, dtype=np.float64)
            amp0 = float(np.abs(complex(acf[0])) + 1e-10)

            valid = amps > 0.01 * amp0
            if np.sum(valid) < 2:
                raise ValueError("Insufficient valid amplitudes")

            y = np.log(np.clip(amps[valid], 1e-10, None))
            x = lags_np[valid]
            A = np.vstack([np.ones_like(x), x]).T
            b0, b1 = np.linalg.lstsq(A, y, rcond=None)[0]

            decay_constant = -b1
            width = min(abs(decay_constant) * (100.0 / lag_time_step_sec), max_width)
            
        except Exception:
            width = 50.0  # Default width
        
        # Fit velocity from unwrapped phase progression.
        try:
            lag_time_step_sec = max(float(lag_time_step_sec), 1e-9)
            lag_times = np.arange(1, n_lags, dtype=np.float64) * lag_time_step_sec
            phase_vals = np.angle(np.asarray(acf[1:], dtype=np.complex128))
            valid_phase_mask = np.asarray(amplitudes > 0.01 * self.xp.abs(acf[0]))
            if np.sum(valid_phase_mask) < 2:
                raise ValueError("Insufficient valid phase samples")

            t = lag_times[valid_phase_mask]
            p = np.unwrap(phase_vals[valid_phase_mask])
            w = np.asarray(valid_amps, dtype=np.float64)

            # Weighted linear regression p(t) = a + b*t
            wsum = np.sum(w) + 1e-12
            t_mean = np.sum(w * t) / wsum
            p_mean = np.sum(w * p) / wsum
            cov_tp = np.sum(w * (t - t_mean) * (p - p_mean))
            var_t = np.sum(w * (t - t_mean) * (t - t_mean)) + 1e-12
            phase_slope_time = cov_tp / var_t

            # phase = velocity * t / 200 => velocity = slope * 200
            velocity = float(np.clip(phase_slope_time * 200.0, -max_velocity, max_velocity))
            
        except Exception:
            velocity = 0.0
        
        # Simple error estimates (would be more sophisticated in full implementation)
        velocity_error = max(abs(velocity) * 0.1, 1e-6)  # keep strictly positive
        if width <= 1e-6:
            width_error = 1e-7
        else:
            width_error = min(max(width * 0.1, 1e-7), width * 0.99)
        power_error = power * 0.05
        
        return {
            'velocity': float(velocity),
            'velocity_error': float(velocity_error),
            'spectral_width': float(width),
            'spectral_width_error': float(width_error),
            'power': float(power),
            'power_error': float(power_error)
        }
    
    def _zero_result(self) -> Dict[str, float]:
        """Return zero result for failed fits"""
        return {
            'velocity': 0.0,
            'velocity_error': 0.0,
            'spectral_width': 0.0,
            'spectral_width_error': 0.0,
            'power': 0.0,
            'power_error': 0.0
        }

class PhaseUnwrapper:
    """
    GPU-accelerated phase unwrapping for SuperDARN data
    """
    
    def __init__(self):
        self.xp = get_array_module()
    
    def unwrap(self, phases: Any, 
               tolerance: float = 0.1) -> Any:
        """
        Unwrap phase values to remove 2π discontinuities
        
        Parameters
        ----------
        phases : array
            Phase values in radians
        tolerance : float
            Tolerance for unwrapping decisions
            
        Returns
        -------
        array
            Unwrapped phases
        """
        if get_backend() == Backend.CUPY:
            return self._unwrap_gpu(phases, tolerance)
        else:
            return self._unwrap_cpu(phases, tolerance)
    
    def _unwrap_gpu(self, phases: Any, tolerance: float) -> Any:
        """GPU-accelerated phase unwrapping using CuPy"""
        try:
            import cupy as cp
            
            # Use CuPy's unwrap function if available
            if hasattr(cp, 'unwrap'):
                return cp.unwrap(phases)
            else:
                # Fallback to CPU implementation
                return self._unwrap_cpu(phases, tolerance)
                
        except Exception:
            return self._unwrap_cpu(phases, tolerance)
    
    def _unwrap_cpu(self, phases: Any, tolerance: float) -> Any:
        """CPU implementation of phase unwrapping"""
        
        if len(phases) < 2:
            return phases
        
        unwrapped = self.xp.copy(phases)
        
        # Simple unwrapping algorithm
        for i in range(1, len(phases)):
            diff = phases[i] - phases[i-1]
            
            # Check for phase jumps
            if diff > self.xp.pi:
                unwrapped[i:] -= 2 * self.xp.pi
            elif diff < -self.xp.pi:
                unwrapped[i:] += 2 * self.xp.pi
        
        return unwrapped

class NonLinearFitter:
    """
    Advanced non-linear fitting for SuperDARN applications
    """
    
    def __init__(self):
        self.xp = get_array_module()
        
        # Initialize optimization libraries
        if get_backend() == Backend.CUPY:
            try:
                import cupy as cp
                # CuPy optimization routines if available
                self.optimizer = 'cupy'
            except ImportError:
                self.optimizer = 'scipy'
        else:
            self.optimizer = 'scipy'
    
    def fit_complex_lorentzian(self, 
                              acf_data: Any,
                              initial_guess: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Advanced Lorentzian fitting with full complex ACF model
        
        This would implement the complete FITACF v3.0 algorithm
        """
        # Placeholder for advanced fitting implementation
        raise NotImplementedError("Advanced Lorentzian fitting not yet implemented")
    
    def fit_ionospheric_model(self, 
                             grid_data: Any,
                             model_order: int = 8) -> Dict[str, Any]:
        """
        Fit spherical harmonic model to ionospheric convection data
        """
        # Placeholder for spherical harmonic fitting
        raise NotImplementedError("Spherical harmonic fitting not yet implemented")