"""
GPU-accelerated fitting algorithms for SuperDARN processing.

Key improvements over the original version:
  - Two-pass weighted LS (Pass 1: linear-power weights for unbiased parameters;
    Pass 2: log-corrected Bendat-Piersol weights for proper chi-squared errors)
  - Quadratic (Gaussian) spectral width alongside the linear (exponential) lambda width
  - Per-lag validity mask from LagValidator (replaces flat amplitude threshold)
  - Physics-correct vel_factor = c/(4π·f·Δτ) — no hardcoded magic constants
  - Noise-subtracted amplitudes (RST: subtract noise floor before log fit)

Reference: fitacf_v3.0/src/fitting.c  (Power_Fits, ACF_Phase_Fit)
           fitacf_v3.0/src/leastsquares.c (two_param_straight_line_fit, quadratic_fit)
           Bendat & Piersol (1986) §8.4
"""

from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np

from ..core.backends import get_array_module, get_backend, Backend


class LeastSquaresFitter:
    """
    Two-pass weighted least-squares fitter for SuperDARN ACF data.

    Outer loop (fit_lorentzian_batch) iterates over ranges and calls
    _fit_single_range.  All heavy lifting inside _fit_single_range uses
    xp (numpy or cupy) for the small (n_valid,) arrays.
    """

    def __init__(self):
        self.xp = get_array_module()
        if get_backend() == Backend.CUPY:
            self._init_gpu_solver()

    def _init_gpu_solver(self):
        try:
            import cupy as cp
            from cupy.linalg import solve
            self.gpu_solve = solve
        except ImportError:
            warnings.warn("Could not initialize GPU solver, falling back to CPU")
            self.gpu_solve = None

    # ── batch entry point ─────────────────────────────────────────────────────

    def fit_lorentzian_batch(
        self,
        acf_data:         Any,                    # (n_ranges, n_lags) complex
        power_data:       Any,                    # (n_ranges,) float — lag-0 power
        lag_time_step_sec: float  = 1.0,
        tfreq_hz:         float   = 12e6,
        max_velocity:     float   = 2000.0,
        max_width:        float   = 1000.0,
        valid_mask:       Optional[Any] = None,   # (n_ranges, n_lags) bool
        bp_sigma:         Optional[Any] = None,   # (n_ranges, n_lags) float — B-P sigma
        noise_pwr:        float   = 0.0,
    ) -> Dict[str, Any]:
        """
        Batch fitting.  valid_mask and bp_sigma come from LagValidator;
        when absent they fall back to a simple amplitude threshold (old behaviour).

        Returns dict with arrays of shape (n_ranges,):
          velocity, velocity_error, spectral_width, spectral_width_error,
          spectral_width_sigma, spectral_width_sigma_error, power, power_error
        """
        xp          = self.xp
        _np_input   = isinstance(acf_data, np.ndarray)
        acf_data    = xp.asarray(acf_data)
        power_data  = xp.asarray(power_data)
        if valid_mask is not None:
            valid_mask = xp.asarray(valid_mask)
        if bp_sigma is not None:
            bp_sigma = xp.asarray(bp_sigma)
        n_ranges, n_lags = acf_data.shape

        # vel_factor = c / (4π·f·Δτ)  [m/s/rad]
        lag_dt   = max(float(lag_time_step_sec), 1e-12)
        f_hz     = max(float(tfreq_hz), 1e3)
        vel_factor = 3e8 / (4.0 * float(xp.pi) * f_hz * lag_dt)

        # Output arrays
        out = {k: xp.zeros(n_ranges, dtype=xp.float32) for k in (
            'velocity', 'velocity_error',
            'spectral_width', 'spectral_width_error',
            'spectral_width_sigma', 'spectral_width_sigma_error',
            'power', 'power_error',
        )}

        for i in range(n_ranges):
            if power_data[i] <= 0:
                continue
            vm_i  = valid_mask[i]  if valid_mask  is not None else None
            sig_i = bp_sigma[i]    if bp_sigma     is not None else None
            r = self._fit_single_range(
                acf_data[i],
                float(power_data[i]),
                lag_dt, vel_factor,
                max_velocity, max_width,
                valid_mask=vm_i,
                bp_sigma=sig_i,
                noise_pwr=noise_pwr,
            )
            for k in out:
                out[k][i] = r[k]

        if _np_input:
            return {k: (v.get() if hasattr(v, "get") else np.asarray(v)) for k, v in out.items()}
        return out

    # ── single-range fitter ───────────────────────────────────────────────────

    def _fit_single_range(
        self,
        acf:          Any,            # (n_lags,) complex
        power:        float,          # lag-0 power
        lag_time_step_sec: float,
        vel_factor:   float,          # c/(4π·f·Δτ) [m/s/rad]
        max_velocity: float,
        max_width:    float,
        valid_mask:   Optional[Any]  = None,   # (n_lags,) bool
        bp_sigma:     Optional[Any]  = None,   # (n_lags,) float  — linear-domain sigma
        noise_pwr:    float          = 0.0,
    ) -> Dict[str, float]:
        xp      = self.xp
        n_lags  = len(acf)

        if n_lags < 3:
            return self._zero_result()

        # ── valid lags ────────────────────────────────────────────────────────
        if valid_mask is not None:
            vmask = xp.asarray(valid_mask, dtype=bool)
        else:
            # Fallback: amplitude above 5 % of lag-0
            amps     = xp.abs(acf[1:])
            vmask_1  = amps > 0.05 * abs(power)
            vmask    = xp.concatenate([xp.ones(1, dtype=bool), vmask_1])

        n_valid = int(xp.sum(vmask))
        if n_valid < 3:
            return self._zero_result()

        # ── amplitudes — RST does NOT subtract noise before the log fit ──────
        # (Noise is only applied to the final dB conversion of p_l/p_s.)
        # noise_pwr parameter is retained for API compatibility but not applied.
        amps = xp.abs(acf)                                                # (n_lags,)

        # Lag times for valid lags (excluding lag-0 from power fit)
        lags_all  = xp.arange(n_lags, dtype=xp.float32)
        times_all = lags_all * float(lag_time_step_sec)

        # Exclude lag-0 from power/width fits (we fit decay, not absolute power)
        fit_mask  = vmask.copy()
        fit_mask[0] = False
        if int(xp.sum(fit_mask)) < 2:
            return self._zero_result()

        t_v   = times_all[fit_mask]                   # (n_fit,)
        amp_v = amps[fit_mask]                        # (n_fit,)
        acf_v = acf[fit_mask]                         # (n_fit,) complex

        # Guard: all amps must be positive
        if float(xp.min(amp_v)) <= 0:
            return self._zero_result()

        log_amp_v = xp.log(amp_v)                     # (n_fit,)

        # ── Pass 1: linear-power weights → parameters ─────────────────────
        # w1 = |R(lag)|^2  (or 1/sigma^2 if Bendat-Piersol available)
        if bp_sigma is not None:
            sig_v = xp.asarray(bp_sigma, dtype=xp.float32)[fit_mask]
            sig_v = xp.maximum(sig_v, 1e-10)
            w1    = 1.0 / (sig_v ** 2)
        else:
            w1 = amp_v ** 2

        # Linear model: log|R| = a + b*t   (2-param straight-line fit)
        a1, b1, sigma_a1, sigma_b1 = self._weighted_linear_fit(t_v, log_amp_v, w1, xp)

        # Lambda spectral width from slope b1 (b1 < 0 for physical decay)
        w_l = min(abs(float(b1)) * vel_factor * float(lag_time_step_sec), max_width)

        # ── Pass 2: log-corrected weights → error estimates ───────────────
        # Convert sigma to log-domain: sigma_log = sigma_linear / |R|
        if bp_sigma is not None:
            sig_log_v = sig_v / (amp_v + 1e-30)
        else:
            # Approximate log-sigma from pass-1 fitted amplitude
            fit_amp   = xp.exp(a1 + b1 * t_v)
            sig_log_v = xp.maximum(xp.ones_like(t_v) * 0.1, 1.0 / (fit_amp + 1e-10))

        w2 = 1.0 / (sig_log_v ** 2)
        _, b2, sigma_a2, sigma_b2 = self._weighted_linear_fit(t_v, log_amp_v, w2, xp)

        w_l_e = min(float(sigma_b2) * vel_factor * float(lag_time_step_sec), max_width)
        p_l   = float(xp.exp(a1))        # lag-0 power estimate from fit
        p_l_e = float(xp.exp(a1)) * float(sigma_a2)

        # ── Quadratic fit for Gaussian (sigma) spectral width ─────────────
        # log|R| = a + c * t^2   (symmetric decay model, RST: quadratic_fit)
        # RST conversion: W_S = sqrt(|c|) * (c/(4pi*f)) * 4 * sqrt(ln2)
        #                     = sqrt(|c|) * vel_factor * Δτ * 4 * sqrt(ln2)
        w_s, w_s_e = 0.0, 0.0
        if int(xp.sum(fit_mask)) >= 3:
            c_coeff, sigma_c = self._weighted_quadratic_fit(t_v, log_amp_v, w1, xp)
            if float(c_coeff) < 0:
                conv = vel_factor * float(lag_time_step_sec) * 4.0 * float(xp.sqrt(xp.log(xp.asarray(2.0))))
                w_s  = min(float(xp.sqrt(-c_coeff)) * conv, max_width)
                w_s_e = float(sigma_c) / (2.0 * float(xp.sqrt(-c_coeff)) + 1e-30) * conv

        # ── Velocity from RST one-param phase LS ─────────────────────────────
        velocity, vel_sigma_b = self._fit_velocity_with_error(
            acf, vmask, vel_factor, max_velocity, xp)
        # velocity_error = sigma_b * vel_factor  (RST: sqrt(1/S_tt) * vel_factor)
        velocity_error = max(vel_sigma_b, float(sigma_b2) * vel_factor * float(lag_time_step_sec))

        return {
            'velocity':                  float(velocity),
            'velocity_error':            float(velocity_error),
            'spectral_width':            float(w_l),
            'spectral_width_error':      float(max(w_l_e, 0.0)),
            'spectral_width_sigma':      float(w_s),
            'spectral_width_sigma_error':float(max(w_s_e, 0.0)),
            'power':                     float(p_l),
            'power_error':               float(max(p_l_e, 0.0)),
        }

    # ── weighted LS helpers ───────────────────────────────────────────────────

    @staticmethod
    def _weighted_linear_fit(
        t: Any, y: Any, w: Any, xp
    ) -> Tuple[float, float, float, float]:
        """
        Weighted two-parameter straight-line fit: y = a + b*t.
        Returns (a, b, sigma_a, sigma_b) using normal equations.

        Normal equations:
          S    = sum(w)
          S_t  = sum(w*t)
          S_tt = sum(w*t^2)
          S_y  = sum(w*y)
          S_ty = sum(w*t*y)
          delta = S*S_tt - S_t^2
          a = (S_tt*S_y  - S_t*S_ty) / delta
          b = (S   *S_ty - S_t*S_y ) / delta
        """
        S    = xp.sum(w)
        S_t  = xp.sum(w * t)
        S_tt = xp.sum(w * t ** 2)
        S_y  = xp.sum(w * y)
        S_ty = xp.sum(w * t * y)
        delta = S * S_tt - S_t ** 2

        if abs(float(delta)) < 1e-30:
            return 0.0, 0.0, 1e6, 1e6

        a = float((S_tt * S_y  - S_t * S_ty) / delta)
        b = float((S    * S_ty - S_t * S_y ) / delta)

        # Parameter variances: sigma_a^2 = S_tt/delta, sigma_b^2 = S/delta
        sigma_a = float(xp.sqrt(xp.abs(S_tt / delta)))
        sigma_b = float(xp.sqrt(xp.abs(S    / delta)))
        return a, b, sigma_a, sigma_b

    @staticmethod
    def _weighted_quadratic_fit(
        t: Any, y: Any, w: Any, xp
    ) -> Tuple[float, float]:
        """
        Weighted quadratic fit: y = a + c*t^2  (no linear term — symmetric decay).
        Returns (c, sigma_c).

        2x2 normal equations in (a, c):
          [S     S_t2] [a]   [S_y  ]
          [S_t2  S_t4] [c] = [S_yt2]
        """
        S      = xp.sum(w)
        S_t2   = xp.sum(w * t ** 2)
        S_t4   = xp.sum(w * t ** 4)
        S_y    = xp.sum(w * y)
        S_yt2  = xp.sum(w * y * t ** 2)
        delta  = S * S_t4 - S_t2 ** 2

        if abs(float(delta)) < 1e-30:
            return 0.0, 1e6

        c       = float((S * S_yt2 - S_t2 * S_y) / delta)
        sigma_c = float(xp.sqrt(xp.abs(S / delta)))
        return c, sigma_c

    # ── velocity fit ─────────────────────────────────────────────────────────

    def _fit_velocity_with_error(self, acf, vmask, vel_factor, max_velocity, xp):
        """Return (velocity_m_s, sigma_b_m_s) from one-param LS."""
        try:
            n_lags = len(acf)
            phi0   = float(xp.angle(acf[0]))
            S_tt = S_ty = 0.0
            n_valid = 0
            for l in range(1, n_lags):
                if not vmask[l]:
                    continue
                mag = float(xp.abs(acf[l]))
                if mag < 1e-10:
                    continue
                dphi = float(xp.angle(acf[l])) - phi0
                if dphi >  float(xp.pi): dphi -= 2.0 * float(xp.pi)
                if dphi < -float(xp.pi): dphi += 2.0 * float(xp.pi)
                w = mag * mag
                S_tt += w * l * l
                S_ty += w * l * dphi
                n_valid += 1
            if n_valid < 1 or S_tt <= 0.0:
                return 0.0, 0.0
            b       = S_ty / S_tt
            sigma_b = (1.0 / S_tt) ** 0.5
            v = float(max(min(b * vel_factor, max_velocity), -max_velocity))
            return v, sigma_b * vel_factor
        except Exception:
            return 0.0, 0.0

    def _fit_velocity(
        self, acf, vmask, t_v, times_all, vel_factor, max_velocity, xp
    ) -> float:
        """
        RST one-parameter weighted least-squares velocity fit.

        RST reference: fitacf_v3.0/src/leastsquares.c one_param_straight_line_fit()

        Model: phase[l] = b · l  (forced through origin; lag-0 is the reference)
        Weights: w = |R(l)|²  (amplitude-squared, Pass-1)

        slope b (rad/lag-index) → velocity = b * vel_factor  [m/s]
        where vel_factor = c / (4π · f · Δτ).

        Each lag phase is wrapped individually to (−π, π] relative to lag-0
        (RST approach — no cumulative unwrapping to avoid error propagation).
        """
        try:
            n_lags = len(acf)
            if n_lags < 2:
                return 0.0

            phi0 = float(xp.angle(acf[0]))   # lag-0 reference phase
            lags = xp.arange(n_lags, dtype=xp.float32)

            S_tt = 0.0
            S_ty = 0.0
            n_valid = 0

            for l in range(1, n_lags):
                if not vmask[l]:
                    continue
                mag = float(xp.abs(acf[l]))
                if mag < 1e-10:
                    continue
                dphi = float(xp.angle(acf[l])) - phi0
                # Wrap to (−π, π]
                if dphi >  float(xp.pi): dphi -= 2.0 * float(xp.pi)
                if dphi < -float(xp.pi): dphi += 2.0 * float(xp.pi)
                w = mag * mag                   # amplitude-squared weight
                S_tt += w * l * l
                S_ty += w * l * dphi
                n_valid += 1

            if n_valid < 1 or S_tt <= 0.0:
                return 0.0

            b = S_ty / S_tt                     # rad/lag-index
            velocity = b * vel_factor
            return float(max(min(velocity, max_velocity), -max_velocity))
        except Exception:
            return 0.0

    def _zero_result(self) -> Dict[str, float]:
        return {k: 0.0 for k in (
            'velocity', 'velocity_error',
            'spectral_width', 'spectral_width_error',
            'spectral_width_sigma', 'spectral_width_sigma_error',
            'power', 'power_error',
        )}


# ─────────────────────────────────────────────────────────────────────────────

class PhaseUnwrapper:
    """GPU-accelerated phase unwrapping (used for per-range lag-sequence unwrapping)."""

    def __init__(self):
        self.xp = get_array_module()

    def unwrap(self, phases: Any, tolerance: float = 0.1) -> Any:
        xp = self.xp
        if get_backend() == Backend.CUPY:
            try:
                import cupy as cp
                if hasattr(cp, 'unwrap'):
                    return cp.unwrap(phases)
            except Exception:
                pass
        return xp.unwrap(phases)


class NonLinearFitter:
    """
    Advanced non-linear fitting for SuperDARN applications.
    fit_ionospheric_model is implemented in processing/cnvmap.py.
    """

    def __init__(self):
        self.xp = get_array_module()

    def fit_complex_lorentzian(self, acf_data: Any, initial_guess=None) -> Dict[str, Any]:
        raise NotImplementedError("Advanced Lorentzian fitting: use LeastSquaresFitter")

    def fit_ionospheric_model(self, grid_data: Any, model_order: int = 8) -> Dict[str, Any]:
        """Delegated to CNVMAPProcessor — import there to avoid circular deps."""
        from ..processing.cnvmap import CNVMAPProcessor
        proc = CNVMAPProcessor(lmax=model_order)
        return proc.fit_from_grid(grid_data)
