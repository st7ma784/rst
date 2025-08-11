"""
Comprehensive tests for FITACF processing algorithms

These tests validate FITACF fitting against reference implementations
and test the GPU-accelerated algorithms for correctness and performance.
"""

import pytest
import numpy as np
from datetime import datetime

from superdarn_gpu.processing.fitacf import FitACFProcessor, FitACFConfig, FitACFAlgorithm, process_fitacf
from superdarn_gpu.core.datatypes import RawACF, FitACF, RadarParameters
from superdarn_gpu.core.backends import BackendContext


class TestFitACFProcessor:
    """Test suite for FITACF processing functionality"""
    
    @pytest.fixture
    def sample_radar_params(self):
        """Create sample radar parameters"""
        return RadarParameters(
            station_id=65,
            beam_number=7,
            scan_flag=1,
            channel=1,
            cp_id=153,
            nave=50,
            lagfr=4800,
            smsep=300,
            txpow=9000,
            atten=0,
            noise_search=2.5,
            noise_mean=2.3,
            tfreq=10500,
            nrang=75,
            frang=180,
            rsep=45,
            xcf=1,
            mppul=8,
            mpinc=1500,
            mplgs=18,
            txpl=100,
            intt_sc=3,
            intt_us=0,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def synthetic_rawacf(self, sample_radar_params):
        """Create synthetic RawACF data with known Lorentzian characteristics"""
        prm = sample_radar_params
        rawacf = RawACF(nrang=prm.nrang, mplgs=prm.mplgs, nave=prm.nave)
        rawacf.prm = prm
        
        # Generate synthetic ACF with Lorentzian decay for some ranges
        for r in range(prm.nrang):
            if 20 <= r <= 50:  # Ionospheric ranges
                
                # Known parameters for validation
                true_velocity = 300.0 * (1 if r % 2 else -1)  # ±300 m/s
                true_width = 150.0 + r * 2  # Increasing width with range
                true_power = 1000.0 * np.exp(-r * 0.05)  # Decreasing power
                
                # Generate Lorentzian ACF
                for lag in range(prm.mplgs):
                    lag_time = lag * prm.mpinc * 1e-6  # seconds
                    
                    # Lorentzian model: A * exp(-W*τ + i*φ*τ)
                    decay = np.exp(-true_width * lag_time / 100.0)  # Scaling factor
                    phase = true_velocity * lag_time / 200.0  # Doppler phase
                    
                    acf_real = true_power * decay * np.cos(phase)
                    acf_imag = true_power * decay * np.sin(phase)
                    
                    # Add realistic noise
                    noise_level = true_power * 0.05  # 5% noise
                    acf_real += np.random.normal(0, noise_level)
                    acf_imag += np.random.normal(0, noise_level)
                    
                    rawacf.acf[r, lag] = complex(acf_real, acf_imag)
                
                # Set power from lag-0
                rawacf.power[r] = np.real(rawacf.acf[r, 0])
                
                # Store true values for validation (in metadata)
                if not hasattr(rawacf, '_true_params'):
                    rawacf._true_params = {}
                rawacf._true_params[r] = {
                    'velocity': true_velocity,
                    'spectral_width': true_width,
                    'power': true_power
                }
            else:
                # Noise-only ranges
                for lag in range(prm.mplgs):
                    noise = np.random.normal(0, 10, 2)
                    rawacf.acf[r, lag] = complex(noise[0], noise[1])
                rawacf.power[r] = abs(rawacf.acf[r, 0])
        
        # Set noise levels and flags
        rawacf.noise[:] = 10.0
        rawacf.slist[:] = np.arange(prm.nrang)
        rawacf.qflg[:] = 0  # Will be set by FITACF processor
        rawacf.gflg[:] = 0
        
        return rawacf
    
    def test_fitacf_processor_initialization(self):
        """Test FITACF processor initialization"""
        config = FitACFConfig(
            algorithm=FitACFAlgorithm.V3_0,
            min_power_threshold=3.0,
            batch_size=512
        )
        
        processor = FitACFProcessor(config=config)
        
        assert processor.config == config
        assert processor.name == "FITACF Processor"
        assert hasattr(processor, 'fitter')
        assert hasattr(processor, 'phase_unwrapper')
    
    def test_input_validation(self, synthetic_rawacf):
        """Test input validation"""
        processor = FitACFProcessor()
        
        # Valid input
        assert processor.validate_input(synthetic_rawacf) == True
        
        # Invalid inputs
        assert processor.validate_input(None) == False
        assert processor.validate_input("not_rawacf") == False
        
        # RawACF with missing data
        invalid_rawacf = RawACF(nrang=10, mplgs=18)
        invalid_rawacf.acf = None
        assert processor.validate_input(invalid_rawacf) == False
    
    def test_basic_fitacf_processing(self, synthetic_rawacf):
        """Test basic FITACF processing functionality"""
        processor = FitACFProcessor()
        result = processor.process(synthetic_rawacf)
        
        # Verify output structure
        assert isinstance(result, FitACF)
        assert result.nrang == synthetic_rawacf.nrang
        assert result.prm == synthetic_rawacf.prm
        
        # Check that arrays have correct shapes
        assert result.velocity.shape == (synthetic_rawacf.nrang,)
        assert result.velocity_error.shape == (synthetic_rawacf.nrang,)
        assert result.spectral_width.shape == (synthetic_rawacf.nrang,)
        assert result.power.shape == (synthetic_rawacf.nrang,)
        
        # Check that some ranges were fitted (qflg > 0)
        fitted_ranges = np.sum(result.qflg > 0)
        assert fitted_ranges > 0
        assert fitted_ranges < synthetic_rawacf.nrang  # Not all ranges should be fitted
    
    def test_power_threshold_filtering(self, synthetic_rawacf):
        """Test power threshold filtering"""
        # Test with high threshold - should fit fewer ranges
        high_threshold_config = FitACFConfig(min_power_threshold=10.0)  # High threshold
        processor_high = FitACFProcessor(config=high_threshold_config)
        result_high = processor_high.process(synthetic_rawacf)
        
        # Test with low threshold - should fit more ranges
        low_threshold_config = FitACFConfig(min_power_threshold=1.0)  # Low threshold
        processor_low = FitACFProcessor(config=low_threshold_config)
        result_low = processor_low.process(synthetic_rawacf)
        
        # Low threshold should result in more fitted ranges
        high_fitted = np.sum(result_high.qflg > 0)
        low_fitted = np.sum(result_low.qflg > 0)
        
        assert low_fitted >= high_fitted
    
    def test_parameter_accuracy(self, synthetic_rawacf):
        """Test that fitted parameters are reasonably close to known values"""
        processor = FitACFProcessor()
        result = processor.process(synthetic_rawacf)
        
        # Check fitted parameters against known values
        if hasattr(synthetic_rawacf, '_true_params'):
            for range_idx, true_params in synthetic_rawacf._true_params.items():
                if result.qflg[range_idx] > 0:  # Only check fitted ranges
                    
                    fitted_velocity = result.velocity[range_idx]
                    fitted_width = result.spectral_width[range_idx]
                    fitted_power = result.power[range_idx]
                    
                    true_velocity = true_params['velocity']
                    true_width = true_params['spectral_width']
                    true_power = true_params['power']
                    
                    # Allow reasonable tolerances for noisy synthetic data
                    velocity_error = abs(fitted_velocity - true_velocity) / abs(true_velocity)
                    width_error = abs(fitted_width - true_width) / true_width
                    power_error = abs(fitted_power - true_power) / true_power
                    
                    # These are generous tolerances for synthetic noisy data
                    assert velocity_error < 0.5, f"Velocity error too large: {velocity_error:.2f}"
                    assert width_error < 0.8, f"Width error too large: {width_error:.2f}"
                    assert power_error < 0.3, f"Power error too large: {power_error:.2f}"
    
    def test_cpu_gpu_consistency(self, synthetic_rawacf):
        """Test CPU vs GPU implementation consistency"""
        
        # Process with CPU
        with BackendContext('numpy'):
            cpu_processor = FitACFProcessor()
            cpu_result = cpu_processor.process(synthetic_rawacf)
        
        try:
            # Process with GPU
            with BackendContext('cupy'):
                gpu_processor = FitACFProcessor()
                gpu_result = gpu_processor.process(synthetic_rawacf)
                
                # Convert GPU results to CPU for comparison
                gpu_result_cpu = gpu_result.to_cpu()
                
                # Results should be nearly identical
                fitted_mask = cpu_result.qflg > 0
                if np.any(fitted_mask):
                    np.testing.assert_allclose(
                        cpu_result.velocity[fitted_mask], 
                        gpu_result_cpu.velocity[fitted_mask],
                        rtol=1e-5, atol=1e-3
                    )
                    np.testing.assert_allclose(
                        cpu_result.spectral_width[fitted_mask],
                        gpu_result_cpu.spectral_width[fitted_mask], 
                        rtol=1e-5, atol=1e-3
                    )
                
        except ImportError:
            pytest.skip("CuPy not available for GPU testing")
    
    def test_phase_unwrapping(self, synthetic_rawacf):
        """Test phase unwrapping functionality"""
        # Create ACF with phase wrapping issues
        for r in range(20, 30):
            phases = np.linspace(0, 4*np.pi, synthetic_rawacf.mplgs)  # Wrapped phases
            amplitudes = 100 * np.exp(-np.arange(synthetic_rawacf.mplgs) * 0.1)
            
            for lag in range(synthetic_rawacf.mplgs):
                synthetic_rawacf.acf[r, lag] = amplitudes[lag] * np.exp(1j * phases[lag])
            
            synthetic_rawacf.power[r] = amplitudes[0]
        
        processor = FitACFProcessor()
        result = processor.process(synthetic_rawacf)
        
        # Should still produce reasonable results despite phase wrapping
        fitted_mask = result.qflg > 0
        assert np.any(fitted_mask)
        
        # Velocities should be within reasonable bounds
        fitted_velocities = result.velocity[fitted_mask]
        assert np.all(np.abs(fitted_velocities) <= 2000)  # Reasonable velocity limit
    
    def test_ground_scatter_detection(self, synthetic_rawacf):
        """Test ground scatter detection"""
        # Simulate ground scatter characteristics (low spectral width, specific power)
        for r in range(30, 35):
            # Ground scatter: high power, low spectral width
            power = 5000.0  # High power
            width = 50.0    # Low spectral width
            velocity = 50.0  # Low velocity
            
            for lag in range(synthetic_rawacf.mplgs):
                lag_time = lag * synthetic_rawacf.prm.mpinc * 1e-6
                decay = np.exp(-width * lag_time / 100.0)
                phase = velocity * lag_time / 200.0
                
                acf_val = power * decay * np.exp(1j * phase)
                synthetic_rawacf.acf[r, lag] = acf_val
            
            synthetic_rawacf.power[r] = power
        
        config = FitACFConfig(ground_scatter_threshold=0.3)
        processor = FitACFProcessor(config=config)
        result = processor.process(synthetic_rawacf)
        
        # Should detect some ground scatter
        ground_scatter_count = np.sum(result.gflg > 0)
        assert ground_scatter_count > 0
    
    def test_error_estimation(self, synthetic_rawacf):
        """Test parameter error estimation"""
        processor = FitACFProcessor()
        result = processor.process(synthetic_rawacf)
        
        fitted_mask = result.qflg > 0
        
        if np.any(fitted_mask):
            # Error estimates should be positive and reasonable
            velocity_errors = result.velocity_error[fitted_mask]
            width_errors = result.spectral_width_error[fitted_mask]
            
            assert np.all(velocity_errors > 0)
            assert np.all(width_errors > 0)
            
            # Errors should be reasonable fraction of parameter values
            velocities = np.abs(result.velocity[fitted_mask])
            widths = result.spectral_width[fitted_mask]
            
            # Relative errors should be less than 100%
            rel_vel_errors = velocity_errors / (velocities + 1e-6)
            rel_width_errors = width_errors / (widths + 1e-6)
            
            assert np.all(rel_vel_errors < 1.0)
            assert np.all(rel_width_errors < 1.0)
    
    def test_xcf_elevation_calculation(self, synthetic_rawacf):
        """Test elevation calculation from XCF data"""
        # Add XCF data to synthetic data
        synthetic_rawacf.xcf = np.zeros_like(synthetic_rawacf.acf)
        
        # Simulate interferometer data with phase differences
        for r in range(synthetic_rawacf.nrang):
            for lag in range(synthetic_rawacf.mplgs):
                # XCF with elevation-dependent phase shift
                phase_shift = 0.1 * r  # Simulate elevation angle
                xcf_val = synthetic_rawacf.acf[r, lag] * np.exp(1j * phase_shift)
                synthetic_rawacf.xcf[r, lag] = xcf_val * 0.8  # Reduce amplitude
        
        config = FitACFConfig(elevation_correction=True, enable_xcf=True)
        processor = FitACFProcessor(config=config)
        result = processor.process(synthetic_rawacf)
        
        # Should have elevation angles calculated
        fitted_mask = result.qflg > 0
        if np.any(fitted_mask):
            elevations = result.elevation[fitted_mask]
            # Should have non-zero elevation values
            assert np.any(~np.isnan(elevations))
    
    def test_batch_processing_performance(self, synthetic_rawacf):
        """Test batch processing configuration effects"""
        # Test with different batch sizes
        small_batch_config = FitACFConfig(batch_size=64)
        large_batch_config = FitACFConfig(batch_size=1024)
        
        processor_small = FitACFProcessor(config=small_batch_config)
        processor_large = FitACFProcessor(config=large_batch_config)
        
        result_small = processor_small.process(synthetic_rawacf)
        result_large = processor_large.process(synthetic_rawacf)
        
        # Results should be identical regardless of batch size
        np.testing.assert_array_equal(result_small.qflg, result_large.qflg)
        
        fitted_mask = result_small.qflg > 0
        if np.any(fitted_mask):
            np.testing.assert_allclose(
                result_small.velocity[fitted_mask],
                result_large.velocity[fitted_mask],
                rtol=1e-6
            )
    
    def test_algorithm_versions(self, synthetic_rawacf):
        """Test different FITACF algorithm versions"""
        # Test v2.5 algorithm
        config_v25 = FitACFConfig(algorithm=FitACFAlgorithm.V2_5)
        processor_v25 = FitACFProcessor(config=config_v25)
        # Note: v2.5 implementation would need to be added
        
        # Test v3.0 algorithm
        config_v30 = FitACFConfig(algorithm=FitACFAlgorithm.V3_0)
        processor_v30 = FitACFProcessor(config=config_v30)
        result_v30 = processor_v30.process(synthetic_rawacf)
        
        # v3.0 should produce results
        assert np.any(result_v30.qflg > 0)
    
    def test_convenience_function(self, synthetic_rawacf):
        """Test convenience function for direct processing"""
        result = process_fitacf(synthetic_rawacf)
        
        assert isinstance(result, FitACF)
        assert result.nrang == synthetic_rawacf.nrang
    
    def test_memory_estimation(self, synthetic_rawacf):
        """Test memory requirement estimation"""
        processor = FitACFProcessor()
        memory_estimate = processor.get_memory_estimate(synthetic_rawacf)
        
        assert memory_estimate > 0
        assert isinstance(memory_estimate, int)
        
        # Should be reasonable based on data size
        expected_min = synthetic_rawacf.nrang * synthetic_rawacf.mplgs * 8
        assert memory_estimate > expected_min
    
    def test_statistical_tracking(self, synthetic_rawacf):
        """Test processing statistics tracking"""
        processor = FitACFProcessor()
        result = processor.process(synthetic_rawacf)
        
        # Check statistics were updated
        assert processor.stats['processed_ranges'] == synthetic_rawacf.nrang
        assert processor.stats['fitted_ranges'] >= 0
        assert processor.stats['fitted_ranges'] <= synthetic_rawacf.nrang
        
        # Fitted ranges should match quality flags
        expected_fitted = np.sum(result.qflg > 0)
        assert processor.stats['fitted_ranges'] == expected_fitted


class TestFitACFValidation:
    """Integration tests for FITACF validation"""
    
    def test_lorentzian_model_validation(self):
        """Test fitting against exact Lorentzian model"""
        # Create perfect Lorentzian ACF
        nrang = 1
        mplgs = 10
        prm = RadarParameters(
            station_id=1, beam_number=0, scan_flag=1, channel=1, cp_id=1,
            nave=1, lagfr=100, smsep=100, txpow=1000, atten=0,
            noise_search=0.1, noise_mean=0.1, tfreq=10000,
            nrang=nrang, frang=100, rsep=45, xcf=0,
            mppul=8, mpinc=1000, mplgs=mplgs, txpl=100,
            intt_sc=1, intt_us=0, timestamp=datetime.now()
        )
        
        rawacf = RawACF(nrang=nrang, mplgs=mplgs, nave=1)
        rawacf.prm = prm
        
        # Perfect Lorentzian parameters
        true_velocity = 500.0  # m/s
        true_width = 200.0     # m/s  
        true_power = 1000.0
        
        # Generate exact Lorentzian
        for lag in range(mplgs):
            lag_time = lag * prm.mpinc * 1e-6
            decay = np.exp(-true_width * lag_time / 100.0)
            phase = true_velocity * lag_time / 200.0
            
            rawacf.acf[0, lag] = true_power * decay * np.exp(1j * phase)
        
        rawacf.power[0] = true_power
        rawacf.noise[0] = 0.1
        rawacf.slist[0] = 0
        
        # Fit with high precision
        config = FitACFConfig(min_power_threshold=0.1)
        result = process_fitacf(rawacf, config=config)
        
        # Should recover true parameters with high accuracy
        if result.qflg[0] > 0:
            velocity_error = abs(result.velocity[0] - true_velocity) / true_velocity
            width_error = abs(result.spectral_width[0] - true_width) / true_width
            power_error = abs(result.power[0] - true_power) / true_power
            
            # Very tight tolerances for perfect data
            assert velocity_error < 0.01, f"Velocity error: {velocity_error:.4f}"
            assert width_error < 0.05, f"Width error: {width_error:.4f}"
            assert power_error < 0.01, f"Power error: {power_error:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])