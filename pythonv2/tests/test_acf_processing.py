"""
Comprehensive tests for ACF processing algorithms

These tests validate the ACF calculation against known reference data
and ensure GPU/CPU implementations produce identical results.
"""

import pytest
import numpy as np
import tempfile
import os
from datetime import datetime

# Import modules to test
from superdarn_gpu.processing.acf import ACFProcessor, ACFConfig, calculate_acf, create_lag_table
from superdarn_gpu.core.datatypes import RadarParameters, RawACF
from superdarn_gpu.core.backends import BackendContext, get_backend, Backend


class TestACFProcessor:
    """Test suite for ACF processing functionality"""
    
    @pytest.fixture
    def sample_radar_params(self):
        """Create sample radar parameters for testing"""
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
    def synthetic_iq_samples(self, sample_radar_params):
        """Generate synthetic I/Q samples with known ACF properties"""
        prm = sample_radar_params
        nsamp = prm.nrang * prm.nave * 2  # 2 samples per range gate
        
        # Create synthetic signal with known Doppler and spectral characteristics
        samples = []
        
        for range_idx in range(prm.nrang):
            for avg in range(prm.nave):
                
                # Add realistic ionospheric scatter for some ranges
                if 20 <= range_idx <= 50:
                    # Simulate Doppler-shifted signal
                    velocity = 400.0 * (1 if range_idx % 2 else -1)  # ±400 m/s
                    frequency = velocity / (3e8 / prm.tfreq / 1e3)  # Doppler frequency
                    
                    # Create coherent signal with decay
                    amplitude = 100.0 * np.exp(-range_idx * 0.02)
                    phase = 2 * np.pi * frequency * avg
                    
                    # I/Q samples
                    i_sample = amplitude * np.cos(phase) + np.random.normal(0, 5)
                    q_sample = amplitude * np.sin(phase) + np.random.normal(0, 5)
                else:
                    # Noise-only ranges
                    i_sample = np.random.normal(0, 10)
                    q_sample = np.random.normal(0, 10)
                
                samples.append(complex(i_sample, q_sample))
        
        return np.array(samples, dtype=np.complex64)
    
    @pytest.fixture
    def sample_lag_table(self, sample_radar_params):
        """Create sample lag table for testing"""
        # Standard 18-lag table for SuperDARN
        pulse_table = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153]
        return create_lag_table(pulse_table, sample_radar_params.mpinc, sample_radar_params.mplgs)
    
    def test_acf_processor_initialization(self):
        """Test ACF processor initialization"""
        config = ACFConfig(
            bad_sample_threshold=1e6,
            dc_offset_removal=True,
            xcf_processing=True
        )
        
        processor = ACFProcessor(config=config)
        
        assert processor.config == config
        assert processor.name == "ACF Processor"
        assert 'samples_processed' in processor.stats
    
    def test_input_validation(self, sample_radar_params, synthetic_iq_samples, sample_lag_table):
        """Test input validation for ACF processing"""
        processor = ACFProcessor()
        
        # Valid input
        valid_input = {
            'samples': synthetic_iq_samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        assert processor.validate_input(valid_input) == True
        
        # Invalid inputs
        assert processor.validate_input({}) == False
        assert processor.validate_input({'samples': None, 'prm': None, 'lag_table': None}) == False
        assert processor.validate_input({'samples': [], 'prm': sample_radar_params, 'lag_table': sample_lag_table}) == False
    
    def test_memory_estimation(self, sample_radar_params, synthetic_iq_samples, sample_lag_table):
        """Test memory requirement estimation"""
        processor = ACFProcessor()
        
        iq_data = {
            'samples': synthetic_iq_samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        memory_estimate = processor.get_memory_estimate(iq_data)
        
        assert memory_estimate > 0
        assert isinstance(memory_estimate, int)
        
        # Should be reasonable estimate based on data size
        expected_min = len(synthetic_iq_samples) * 8  # Complex64 input size
        assert memory_estimate > expected_min
    
    def test_acf_calculation_basic(self, sample_radar_params, synthetic_iq_samples, sample_lag_table):
        """Test basic ACF calculation functionality"""
        processor = ACFProcessor()
        
        iq_data = {
            'samples': synthetic_iq_samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        result = processor.process(iq_data)
        
        # Verify output structure
        assert isinstance(result, RawACF)
        assert result.nrang == sample_radar_params.nrang
        assert result.mplgs == sample_radar_params.mplgs
        assert result.acf.shape == (sample_radar_params.nrang, sample_radar_params.mplgs)
        
        # Verify ACF properties
        # Lag-0 should be real and positive (power)
        assert np.all(np.imag(result.acf[:, 0]) < 1e-6)  # Nearly real
        assert np.all(np.real(result.acf[:, 0]) >= 0)     # Non-negative power
        
        # Power should match lag-0 real part
        np.testing.assert_allclose(result.power, np.real(result.acf[:, 0]), rtol=1e-6)
    
    def test_cpu_gpu_consistency(self, sample_radar_params, synthetic_iq_samples, sample_lag_table):
        """Test that CPU and GPU implementations produce identical results"""
        
        iq_data = {
            'samples': synthetic_iq_samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        # Process with CPU backend
        with BackendContext('numpy'):
            cpu_processor = ACFProcessor()
            cpu_result = cpu_processor.process(iq_data)
        
        # Process with GPU backend (if available)
        try:
            with BackendContext('cupy'):
                gpu_processor = ACFProcessor()
                gpu_result = gpu_processor.process(iq_data)
                
                # Convert GPU result to CPU for comparison
                gpu_acf_cpu = gpu_result.to_cpu().acf
                
                # Results should be nearly identical
                np.testing.assert_allclose(cpu_result.acf, gpu_acf_cpu, rtol=1e-6, atol=1e-8)
                np.testing.assert_allclose(cpu_result.power, gpu_result.to_cpu().power, rtol=1e-6)
                
        except ImportError:
            pytest.skip("CuPy not available for GPU testing")
    
    def test_dc_offset_removal(self, sample_radar_params, sample_lag_table):
        """Test DC offset removal functionality"""
        # Create samples with known DC offset
        nsamp = sample_radar_params.nrang * sample_radar_params.nave * 2
        samples_with_dc = np.random.randn(nsamp).astype(np.complex64)
        dc_offset = 100.0 + 50.0j
        samples_with_dc += dc_offset
        
        iq_data = {
            'samples': samples_with_dc,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        # Process with DC removal enabled
        config = ACFConfig(dc_offset_removal=True)
        processor = ACFProcessor(config=config)
        result_with_removal = processor.process(iq_data)
        
        # Process with DC removal disabled
        config = ACFConfig(dc_offset_removal=False)
        processor = ACFProcessor(config=config)
        result_without_removal = processor.process(iq_data)
        
        # Results should be different
        assert not np.allclose(result_with_removal.acf, result_without_removal.acf)
    
    def test_bad_sample_detection(self, sample_radar_params, sample_lag_table):
        """Test bad sample detection and removal"""
        # Create samples with some saturated (bad) samples
        nsamp = sample_radar_params.nrang * sample_radar_params.nave * 2
        samples = np.random.randn(nsamp).astype(np.complex64) * 10
        
        # Add some saturated samples
        bad_indices = [100, 200, 300]
        samples[bad_indices] = 1e8 + 1e8j  # Very large values
        
        iq_data = {
            'samples': samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        config = ACFConfig(bad_sample_threshold=1e6)
        processor = ACFProcessor(config=config)
        result = processor.process(iq_data)
        
        # Should have detected bad samples
        assert processor.stats['bad_samples_detected'] > 0
        
        # ACF should still be calculated (bad samples replaced)
        assert result.acf.shape == (sample_radar_params.nrang, sample_radar_params.mplgs)
    
    def test_xcf_calculation(self, sample_radar_params, synthetic_iq_samples, sample_lag_table):
        """Test cross-correlation function calculation"""
        # Create interferometer samples
        int_samples = synthetic_iq_samples * 0.8 + np.random.randn(len(synthetic_iq_samples)) * 5
        
        iq_data = {
            'samples': synthetic_iq_samples,
            'int_samples': int_samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        config = ACFConfig(xcf_processing=True)
        processor = ACFProcessor(config=config)
        result = processor.process(iq_data)
        
        # Should have XCF data
        assert result.xcf is not None
        assert result.xcf.shape == (sample_radar_params.nrang, sample_radar_params.mplgs)
    
    def test_lag_table_creation(self):
        """Test lag table creation function"""
        pulse_table = [0, 1, 3, 6, 10, 15, 21, 28]
        mpinc = 1500
        mplgs = 8
        
        lag_table = create_lag_table(pulse_table, mpinc, mplgs)
        
        assert len(lag_table) == mplgs
        assert lag_table[0] == 0  # First lag should be zero
        assert np.all(lag_table >= 0)  # All lags should be non-negative
        
        # Check that lags increase appropriately
        assert lag_table[1] > lag_table[0]
    
    def test_noise_estimation(self, sample_radar_params, sample_lag_table):
        """Test noise level estimation"""
        # Create samples with known noise characteristics
        nsamp = sample_radar_params.nrang * sample_radar_params.nave * 2
        noise_level = 15.0
        samples = np.random.randn(nsamp).astype(np.complex64) * noise_level
        
        iq_data = {
            'samples': samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        config = ACFConfig(noise_level_estimation=True)
        processor = ACFProcessor(config=config)
        result = processor.process(iq_data)
        
        # Noise estimate should be reasonable
        estimated_noise = np.mean(result.noise)
        assert 0.5 * noise_level < estimated_noise < 2.0 * noise_level
    
    def test_performance_statistics(self, sample_radar_params, synthetic_iq_samples, sample_lag_table):
        """Test performance statistics tracking"""
        processor = ACFProcessor()
        
        iq_data = {
            'samples': synthetic_iq_samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        # Process data
        result = processor.process(iq_data)
        
        # Check statistics were updated
        assert processor.stats['samples_processed'] == len(synthetic_iq_samples)
        assert processor.stats['ranges_processed'] == sample_radar_params.nrang
        assert processor.stats['processing_time'] >= 0
    
    def test_convenience_function(self, sample_radar_params, synthetic_iq_samples, sample_lag_table):
        """Test convenience function for direct ACF calculation"""
        iq_data = {
            'samples': synthetic_iq_samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        # Test direct function call
        result = calculate_acf(iq_data)
        
        assert isinstance(result, RawACF)
        assert result.nrang == sample_radar_params.nrang
        assert result.mplgs == sample_radar_params.mplgs
    
    @pytest.mark.parametrize("config_param,expected_behavior", [
        ("bad_sample_threshold", "different_results"),
        ("dc_offset_removal", "different_results"),
        ("xcf_processing", "xcf_presence"),
    ])
    def test_configuration_parameters(self, config_param, expected_behavior, 
                                    sample_radar_params, synthetic_iq_samples, sample_lag_table):
        """Test various configuration parameters"""
        base_config = ACFConfig()
        
        iq_data = {
            'samples': synthetic_iq_samples,
            'prm': sample_radar_params,
            'lag_table': sample_lag_table
        }
        
        if config_param == "bad_sample_threshold":
            config1 = ACFConfig(bad_sample_threshold=1e3)
            config2 = ACFConfig(bad_sample_threshold=1e9)
        elif config_param == "dc_offset_removal":
            config1 = ACFConfig(dc_offset_removal=True)
            config2 = ACFConfig(dc_offset_removal=False)
        elif config_param == "xcf_processing":
            iq_data['int_samples'] = synthetic_iq_samples * 0.9
            config1 = ACFConfig(xcf_processing=True)
            config2 = ACFConfig(xcf_processing=False)
        
        processor1 = ACFProcessor(config=config1)
        processor2 = ACFProcessor(config=config2)
        
        result1 = processor1.process(iq_data)
        result2 = processor2.process(iq_data)
        
        if expected_behavior == "different_results":
            # Results should be different
            assert not np.allclose(result1.acf, result2.acf, rtol=1e-6)
        elif expected_behavior == "xcf_presence":
            # XCF should be present in one result but not the other
            assert (result1.xcf is not None) != (result2.xcf is not None)


class TestACFValidation:
    """Integration tests with reference data validation"""
    
    def test_known_signal_acf(self):
        """Test ACF calculation with analytically known signal"""
        # Create a pure sinusoid - ACF should match theoretical expectation
        frequency = 100.0  # Hz
        amplitude = 50.0
        sample_rate = 1000.0  # Hz
        duration = 1.0  # seconds
        
        t = np.arange(0, duration, 1/sample_rate)
        signal = amplitude * np.exp(2j * np.pi * frequency * t)
        
        # Add noise
        noise_level = 5.0
        noisy_signal = signal + noise_level * (np.random.randn(len(signal)) + 
                                              1j * np.random.randn(len(signal)))
        
        # Create radar parameters for this test
        prm = RadarParameters(
            station_id=1, beam_number=0, scan_flag=1, channel=1, cp_id=1,
            nave=10, lagfr=100, smsep=100, txpow=1000, atten=0,
            noise_search=noise_level, noise_mean=noise_level,
            tfreq=int(frequency), nrang=10, frang=100, rsep=45, xcf=0,
            mppul=8, mpinc=100, mplgs=10, txpl=100,
            intt_sc=1, intt_us=0, timestamp=datetime.now()
        )
        
        lag_table = np.arange(10, dtype=np.int32) * 10  # Simple lag table
        
        iq_data = {
            'samples': noisy_signal.astype(np.complex64),
            'prm': prm,
            'lag_table': lag_table
        }
        
        result = calculate_acf(iq_data)
        
        # Theoretical ACF for sinusoid should have specific properties
        # Lag-0 should have maximum power
        lag0_power = np.abs(result.acf[0, 0])
        assert lag0_power > 0
        
        # Higher lags should show decay for noisy signal
        higher_lag_power = np.abs(result.acf[0, -1])
        assert higher_lag_power < lag0_power


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])