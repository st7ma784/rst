"""
Comprehensive tests for convection map processing algorithms

These tests validate spherical harmonic fitting, velocity potential calculation,
and comparison between GPU and CPU implementations.
"""

import pytest
import numpy as np
from datetime import datetime

from superdarn_gpu.processing.convmap import (
    ConvMapProcessor, ConvMapConfig, ConvMapData,
    ConvectionModel, CoordinateSystem
)
from superdarn_gpu.core.backends import BackendContext


class TestConvMapProcessor:
    """Test suite for ConvMap processing functionality"""
    
    @pytest.fixture
    def sample_grid_data(self):
        """Create sample grid data with velocity vectors"""
        np.random.seed(42)
        
        n_vectors = 500
        vectors = []
        
        for i in range(n_vectors):
            lat = np.random.uniform(60.0, 85.0)
            lon = np.random.uniform(-150.0, -50.0)
            
            # Create convection-like pattern
            vel_n = 200.0 * np.sin(np.radians(lat - 72.0))
            vel_e = 150.0 * np.cos(np.radians(lon + 100.0))
            
            velocity = np.sqrt(vel_n**2 + vel_e**2)
            azimuth = np.degrees(np.arctan2(vel_e, vel_n))
            
            vectors.append({
                'lat': lat,
                'lon': lon,
                'velocity': velocity + np.random.normal(0, 20),
                'azimuth': azimuth + np.random.normal(0, 5),
                'velocity_error': np.random.uniform(20, 80),
                'power': np.random.uniform(10, 50),
                'spectral_width': np.random.uniform(50, 200),
                'station_id': np.random.choice([65, 66, 67]),
                'beam': np.random.randint(0, 16),
                'range_gate': np.random.randint(10, 70)
            })
        
        return {
            'vectors': vectors,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_convmap_processor_initialization(self):
        """Test ConvMap processor initialization"""
        config = ConvMapConfig(
            model=ConvectionModel.STATISTICAL,
            order=8,
            latmin=60.0
        )
        
        processor = ConvMapProcessor(config=config)
        
        assert processor.config == config
        assert processor.name == "ConvMap Processor"
        assert processor.config.order == 8
    
    def test_input_validation(self, sample_grid_data):
        """Test input validation"""
        processor = ConvMapProcessor()
        
        # Valid input
        assert processor.validate_input(sample_grid_data) == True
        
        # Invalid inputs
        assert processor.validate_input(None) == False
        assert processor.validate_input({}) == False
        assert processor.validate_input({'vectors': []}) == False
    
    def test_basic_convmap_processing(self, sample_grid_data):
        """Test basic convection map generation"""
        processor = ConvMapProcessor()
        result = processor.process(sample_grid_data, imf_by=0, imf_bz=-5)
        
        # Verify output structure
        assert isinstance(result, ConvMapData)
        assert result.potential is not None
        assert result.velocity_north is not None
        assert result.velocity_east is not None
        assert result.velocity_magnitude is not None
        assert result.coefficients is not None
        assert result.hm_boundary is not None
        
        # Check shapes
        assert result.potential.shape[0] == len(result.latitudes)
        assert result.potential.shape[1] == len(result.longitudes)
        assert result.velocity_north.shape == result.potential.shape
        
        # Check chi-square is calculated
        assert result.chi_square >= 0
        assert result.chi_square_normalized >= 0
    
    def test_different_imf_conditions(self, sample_grid_data):
        """Test convection maps under different IMF conditions"""
        processor = ConvMapProcessor()
        
        # Southward IMF (more active convection)
        result_south = processor.process(sample_grid_data, imf_by=0, imf_bz=-10)
        
        # Northward IMF (weaker convection)
        result_north = processor.process(sample_grid_data, imf_by=0, imf_bz=5)
        
        # Generally expect different patterns
        assert not np.allclose(result_south.potential, result_north.potential)
        
        # IMF By asymmetry
        result_by_pos = processor.process(sample_grid_data, imf_by=5, imf_bz=-5)
        result_by_neg = processor.process(sample_grid_data, imf_by=-5, imf_bz=-5)
        
        # Different By should produce different patterns
        assert not np.allclose(result_by_pos.potential, result_by_neg.potential)
    
    def test_spherical_harmonic_order(self, sample_grid_data):
        """Test different spherical harmonic orders"""
        orders = [4, 8, 12]
        results = {}
        
        for order in orders:
            config = ConvMapConfig(order=order)
            processor = ConvMapProcessor(config=config)
            results[order] = processor.process(sample_grid_data, imf_by=0, imf_bz=-5)
            
            # Higher order should have more coefficients
            n_coeffs_expected = (order + 1) * (order + 2) // 2
            assert results[order].coefficients.shape[1] == order + 1
    
    def test_velocity_reasonableness(self, sample_grid_data):
        """Test that output velocities are physically reasonable"""
        processor = ConvMapProcessor()
        result = processor.process(sample_grid_data, imf_by=0, imf_bz=-5)
        
        # Maximum velocities should be reasonable (< 3 km/s)
        max_vel = np.max(result.velocity_magnitude)
        assert max_vel < 3000, f"Unrealistic velocity: {max_vel} m/s"
        
        # Should have some structure (not all zeros)
        assert np.std(result.velocity_magnitude) > 0
    
    def test_hm_boundary_calculation(self, sample_grid_data):
        """Test Heppner-Maynard boundary calculation"""
        processor = ConvMapProcessor()
        result = processor.process(sample_grid_data, imf_by=0, imf_bz=-5)
        
        # Boundary should have MLT resolution
        assert len(result.hm_boundary) == 48  # 0.5 hour resolution
        
        # Boundary latitudes should be reasonable
        assert np.min(result.hm_boundary) >= 50
        assert np.max(result.hm_boundary) <= 80
    
    def test_cpu_gpu_consistency(self, sample_grid_data):
        """Test that CPU and GPU implementations produce similar results"""
        
        # CPU processing
        with BackendContext('numpy'):
            cpu_processor = ConvMapProcessor()
            cpu_result = cpu_processor.process(sample_grid_data, imf_by=2, imf_bz=-5)
            cpu_time = cpu_processor.stats.get('processing_time', 0)
        
        # GPU processing (if available)
        try:
            with BackendContext('cupy'):
                gpu_processor = ConvMapProcessor()
                gpu_result = gpu_processor.process(sample_grid_data, imf_by=2, imf_bz=-5)
                gpu_time = gpu_processor.stats.get('processing_time', 0)
                
                # Compare potential fields
                np.testing.assert_allclose(
                    cpu_result.potential,
                    gpu_result.potential,
                    rtol=0.01, atol=100,
                    err_msg="Potential fields differ between CPU and GPU"
                )
                
                # Compare velocity magnitudes
                np.testing.assert_allclose(
                    cpu_result.velocity_magnitude,
                    gpu_result.velocity_magnitude,
                    rtol=0.05, atol=10,
                    err_msg="Velocities differ between CPU and GPU"
                )
                
                # Report performance
                if cpu_time > 0 and gpu_time > 0:
                    speedup = cpu_time / gpu_time
                    print(f"\nConvMap GPU Speedup: {speedup:.2f}x")
                    print(f"  CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms")
                    
        except ImportError:
            pytest.skip("CuPy not available for GPU testing")
    
    def test_model_weighting(self, sample_grid_data):
        """Test effect of model weighting parameter"""
        
        # Pure data-driven (no model)
        config_no_model = ConvMapConfig(model_weight=0.0)
        processor_no_model = ConvMapProcessor(config=config_no_model)
        result_no_model = processor_no_model.process(sample_grid_data, imf_by=0, imf_bz=-5)
        
        # Heavy model weighting
        config_model = ConvMapConfig(model_weight=0.8)
        processor_model = ConvMapProcessor(config=config_model)
        result_model = processor_model.process(sample_grid_data, imf_by=0, imf_bz=-5)
        
        # Results should differ
        assert not np.allclose(result_no_model.potential, result_model.potential)
    
    def test_error_weighting(self, sample_grid_data):
        """Test effect of error weighting"""
        # With error weighting
        config_weighted = ConvMapConfig(error_weighting=True)
        processor_weighted = ConvMapProcessor(config=config_weighted)
        result_weighted = processor_weighted.process(sample_grid_data, imf_by=0, imf_bz=-5)
        
        # Without error weighting
        config_uniform = ConvMapConfig(error_weighting=False)
        processor_uniform = ConvMapProcessor(config=config_uniform)
        result_uniform = processor_uniform.process(sample_grid_data, imf_by=0, imf_bz=-5)
        
        # Chi-square should be different
        # Note: actual values depend on data quality distribution
        assert result_weighted.chi_square_normalized != result_uniform.chi_square_normalized
    
    def test_processing_statistics(self, sample_grid_data):
        """Test that processing statistics are tracked"""
        processor = ConvMapProcessor()
        result = processor.process(sample_grid_data, imf_by=0, imf_bz=-5)
        
        # Check stats are populated
        assert processor.stats['vectors_used'] > 0
        assert processor.stats['processing_time'] > 0
        
        # vectors_used + vectors_rejected should equal input
        total = processor.stats['vectors_used'] + processor.stats['vectors_rejected']
        assert total == len(sample_grid_data['vectors'])


class TestConvMapPerformance:
    """Performance tests for ConvMap processing"""
    
    @pytest.fixture
    def large_grid_data(self):
        """Create large grid dataset for performance testing"""
        np.random.seed(42)
        
        n_vectors = 5000
        vectors = []
        
        for i in range(n_vectors):
            vectors.append({
                'lat': np.random.uniform(55.0, 88.0),
                'lon': np.random.uniform(-180.0, 180.0),
                'velocity': np.random.uniform(50, 800),
                'azimuth': np.random.uniform(0, 360),
                'velocity_error': np.random.uniform(10, 100),
                'power': np.random.uniform(5, 60),
                'spectral_width': np.random.uniform(30, 300),
                'station_id': 65,
                'beam': 8,
                'range_gate': 40
            })
        
        return {
            'vectors': vectors,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_large_dataset_performance(self, large_grid_data):
        """Test performance with large dataset"""
        import time
        
        processor = ConvMapProcessor()
        
        start = time.time()
        result = processor.process(large_grid_data, imf_by=0, imf_bz=-5)
        elapsed = time.time() - start
        
        print(f"\nLarge dataset ({len(large_grid_data['vectors'])} vectors):")
        print(f"  Processing time: {elapsed*1000:.2f}ms")
        print(f"  Vectors used: {processor.stats['vectors_used']}")
        print(f"  Chi-square: {result.chi_square_normalized:.2f}")
        
        # Should complete in reasonable time
        assert elapsed < 30.0, f"Processing took too long: {elapsed}s"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
