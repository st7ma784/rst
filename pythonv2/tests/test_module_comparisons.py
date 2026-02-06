"""
Comprehensive side-by-side module comparison tests

These tests compare C/CUDA and Python implementations for all
SuperDARN processing modules with detailed metrics.
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

from comparison.framework import (
    ComparisonTestFramework, Backend, TestStatus
)
from comparison.fixtures import generate_test_data
from comparison.reporters import JSONReporter, HTMLReporter, ConsoleReporter

# Import processors
from superdarn_gpu.processing.acf import ACFProcessor
from superdarn_gpu.processing.fitacf import FitACFProcessor
from superdarn_gpu.processing.grid import GridProcessor
from superdarn_gpu.processing.convmap import ConvMapProcessor
from superdarn_gpu.core.backends import BackendContext, get_backend


class TestModuleComparisons:
    """
    Side-by-side comparison tests for all modules
    
    Compares correctness and performance between:
    - C CPU implementation
    - C CUDA implementation  
    - Python NumPy implementation
    - Python CuPy implementation
    """
    
    @pytest.fixture
    def framework(self):
        """Initialize test framework"""
        return ComparisonTestFramework(
            tolerance_rtol=1e-4,
            tolerance_atol=1e-6
        )
    
    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory for reports"""
        report_dir = tmp_path / "comparison_reports"
        report_dir.mkdir()
        return report_dir
    
    # =========================================================================
    # ACF Module Tests
    # =========================================================================
    
    class TestACFComparison:
        """ACF processing comparison tests"""
        
        @pytest.fixture
        def acf_test_data(self):
            """Generate ACF test data"""
            return generate_test_data('acf', size='medium')
        
        def test_acf_numpy_vs_cupy(self, acf_test_data):
            """Compare NumPy and CuPy ACF implementations"""
            # NumPy processing
            with BackendContext('numpy'):
                np_processor = ACFProcessor()
                np_result = np_processor.process(acf_test_data)
                np_time = np_processor.stats.get('processing_time', 0)
            
            # CuPy processing (if available)
            try:
                with BackendContext('cupy'):
                    cp_processor = ACFProcessor()
                    cp_result = cp_processor.process(acf_test_data)
                    cp_time = cp_processor.stats.get('processing_time', 0)
                    
                    # Convert result to numpy for comparison
                    cp_acf = cp_result.acf.get() if hasattr(cp_result.acf, 'get') else cp_result.acf
                    
                    # Compare ACF values
                    np.testing.assert_allclose(
                        np_result.acf, cp_acf,
                        rtol=1e-4, atol=1e-6,
                        err_msg="ACF values differ between NumPy and CuPy"
                    )
                    
                    # Compare power values
                    cp_power = cp_result.power.get() if hasattr(cp_result.power, 'get') else cp_result.power
                    np.testing.assert_allclose(
                        np_result.power, cp_power,
                        rtol=1e-4, atol=1e-6,
                        err_msg="Power values differ between NumPy and CuPy"
                    )
                    
                    # Report speedup
                    if np_time > 0 and cp_time > 0:
                        speedup = np_time / cp_time
                        print(f"\nACF CuPy Speedup: {speedup:.2f}x")
                        print(f"  NumPy: {np_time*1000:.2f}ms, CuPy: {cp_time*1000:.2f}ms")
                        
            except ImportError:
                pytest.skip("CuPy not available")
        
        def test_acf_correctness(self, acf_test_data):
            """Test ACF calculation correctness"""
            processor = ACFProcessor()
            result = processor.process(acf_test_data)
            
            # Verify structure
            assert result.acf is not None
            assert result.power is not None
            
            # Lag-0 should be real (power)
            lag0 = result.acf[:, 0]
            assert np.allclose(np.imag(lag0), 0, atol=1e-6)
            
            # Power should be non-negative
            assert np.all(result.power >= 0)
            
            # ACF should decay with lag for signal ranges
            expected = acf_test_data.get('_expected', {})
            signal_ranges = expected.get('signal_ranges', range(10, 50))
            
            for r in list(signal_ranges)[:5]:  # Check first few
                acf_mag = np.abs(result.acf[r, :])
                # Should generally decrease
                assert acf_mag[0] > acf_mag[-1], f"ACF not decaying at range {r}"
    
    # =========================================================================
    # FITACF Module Tests
    # =========================================================================
    
    class TestFitACFComparison:
        """FITACF processing comparison tests"""
        
        @pytest.fixture
        def fitacf_test_data(self):
            """Generate FITACF test data"""
            return generate_test_data('fitacf', size='medium')
        
        def test_fitacf_numpy_vs_cupy(self, fitacf_test_data):
            """Compare NumPy and CuPy FITACF implementations"""
            # NumPy processing
            with BackendContext('numpy'):
                np_processor = FitACFProcessor()
                np_result = np_processor.process(fitacf_test_data)
                np_time = np_processor.stats.get('processing_time', 0)
            
            try:
                with BackendContext('cupy'):
                    cp_processor = FitACFProcessor()
                    cp_result = cp_processor.process(fitacf_test_data)
                    cp_time = cp_processor.stats.get('processing_time', 0)
                    
                    # Compare velocity
                    cp_vel = cp_result.velocity.get() if hasattr(cp_result.velocity, 'get') else cp_result.velocity
                    np.testing.assert_allclose(
                        np_result.velocity, cp_vel,
                        rtol=0.05, atol=10.0,  # 5% tolerance for velocities
                        err_msg="Velocities differ between backends"
                    )
                    
                    # Compare spectral width
                    cp_width = cp_result.spectral_width.get() if hasattr(cp_result.spectral_width, 'get') else cp_result.spectral_width
                    np.testing.assert_allclose(
                        np_result.spectral_width, cp_width,
                        rtol=0.1, atol=20.0,  # 10% tolerance for width
                        err_msg="Spectral widths differ"
                    )
                    
                    if np_time > 0 and cp_time > 0:
                        speedup = np_time / cp_time
                        print(f"\nFITACF CuPy Speedup: {speedup:.2f}x")
                        
            except ImportError:
                pytest.skip("CuPy not available")
        
        def test_fitacf_known_values(self, fitacf_test_data):
            """Test FITACF recovery of known parameters"""
            processor = FitACFProcessor()
            result = processor.process(fitacf_test_data)
            
            expected = fitacf_test_data.get('_expected', {})
            
            errors = {'velocity': [], 'width': []}
            
            for r, params in expected.items():
                if not isinstance(r, int):
                    continue
                    
                # Check velocity recovery
                fitted_vel = float(result.velocity[r])
                true_vel = params['velocity']
                vel_error = abs(fitted_vel - true_vel)
                errors['velocity'].append(vel_error)
                
                # Check width recovery
                fitted_width = float(result.spectral_width[r])
                true_width = params['spectral_width']
                width_error = abs(fitted_width - true_width)
                errors['width'].append(width_error)
            
            # Report average errors
            if errors['velocity']:
                avg_vel_error = np.mean(errors['velocity'])
                avg_width_error = np.mean(errors['width'])
                
                print(f"\nFITACF Recovery Errors:")
                print(f"  Avg velocity error: {avg_vel_error:.1f} m/s")
                print(f"  Avg width error: {avg_width_error:.1f} m/s")
                
                # Should be within reasonable limits
                assert avg_vel_error < 100, "Velocity recovery too inaccurate"
                assert avg_width_error < 50, "Width recovery too inaccurate"
    
    # =========================================================================
    # Grid Module Tests
    # =========================================================================
    
    class TestGridComparison:
        """Grid processing comparison tests"""
        
        @pytest.fixture
        def grid_test_data(self):
            """Generate grid test data"""
            return generate_test_data('grid', size='medium')
        
        def test_grid_numpy_vs_cupy(self, grid_test_data):
            """Compare NumPy and CuPy grid implementations"""
            with BackendContext('numpy'):
                np_processor = GridProcessor()
                np_result = np_processor.process(grid_test_data)
                np_time = np_processor.stats.get('processing_time', 0)
            
            try:
                with BackendContext('cupy'):
                    cp_processor = GridProcessor()
                    cp_result = cp_processor.process(grid_test_data)
                    cp_time = cp_processor.stats.get('processing_time', 0)
                    
                    # Compare gridded values
                    def to_numpy(arr):
                        return arr.get() if hasattr(arr, 'get') else arr
                    
                    np.testing.assert_allclose(
                        to_numpy(np_result.velocity),
                        to_numpy(cp_result.velocity),
                        rtol=1e-4, atol=1e-6,
                        err_msg="Gridded velocities differ"
                    )
                    
                    if np_time > 0 and cp_time > 0:
                        speedup = np_time / cp_time
                        print(f"\nGrid CuPy Speedup: {speedup:.2f}x")
                        
            except ImportError:
                pytest.skip("CuPy not available")
        
        def test_grid_coverage(self, grid_test_data):
            """Test grid spatial coverage"""
            processor = GridProcessor()
            result = processor.process(grid_test_data)
            
            # Check coverage statistics
            coverage = processor.stats.get('spatial_coverage', 0)
            cells_filled = processor.stats.get('grid_cells_filled', 0)
            
            print(f"\nGrid Coverage: {coverage:.1f}%")
            print(f"Cells Filled: {cells_filled}")
            
            # Should have reasonable coverage
            assert cells_filled > 0, "No grid cells filled"
    
    # =========================================================================
    # ConvMap Module Tests
    # =========================================================================
    
    class TestConvMapComparison:
        """Convection map processing comparison tests"""
        
        @pytest.fixture
        def convmap_test_data(self):
            """Generate convmap test data"""
            return generate_test_data('convmap', size='medium')
        
        def test_convmap_numpy_vs_cupy(self, convmap_test_data):
            """Compare NumPy and CuPy convmap implementations"""
            grid_data = convmap_test_data['grid_data']
            imf_by = convmap_test_data.get('imf_by', 0)
            imf_bz = convmap_test_data.get('imf_bz', -5)
            
            with BackendContext('numpy'):
                np_processor = ConvMapProcessor()
                np_result = np_processor.process(grid_data, imf_by, imf_bz)
                np_time = np_processor.stats.get('processing_time', 0)
            
            try:
                with BackendContext('cupy'):
                    cp_processor = ConvMapProcessor()
                    cp_result = cp_processor.process(grid_data, imf_by, imf_bz)
                    cp_time = cp_processor.stats.get('processing_time', 0)
                    
                    # Compare potential field
                    np.testing.assert_allclose(
                        np_result.potential,
                        cp_result.potential,
                        rtol=0.01, atol=100,  # kV tolerance
                        err_msg="Potential fields differ"
                    )
                    
                    # Compare chi-square
                    assert abs(np_result.chi_square_normalized - cp_result.chi_square_normalized) < 0.1
                    
                    if np_time > 0 and cp_time > 0:
                        speedup = np_time / cp_time
                        print(f"\nConvMap CuPy Speedup: {speedup:.2f}x")
                        
            except ImportError:
                pytest.skip("CuPy not available")
        
        def test_convmap_output_structure(self, convmap_test_data):
            """Test convmap output structure"""
            grid_data = convmap_test_data['grid_data']
            
            processor = ConvMapProcessor()
            result = processor.process(grid_data, imf_by=2.0, imf_bz=-5.0)
            
            # Check output structure
            assert result.potential is not None
            assert result.velocity_north is not None
            assert result.velocity_east is not None
            assert result.velocity_magnitude is not None
            assert result.coefficients is not None
            assert result.hm_boundary is not None
            
            # Check shapes
            assert result.potential.shape[0] == len(result.latitudes)
            assert result.potential.shape[1] == len(result.longitudes)
            
            # Check velocity magnitudes are reasonable
            max_vel = np.max(result.velocity_magnitude)
            assert max_vel < 3000, f"Unrealistic velocity: {max_vel} m/s"
            
            print(f"\nConvMap Results:")
            print(f"  Grid size: {result.potential.shape}")
            print(f"  Chi-square: {result.chi_square_normalized:.2f}")
            print(f"  Max velocity: {max_vel:.0f} m/s")
            print(f"  Data points: {result.n_data_points}")


class TestComprehensiveReport:
    """Generate comprehensive comparison report"""
    
    def test_generate_full_report(self, tmp_path):
        """Run all comparisons and generate reports"""
        framework = ComparisonTestFramework()
        
        # Register all modules
        framework.register_module(
            'acf',
            python_processor=ACFProcessor,
            description='Auto-correlation function calculation',
            version='1.16'
        )
        
        framework.register_module(
            'fitacf', 
            python_processor=FitACFProcessor,
            description='ACF curve fitting for velocity/width',
            version='3.0'
        )
        
        framework.register_module(
            'grid',
            python_processor=GridProcessor,
            description='Spatial gridding of radar data',
            version='1.24'
        )
        
        framework.register_module(
            'convmap',
            python_processor=ConvMapProcessor,
            description='Convection map generation',
            version='1.17'
        )
        
        # Run comparisons
        for module_name in ['acf', 'fitacf', 'grid', 'convmap']:
            test_data = generate_test_data(module_name, size='small')
            framework.run_module_comparison(module_name, test_data)
        
        # Generate reports
        json_reporter = JSONReporter()
        html_reporter = HTMLReporter()
        console_reporter = ConsoleReporter()
        
        json_path = tmp_path / 'comparison_report.json'
        html_path = tmp_path / 'comparison_report.html'
        
        json_reporter.generate_report(framework.results, json_path)
        html_reporter.generate_report(framework.results, html_path)
        console_reporter.generate_report(framework.results)
        
        # Verify reports were generated
        assert json_path.exists()
        assert html_path.exists()
        
        print(f"\nReports generated at:")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
        
        # Get summary
        summary = framework.get_summary()
        print(f"\nSummary:")
        print(f"  Modules: {summary['modules_tested']}")
        print(f"  Tests: {summary['passed']}/{summary['total_tests']} passed")


# Run comparison tests when executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
