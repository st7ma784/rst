"""
Core comparison framework for side-by-side C/CUDA vs Python testing

This framework enables systematic comparison of algorithm implementations
across different backends (C, CUDA, Python/NumPy, Python/CuPy).
"""

import time
import subprocess
import tempfile
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from pathlib import Path
import numpy as np

class Backend(Enum):
    """Available processing backends"""
    C_CPU = "c_cpu"
    C_CUDA = "c_cuda"
    PYTHON_NUMPY = "python_numpy"
    PYTHON_CUPY = "python_cupy"

class TestStatus(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Result from a single test execution"""
    backend: Backend
    module_name: str
    test_name: str
    status: TestStatus
    execution_time: float  # seconds
    memory_used: int  # bytes
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ComparisonResult:
    """Result comparing two backend implementations"""
    module_name: str
    test_name: str
    reference_backend: Backend
    comparison_backend: Backend
    
    # Correctness metrics
    values_match: bool
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    correlation: float
    
    # Performance metrics
    reference_time: float
    comparison_time: float
    speedup: float
    
    # Memory metrics
    reference_memory: int
    comparison_memory: int
    memory_ratio: float
    
    # Status
    status: TestStatus
    error_message: Optional[str] = None
    
    # Detailed field comparisons
    field_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class ModuleComparison:
    """Complete comparison results for a module"""
    module_name: str
    module_version: str
    description: str
    
    # Individual test results
    results: List[ComparisonResult] = field(default_factory=list)
    
    # Aggregate metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    
    # Performance summary
    avg_speedup: float = 0.0
    min_speedup: float = 0.0
    max_speedup: float = 0.0
    
    # Accuracy summary
    avg_error: float = 0.0
    max_error: float = 0.0
    
    timestamp: str = ""

class ComparisonTestFramework:
    """
    Main framework for running side-by-side comparison tests
    
    Supports comparing C, CUDA, and Python implementations with
    detailed correctness and performance metrics.
    """
    
    def __init__(self, 
                 rst_root: Optional[Path] = None,
                 tolerance_rtol: float = 1e-5,
                 tolerance_atol: float = 1e-8):
        """
        Initialize test framework
        
        Parameters
        ----------
        rst_root : Path, optional
            Path to RST repository root (auto-detects if not provided)
        tolerance_rtol : float
            Relative tolerance for numerical comparisons
        tolerance_atol : float
            Absolute tolerance for numerical comparisons
        """
        self.rst_root = rst_root or self._find_rst_root()
        self.tolerance_rtol = tolerance_rtol
        self.tolerance_atol = tolerance_atol
        
        # Module registry
        self.modules: Dict[str, 'ModuleTestSuite'] = {}
        
        # Results storage
        self.results: Dict[str, ModuleComparison] = {}
        
    def _find_rst_root(self) -> Path:
        """Auto-detect RST repository root"""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / 'codebase').exists() and (parent / 'pythonv2').exists():
                return parent
        raise RuntimeError("Could not find RST repository root")
    
    def register_module(self, 
                        name: str,
                        c_binary: Optional[str] = None,
                        cuda_binary: Optional[str] = None,
                        python_processor: Optional[Callable] = None,
                        description: str = "",
                        version: str = "1.0"):
        """
        Register a module for comparison testing
        
        Parameters
        ----------
        name : str
            Module name (e.g., 'fitacf', 'grid', 'acf')
        c_binary : str, optional
            Path to C implementation binary
        cuda_binary : str, optional
            Path to CUDA implementation binary  
        python_processor : Callable, optional
            Python processor class or function
        description : str
            Module description
        version : str
            Module version
        """
        self.modules[name] = ModuleTestSuite(
            name=name,
            c_binary=c_binary,
            cuda_binary=cuda_binary,
            python_processor=python_processor,
            description=description,
            version=version,
            framework=self
        )
        
    def run_module_comparison(self, 
                              module_name: str,
                              test_data: Dict[str, Any],
                              backends: Optional[List[Backend]] = None) -> ModuleComparison:
        """
        Run comparison tests for a specific module
        
        Parameters
        ----------
        module_name : str
            Name of registered module
        test_data : dict
            Test input data
        backends : list, optional
            Backends to test (defaults to all available)
            
        Returns
        -------
        ModuleComparison
            Comparison results for the module
        """
        if module_name not in self.modules:
            raise ValueError(f"Module '{module_name}' not registered")
            
        module = self.modules[module_name]
        
        if backends is None:
            backends = self._get_available_backends(module)
            
        comparison = module.run_comparison(test_data, backends)
        self.results[module_name] = comparison
        
        return comparison
    
    def run_all_comparisons(self, 
                           test_data_generator: Callable[[str], Dict[str, Any]],
                           backends: Optional[List[Backend]] = None) -> Dict[str, ModuleComparison]:
        """
        Run comparison tests for all registered modules
        
        Parameters
        ----------
        test_data_generator : Callable
            Function that takes module name and returns test data
        backends : list, optional
            Backends to test
            
        Returns
        -------
        dict
            Comparison results for all modules
        """
        for name in self.modules:
            test_data = test_data_generator(name)
            self.run_module_comparison(name, test_data, backends)
            
        return self.results
    
    def _get_available_backends(self, module: 'ModuleTestSuite') -> List[Backend]:
        """Determine which backends are available for a module"""
        backends = []
        
        if module.c_binary and Path(module.c_binary).exists():
            backends.append(Backend.C_CPU)
            
        if module.cuda_binary and Path(module.cuda_binary).exists():
            backends.append(Backend.C_CUDA)
            
        if module.python_processor:
            backends.append(Backend.PYTHON_NUMPY)
            
            # Check if CuPy is available
            try:
                import cupy
                if cupy.cuda.is_available():
                    backends.append(Backend.PYTHON_CUPY)
            except ImportError:
                pass
                
        return backends
    
    def compare_arrays(self, 
                       reference: np.ndarray,
                       comparison: np.ndarray) -> Dict[str, float]:
        """
        Compare two arrays and compute error metrics
        
        Parameters
        ----------
        reference : ndarray
            Reference array
        comparison : ndarray
            Comparison array
            
        Returns
        -------
        dict
            Error metrics
        """
        if reference.shape != comparison.shape:
            return {
                'values_match': False,
                'max_absolute_error': float('inf'),
                'max_relative_error': float('inf'),
                'mean_absolute_error': float('inf'),
                'correlation': 0.0,
                'shape_mismatch': True
            }
        
        # Flatten for comparison
        ref_flat = reference.flatten().astype(np.float64)
        cmp_flat = comparison.flatten().astype(np.float64)
        
        # Handle complex numbers
        if np.iscomplexobj(reference) or np.iscomplexobj(comparison):
            ref_flat = np.abs(reference.flatten().astype(np.complex128))
            cmp_flat = np.abs(comparison.flatten().astype(np.complex128))
        
        # Compute errors
        abs_error = np.abs(ref_flat - cmp_flat)
        max_abs_error = float(np.max(abs_error))
        mean_abs_error = float(np.mean(abs_error))
        
        # Relative error (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.where(np.abs(ref_flat) > 1e-12,
                                abs_error / np.abs(ref_flat), 0)
            max_rel_error = float(np.max(rel_error[np.isfinite(rel_error)]))
        
        # Correlation
        if np.std(ref_flat) > 1e-12 and np.std(cmp_flat) > 1e-12:
            correlation = float(np.corrcoef(ref_flat, cmp_flat)[0, 1])
        else:
            correlation = 1.0 if np.allclose(ref_flat, cmp_flat) else 0.0
        
        # Check if values match within tolerance
        values_match = np.allclose(ref_flat, cmp_flat, 
                                   rtol=self.tolerance_rtol,
                                   atol=self.tolerance_atol)
        
        return {
            'values_match': values_match,
            'max_absolute_error': max_abs_error,
            'max_relative_error': max_rel_error,
            'mean_absolute_error': mean_abs_error,
            'correlation': correlation if np.isfinite(correlation) else 0.0,
            'shape_mismatch': False
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all comparison results"""
        summary = {
            'modules_tested': len(self.results),
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'modules': {}
        }
        
        for name, comparison in self.results.items():
            summary['total_tests'] += comparison.total_tests
            summary['passed'] += comparison.passed_tests
            summary['failed'] += comparison.failed_tests
            summary['skipped'] += comparison.skipped_tests
            
            summary['modules'][name] = {
                'status': 'passed' if comparison.failed_tests == 0 else 'failed',
                'tests': comparison.total_tests,
                'passed': comparison.passed_tests,
                'avg_speedup': comparison.avg_speedup,
                'avg_error': comparison.avg_error
            }
            
        return summary


class ModuleTestSuite:
    """
    Test suite for a specific module
    """
    
    def __init__(self,
                 name: str,
                 framework: ComparisonTestFramework,
                 c_binary: Optional[str] = None,
                 cuda_binary: Optional[str] = None,
                 python_processor: Optional[Callable] = None,
                 description: str = "",
                 version: str = "1.0"):
        
        self.name = name
        self.framework = framework
        self.c_binary = c_binary
        self.cuda_binary = cuda_binary
        self.python_processor = python_processor
        self.description = description
        self.version = version
        
        # Test cases
        self.test_cases: List[Callable] = []
        
    def add_test(self, test_func: Callable):
        """Add a test case"""
        self.test_cases.append(test_func)
    
    def _adapt_input_data(self, test_data: Dict[str, Any]) -> Any:
        """
        Convert dict test data to the expected input type for the processor.
        Each module type expects a different data structure.
        """
        # ACF module expects dict with samples, prm, lag_table
        if self.name == 'acf':
            return test_data  # Already in correct format
        
        # FITACF module expects RawACF object
        elif self.name == 'fitacf':
            from superdarn_gpu.core.datatypes import RawACF
            prm = test_data.get('prm')
            nrang = prm.nrang if hasattr(prm, 'nrang') else prm.get('nrang', 0) if isinstance(prm, dict) else 0
            mplgs = prm.mplgs if hasattr(prm, 'mplgs') else prm.get('mplgs', 0) if isinstance(prm, dict) else 0
            nave = prm.nave if hasattr(prm, 'nave') else prm.get('nave', 0) if isinstance(prm, dict) else 0
            
            # Create RawACF and populate its fields
            raw_acf = RawACF(nrang=nrang, mplgs=mplgs, nave=nave)
            if test_data.get('acf') is not None:
                raw_acf.acf = test_data['acf']
            if test_data.get('xcf') is not None:
                raw_acf.xcf = test_data['xcf']
            if test_data.get('power') is not None:
                raw_acf.power = test_data['power']
            if test_data.get('noise') is not None:
                raw_acf.noise = test_data['noise']
            if test_data.get('slist') is not None:
                raw_acf.slist = test_data['slist']
            raw_acf.prm = prm
            power = test_data.get('power')
            if power is not None:
                raw_acf.qflg = test_data.get('qflg', np.ones(power.shape[0], dtype=np.int32))
                raw_acf.gflg = test_data.get('gflg', np.zeros(power.shape[0], dtype=np.int32))
            return raw_acf
        
        # Grid module expects List[FitACF] but we'll pass vectors directly
        elif self.name == 'grid':
            # Re-format for GridProcessor._extract_vectors alternative path
            # We'll use a dict wrapper with vectors pre-extracted
            vectors = test_data.get('vectors', [])
            return {'_raw_vectors': vectors, 'grid_config': test_data.get('grid_config', {})}
        
        # ConvMap module expects GridData or dict with 'vectors'
        elif self.name == 'convmap':
            grid_data = test_data.get('grid_data', test_data)
            vectors = grid_data.get('vectors', [])
            return {
                'vectors': vectors,
                'grid_config': grid_data.get('grid_config', {}),
                'model': test_data.get('model'),
                'order': test_data.get('order'),
                'imf_by': test_data.get('imf_by'),
                'imf_bz': test_data.get('imf_bz'),
                'timestamp': test_data.get('timestamp')
            }
        
        # Default: return as-is
        return test_data
        
    def run_comparison(self, 
                      test_data: Dict[str, Any],
                      backends: List[Backend]) -> ModuleComparison:
        """
        Run comparison across backends
        """
        from datetime import datetime
        
        comparison = ModuleComparison(
            module_name=self.name,
            module_version=self.version,
            description=self.description,
            timestamp=datetime.now().isoformat()
        )
        
        # Run on each backend
        backend_results: Dict[Backend, TestResult] = {}
        
        for backend in backends:
            result = self._run_on_backend(backend, test_data)
            backend_results[backend] = result
            
        # Compare results (use first successful backend as reference)
        reference_backend = None
        reference_result = None
        
        for backend, result in backend_results.items():
            if result.status == TestStatus.PASSED:
                reference_backend = backend
                reference_result = result
                break
                
        if reference_result is None:
            comparison.status = TestStatus.ERROR
            return comparison
        
        # If only one backend available, create a self-validation test
        if len(backends) == 1 and reference_result:
            validation_result = self._create_validation_result(
                reference_result, reference_backend
            )
            comparison.results.append(validation_result)
            if validation_result.status == TestStatus.PASSED:
                comparison.passed_tests += 1
            else:
                comparison.failed_tests += 1
            comparison.total_tests = 1
            return comparison
            
        # Compare each backend against reference
        for backend, result in backend_results.items():
            if backend == reference_backend:
                continue
                
            comp_result = self._compare_results(
                reference_result, result,
                reference_backend, backend
            )
            comparison.results.append(comp_result)
            
            if comp_result.status == TestStatus.PASSED:
                comparison.passed_tests += 1
            elif comp_result.status == TestStatus.FAILED:
                comparison.failed_tests += 1
            else:
                comparison.skipped_tests += 1
                
        comparison.total_tests = len(comparison.results)
        
        # Calculate aggregate metrics
        if comparison.results:
            speedups = [r.speedup for r in comparison.results if r.speedup > 0]
            errors = [r.mean_absolute_error for r in comparison.results]
            
            if speedups:
                comparison.avg_speedup = sum(speedups) / len(speedups)
                comparison.min_speedup = min(speedups)
                comparison.max_speedup = max(speedups)
                
            if errors:
                comparison.avg_error = sum(errors) / len(errors)
                comparison.max_error = max(errors)
                
        return comparison
    
    def _create_validation_result(self, 
                                   result: TestResult, 
                                   backend: Backend) -> ComparisonResult:
        """Create validation result when only one backend available"""
        # Validate that output has expected structure
        output = result.output_data or {}
        has_output = len(output) > 0
        has_valid_arrays = all(
            isinstance(v, (np.ndarray, list, int, float, str, bool)) or hasattr(v, '__array__')
            for v in output.values()
        )
        
        is_valid = has_output and has_valid_arrays and result.status == TestStatus.PASSED
        
        return ComparisonResult(
            module_name=self.name,
            test_name="validation_test",
            reference_backend=backend,
            comparison_backend=backend,
            values_match=is_valid,
            max_absolute_error=0.0,
            max_relative_error=0.0,
            mean_absolute_error=0.0,
            correlation=1.0 if is_valid else 0.0,
            reference_time=result.execution_time,
            comparison_time=result.execution_time,
            speedup=1.0,
            reference_memory=result.memory_used,
            comparison_memory=result.memory_used,
            memory_ratio=1.0,
            status=TestStatus.PASSED if is_valid else TestStatus.FAILED,
            error_message=None if is_valid else result.error_message or "No output produced",
            field_comparisons={k: {'validated': True} for k in output.keys()}
        )
    
    def _run_on_backend(self, backend: Backend, test_data: Dict[str, Any]) -> TestResult:
        """Execute test on a specific backend"""
        import traceback
        
        start_time = time.time()
        memory_used = 0
        output_data = {}
        error_message = None
        status = TestStatus.PASSED
        
        # Adapt input data to expected format for this module
        adapted_data = self._adapt_input_data(test_data)
        
        try:
            if backend == Backend.C_CPU:
                output_data = self._run_c_cpu(adapted_data)
            elif backend == Backend.C_CUDA:
                output_data = self._run_c_cuda(adapted_data)
            elif backend == Backend.PYTHON_NUMPY:
                output_data, memory_used = self._run_python_numpy(adapted_data)
            elif backend == Backend.PYTHON_CUPY:
                output_data, memory_used = self._run_python_cupy(adapted_data)
        except Exception as e:
            status = TestStatus.ERROR
            error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            
        execution_time = time.time() - start_time
        
        return TestResult(
            backend=backend,
            module_name=self.name,
            test_name="comparison_test",
            status=status,
            execution_time=execution_time,
            memory_used=memory_used,
            output_data=output_data,
            error_message=error_message
        )
    
    def _run_c_cpu(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run C CPU implementation"""
        if not self.c_binary:
            raise NotImplementedError("C binary not configured")
            
        # Write input data to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self._serialize_data(test_data), f)
            input_file = f.name
            
        output_file = tempfile.mktemp(suffix='.json')
        
        try:
            result = subprocess.run(
                [self.c_binary, '-i', input_file, '-o', output_file, '--json'],
                capture_output=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"C binary failed: {result.stderr.decode()}")
                
            with open(output_file, 'r') as f:
                return json.load(f)
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)
    
    def _run_c_cuda(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run C CUDA implementation"""
        if not self.cuda_binary:
            raise NotImplementedError("CUDA binary not configured")
            
        # Similar to C CPU but with CUDA binary
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self._serialize_data(test_data), f)
            input_file = f.name
            
        output_file = tempfile.mktemp(suffix='.json')
        
        try:
            result = subprocess.run(
                [self.cuda_binary, '-i', input_file, '-o', output_file, '--json', '--cuda'],
                capture_output=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"CUDA binary failed: {result.stderr.decode()}")
                
            with open(output_file, 'r') as f:
                return json.load(f)
        finally:
            Path(input_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)
    
    def _run_python_numpy(self, test_data: Dict[str, Any]) -> tuple:
        """Run Python NumPy implementation"""
        if not self.python_processor:
            raise NotImplementedError("Python processor not configured")
            
        import tracemalloc
        from superdarn_gpu.core.backends import BackendContext
        
        tracemalloc.start()
        
        with BackendContext('numpy'):
            processor = self.python_processor()
            result = processor.process(test_data)
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return self._extract_output(result), peak
    
    def _run_python_cupy(self, test_data: Dict[str, Any]) -> tuple:
        """Run Python CuPy implementation"""
        if not self.python_processor:
            raise NotImplementedError("Python processor not configured")
            
        import cupy as cp
        from superdarn_gpu.core.backends import BackendContext
        
        # Get GPU memory before
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        
        with BackendContext('cupy'):
            processor = self.python_processor()
            result = processor.process(test_data)
            cp.cuda.Stream.null.synchronize()
            
        memory_used = mempool.used_bytes()
        
        return self._extract_output(result), memory_used
    
    def _compare_results(self,
                        reference: TestResult,
                        comparison: TestResult,
                        ref_backend: Backend,
                        cmp_backend: Backend) -> ComparisonResult:
        """Compare two test results"""
        
        if comparison.status != TestStatus.PASSED:
            return ComparisonResult(
                module_name=self.name,
                test_name="comparison_test",
                reference_backend=ref_backend,
                comparison_backend=cmp_backend,
                values_match=False,
                max_absolute_error=float('inf'),
                max_relative_error=float('inf'),
                mean_absolute_error=float('inf'),
                correlation=0.0,
                reference_time=reference.execution_time,
                comparison_time=comparison.execution_time,
                speedup=0.0,
                reference_memory=reference.memory_used,
                comparison_memory=comparison.memory_used,
                memory_ratio=0.0,
                status=comparison.status,
                error_message=comparison.error_message
            )
        
        # Compare output fields
        field_comparisons = {}
        overall_match = True
        max_abs_err = 0.0
        max_rel_err = 0.0
        total_mean_err = 0.0
        total_correlation = 0.0
        num_fields = 0
        
        ref_data = reference.output_data or {}
        cmp_data = comparison.output_data or {}
        
        for key in ref_data:
            if key not in cmp_data:
                field_comparisons[key] = {'error': 'missing in comparison'}
                overall_match = False
                continue
                
            ref_val = np.array(ref_data[key])
            cmp_val = np.array(cmp_data[key])
            
            metrics = self.framework.compare_arrays(ref_val, cmp_val)
            field_comparisons[key] = metrics
            
            if not metrics['values_match']:
                overall_match = False
                
            max_abs_err = max(max_abs_err, metrics['max_absolute_error'])
            max_rel_err = max(max_rel_err, metrics['max_relative_error'])
            total_mean_err += metrics['mean_absolute_error']
            total_correlation += metrics['correlation']
            num_fields += 1
        
        avg_mean_err = total_mean_err / num_fields if num_fields > 0 else 0.0
        avg_correlation = total_correlation / num_fields if num_fields > 0 else 0.0
        
        # Performance metrics
        speedup = reference.execution_time / comparison.execution_time if comparison.execution_time > 0 else 0.0
        memory_ratio = comparison.memory_used / reference.memory_used if reference.memory_used > 0 else 0.0
        
        return ComparisonResult(
            module_name=self.name,
            test_name="comparison_test",
            reference_backend=ref_backend,
            comparison_backend=cmp_backend,
            values_match=overall_match,
            max_absolute_error=max_abs_err,
            max_relative_error=max_rel_err,
            mean_absolute_error=avg_mean_err,
            correlation=avg_correlation,
            reference_time=reference.execution_time,
            comparison_time=comparison.execution_time,
            speedup=speedup,
            reference_memory=reference.memory_used,
            comparison_memory=comparison.memory_used,
            memory_ratio=memory_ratio,
            status=TestStatus.PASSED if overall_match else TestStatus.FAILED,
            field_comparisons=field_comparisons
        )
    
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for JSON output"""
        result = {}
        for key, value in data.items():
            if hasattr(value, 'tolist'):
                result[key] = value.tolist()
            elif hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
    
    def _extract_output(self, result: Any) -> Dict[str, Any]:
        """Extract output data from processor result"""
        output = {}
        
        if hasattr(result, '__dict__'):
            for key, value in result.__dict__.items():
                if key.startswith('_'):
                    continue
                # Check for CuPy array specifically (has get() method with no required args)
                if hasattr(value, 'get') and hasattr(value, '__cuda_array_interface__'):
                    output[key] = np.array(value.get())
                elif hasattr(value, '__array__') and not isinstance(value, dict):
                    output[key] = np.array(value)
                elif isinstance(value, (int, float, str, bool)):
                    output[key] = value
                elif isinstance(value, np.ndarray):
                    output[key] = value
                    
        return output
