"""
Test API routes for module comparison testing

Provides endpoints for running and viewing side-by-side comparison tests
between C/CUDA and Python implementations.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import uuid
import asyncio
import logging
import sys
from pathlib import Path

# Add pythonv2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pythonv2"))

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage for test results
test_results: Dict[str, 'TestRunResult'] = {}
test_runs: Dict[str, 'TestRun'] = {}


class TestStatus(str, Enum):
    """Test status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BackendType(str, Enum):
    """Available backend types"""
    C_CPU = "c_cpu"
    C_CUDA = "c_cuda"
    PYTHON_NUMPY = "python_numpy"
    PYTHON_CUPY = "python_cupy"


class ModuleType(str, Enum):
    """Available module types for testing"""
    ACF = "acf"
    FITACF = "fitacf"
    GRID = "grid"
    CONVMAP = "convmap"
    ALL = "all"


class TestRequest(BaseModel):
    """Request to run module comparison tests"""
    modules: List[ModuleType] = Field(default=[ModuleType.ALL])
    backends: List[BackendType] = Field(default=[
        BackendType.PYTHON_NUMPY,
        BackendType.PYTHON_CUPY
    ])
    data_size: str = Field(default="medium", pattern="^(small|medium|large)$")
    include_performance: bool = Field(default=True)
    include_accuracy: bool = Field(default=True)


class ModuleResult(BaseModel):
    """Result for a single module comparison"""
    module_name: str
    module_version: str
    status: TestStatus
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    
    # Performance metrics
    avg_speedup: Optional[float] = None
    min_speedup: Optional[float] = None
    max_speedup: Optional[float] = None
    
    # Accuracy metrics
    avg_error: Optional[float] = None
    max_error: Optional[float] = None
    correlation: Optional[float] = None
    
    # Timing
    numpy_time_ms: Optional[float] = None
    cupy_time_ms: Optional[float] = None
    c_cpu_time_ms: Optional[float] = None
    c_cuda_time_ms: Optional[float] = None
    
    # Detailed comparisons
    comparisons: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Errors
    error_message: Optional[str] = None


class TestRun(BaseModel):
    """Test run information"""
    run_id: str
    status: TestStatus
    progress: int = 0
    total_modules: int = 0
    completed_modules: int = 0
    created_at: datetime
    completed_at: Optional[datetime] = None
    request: TestRequest
    results: Dict[str, ModuleResult] = Field(default_factory=dict)


class TestSummary(BaseModel):
    """Summary of all test results"""
    total_runs: int
    last_run_id: Optional[str] = None
    last_run_time: Optional[datetime] = None
    modules_available: List[str]
    backends_available: List[str]
    overall_pass_rate: Optional[float] = None


# Module descriptions
MODULE_INFO = {
    ModuleType.ACF: {
        "name": "ACF",
        "full_name": "Auto-Correlation Function",
        "version": "1.16",
        "description": "Calculates auto-correlation functions from raw I/Q samples"
    },
    ModuleType.FITACF: {
        "name": "FITACF",
        "full_name": "Fitted ACF",
        "version": "3.0",
        "description": "Fits Lorentzian curves to ACF data to extract velocity/width"
    },
    ModuleType.GRID: {
        "name": "Grid",
        "full_name": "Spatial Gridding",
        "version": "1.24",
        "description": "Interpolates scattered measurements onto regular grids"
    },
    ModuleType.CONVMAP: {
        "name": "ConvMap",
        "full_name": "Convection Mapping",
        "version": "1.17",
        "description": "Generates global convection maps using spherical harmonic fitting"
    }
}


@router.get("/summary", response_model=TestSummary)
async def get_test_summary():
    """Get summary of test capabilities and recent results"""
    last_run = None
    last_time = None
    overall_pass_rate = None
    
    if test_runs:
        # Find most recent completed run
        completed_runs = [r for r in test_runs.values() if r.status == TestStatus.COMPLETED]
        if completed_runs:
            last_run_obj = max(completed_runs, key=lambda x: x.created_at)
            last_run = last_run_obj.run_id
            last_time = last_run_obj.completed_at
            
            # Calculate pass rate
            total_tests = sum(r.total_tests for r in last_run_obj.results.values())
            passed_tests = sum(r.passed_tests for r in last_run_obj.results.values())
            if total_tests > 0:
                overall_pass_rate = passed_tests / total_tests
    
    # Check available backends
    backends = [BackendType.PYTHON_NUMPY.value]
    try:
        import cupy
        if cupy.cuda.is_available():
            backends.append(BackendType.PYTHON_CUPY.value)
    except ImportError:
        pass
    
    return TestSummary(
        total_runs=len(test_runs),
        last_run_id=last_run,
        last_run_time=last_time,
        modules_available=[m.value for m in ModuleType if m != ModuleType.ALL],
        backends_available=backends,
        overall_pass_rate=overall_pass_rate
    )


@router.get("/modules")
async def get_modules():
    """Get information about available modules"""
    return {
        "modules": [
            {
                "id": m.value,
                **MODULE_INFO[m]
            }
            for m in ModuleType if m != ModuleType.ALL
        ]
    }


@router.post("/run", response_model=TestRun)
async def start_test_run(request: TestRequest, background_tasks: BackgroundTasks):
    """
    Start a new test comparison run
    
    Runs side-by-side comparisons between selected backends for selected modules.
    """
    run_id = str(uuid.uuid4())
    
    # Determine modules to test
    modules = request.modules
    if ModuleType.ALL in modules:
        modules = [m for m in ModuleType if m != ModuleType.ALL]
    
    # Create run record
    test_run = TestRun(
        run_id=run_id,
        status=TestStatus.PENDING,
        progress=0,
        total_modules=len(modules),
        completed_modules=0,
        created_at=datetime.now(),
        request=request
    )
    
    test_runs[run_id] = test_run
    
    # Start async test execution
    background_tasks.add_task(
        execute_test_run,
        run_id=run_id,
        modules=modules,
        backends=request.backends,
        data_size=request.data_size
    )
    
    logger.info(f"Started test run {run_id} for modules: {[m.value for m in modules]}")
    
    return test_run


@router.get("/run/{run_id}", response_model=TestRun)
async def get_test_run(run_id: str):
    """Get status and results of a test run"""
    if run_id not in test_runs:
        raise HTTPException(status_code=404, detail="Test run not found")
    
    return test_runs[run_id]


@router.get("/runs")
async def list_test_runs(limit: int = 10):
    """List recent test runs"""
    runs = sorted(test_runs.values(), key=lambda x: x.created_at, reverse=True)
    return {
        "runs": runs[:limit],
        "total": len(runs)
    }


@router.get("/run/{run_id}/module/{module_name}", response_model=ModuleResult)
async def get_module_result(run_id: str, module_name: str):
    """Get detailed results for a specific module in a test run"""
    if run_id not in test_runs:
        raise HTTPException(status_code=404, detail="Test run not found")
    
    run = test_runs[run_id]
    if module_name not in run.results:
        raise HTTPException(status_code=404, detail="Module not found in this run")
    
    return run.results[module_name]


@router.delete("/run/{run_id}")
async def delete_test_run(run_id: str):
    """Delete a test run record"""
    if run_id not in test_runs:
        raise HTTPException(status_code=404, detail="Test run not found")
    
    del test_runs[run_id]
    return {"message": "Test run deleted", "run_id": run_id}


async def execute_test_run(run_id: str, 
                          modules: List[ModuleType],
                          backends: List[BackendType],
                          data_size: str):
    """Execute test run in background"""
    run = test_runs[run_id]
    run.status = TestStatus.RUNNING
    
    try:
        from tests.comparison.fixtures import generate_test_data
        from superdarn_gpu.processing.acf import ACFProcessor
        from superdarn_gpu.processing.fitacf import FitACFProcessor
        from superdarn_gpu.processing.grid import GridProcessor
        from superdarn_gpu.processing.convmap import ConvMapProcessor
        from superdarn_gpu.core.backends import BackendContext
        import time
        import numpy as np
        
        processor_map = {
            ModuleType.ACF: ACFProcessor,
            ModuleType.FITACF: FitACFProcessor,
            ModuleType.GRID: GridProcessor,
            ModuleType.CONVMAP: ConvMapProcessor
        }
        
        for i, module in enumerate(modules):
            module_result = ModuleResult(
                module_name=module.value,
                module_version=MODULE_INFO[module]["version"],
                status=TestStatus.RUNNING
            )
            
            try:
                # Generate test data
                test_data = generate_test_data(module.value, size=data_size)
                
                results_by_backend = {}
                
                # Run on each backend
                for backend in backends:
                    backend_name = 'numpy' if backend in [BackendType.PYTHON_NUMPY, BackendType.C_CPU] else 'cupy'
                    
                    try:
                        with BackendContext(backend_name):
                            processor_class = processor_map.get(module)
                            if processor_class:
                                processor = processor_class()
                                
                                start_time = time.time()
                                
                                if module == ModuleType.CONVMAP:
                                    grid_data = test_data.get('grid_data', test_data)
                                    result = processor.process(
                                        grid_data,
                                        imf_by=test_data.get('imf_by', 0),
                                        imf_bz=test_data.get('imf_bz', -5)
                                    )
                                else:
                                    result = processor.process(test_data)
                                
                                elapsed = time.time() - start_time
                                
                                results_by_backend[backend.value] = {
                                    'time_ms': elapsed * 1000,
                                    'result': result,
                                    'success': True
                                }
                                
                                # Store timing
                                if backend == BackendType.PYTHON_NUMPY:
                                    module_result.numpy_time_ms = elapsed * 1000
                                elif backend == BackendType.PYTHON_CUPY:
                                    module_result.cupy_time_ms = elapsed * 1000
                                    
                    except ImportError as e:
                        results_by_backend[backend.value] = {
                            'success': False,
                            'error': f"Backend not available: {str(e)}"
                        }
                    except Exception as e:
                        results_by_backend[backend.value] = {
                            'success': False,
                            'error': str(e)
                        }
                
                # Compare results
                comparisons = []
                if BackendType.PYTHON_NUMPY.value in results_by_backend and \
                   BackendType.PYTHON_CUPY.value in results_by_backend:
                    
                    np_res = results_by_backend[BackendType.PYTHON_NUMPY.value]
                    cp_res = results_by_backend[BackendType.PYTHON_CUPY.value]
                    
                    if np_res.get('success') and cp_res.get('success'):
                        # Calculate speedup
                        speedup = np_res['time_ms'] / cp_res['time_ms'] if cp_res['time_ms'] > 0 else 0
                        module_result.avg_speedup = speedup
                        module_result.min_speedup = speedup
                        module_result.max_speedup = speedup
                        
                        # Compare outputs
                        comparison = {
                            'reference': BackendType.PYTHON_NUMPY.value,
                            'comparison': BackendType.PYTHON_CUPY.value,
                            'speedup': speedup,
                            'values_match': True,
                            'error_metrics': {}
                        }
                        
                        # Try to compare numerical outputs
                        try:
                            np_result = np_res['result']
                            cp_result = cp_res['result']
                            
                            # Get relevant output arrays based on module
                            if hasattr(np_result, 'acf') and hasattr(cp_result, 'acf'):
                                np_arr = np.array(np_result.acf)
                                cp_arr = cp_result.acf
                                if hasattr(cp_arr, 'get'):
                                    cp_arr = cp_arr.get()
                                
                                diff = np.abs(np_arr - cp_arr)
                                comparison['error_metrics'] = {
                                    'max_error': float(np.max(diff)),
                                    'mean_error': float(np.mean(diff)),
                                    'values_match': bool(np.allclose(np_arr, cp_arr, rtol=1e-4))
                                }
                                comparison['values_match'] = comparison['error_metrics']['values_match']
                                module_result.avg_error = comparison['error_metrics']['mean_error']
                                module_result.max_error = comparison['error_metrics']['max_error']
                            
                        except Exception as e:
                            comparison['comparison_error'] = str(e)
                        
                        comparisons.append(comparison)
                        
                        module_result.total_tests = 1
                        module_result.passed_tests = 1 if comparison['values_match'] else 0
                        module_result.failed_tests = 0 if comparison['values_match'] else 1
                
                module_result.comparisons = comparisons
                module_result.status = TestStatus.COMPLETED
                
            except Exception as e:
                module_result.status = TestStatus.FAILED
                module_result.error_message = str(e)
                logger.error(f"Error testing module {module.value}: {e}")
            
            # Update run progress
            run.results[module.value] = module_result
            run.completed_modules = i + 1
            run.progress = int((i + 1) / len(modules) * 100)
        
        run.status = TestStatus.COMPLETED
        run.completed_at = datetime.now()
        
    except Exception as e:
        run.status = TestStatus.FAILED
        logger.error(f"Test run {run_id} failed: {e}")


@router.get("/latest-report")
async def get_latest_report():
    """Get the latest test report in a format suitable for the dashboard"""
    if not test_runs:
        return {"message": "No test runs available"}
    
    # Find latest completed run
    completed_runs = [r for r in test_runs.values() if r.status == TestStatus.COMPLETED]
    if not completed_runs:
        return {"message": "No completed test runs"}
    
    latest_run = max(completed_runs, key=lambda x: x.created_at)
    
    # Format for dashboard
    report = {
        "run_id": latest_run.run_id,
        "timestamp": latest_run.completed_at.isoformat() if latest_run.completed_at else None,
        "summary": {
            "total_modules": latest_run.total_modules,
            "passed": sum(1 for r in latest_run.results.values() if r.passed_tests == r.total_tests),
            "failed": sum(1 for r in latest_run.results.values() if r.failed_tests > 0),
            "avg_speedup": 0
        },
        "modules": []
    }
    
    speedups = []
    for module_name, result in latest_run.results.items():
        module_data = {
            "name": module_name,
            "version": result.module_version,
            "status": "passed" if result.passed_tests == result.total_tests else "failed",
            "numpy_time_ms": result.numpy_time_ms,
            "cupy_time_ms": result.cupy_time_ms,
            "speedup": result.avg_speedup,
            "avg_error": result.avg_error,
            "max_error": result.max_error
        }
        report["modules"].append(module_data)
        
        if result.avg_speedup:
            speedups.append(result.avg_speedup)
    
    if speedups:
        report["summary"]["avg_speedup"] = sum(speedups) / len(speedups)
    
    return report
