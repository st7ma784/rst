"""
Data processing service
Main processing logic that integrates with CUDArst and SuperDARN GPU
"""

import logging
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys

# Add pythonv2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "pythonv2"))

from models.schemas import (
    JobStatus, ProcessingMode, ProcessingStage, FitACFParameters,
    JobInfo, ProcessingResult
)

logger = logging.getLogger(__name__)

async def process_data_async(
    job_id: str,
    file_id: str,
    mode: ProcessingMode,
    parameters: FitACFParameters,
    stages: List[ProcessingStage],
    jobs: Dict[str, JobInfo]
):
    """
    Asynchronous data processing function
    
    This function runs in the background and processes SuperDARN data
    through the requested pipeline stages.
    """
    job = jobs[job_id]
    
    try:
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        job.progress = 5
        
        logger.info(f"Starting processing job {job_id} with mode {mode}")
        
        # Load input file
        upload_dir = Path("/tmp/siw_uploads")
        file_path = list(upload_dir.glob(f"{file_id}_*"))[0]
        
        job.progress = 10
        job.current_stage = ProcessingStage.UPLOAD
        
        # Initialize results storage
        results = {
            "stages": {},
            "performance_metrics": {},
            "timing": {}
        }
        
        start_time = time.time()
        
        # Determine processing backend
        use_gpu = False
        if mode == ProcessingMode.CUDA:
            use_gpu = True
        elif mode == ProcessingMode.AUTO:
            try:
                import cupy as cp
                use_gpu = True
                logger.info("GPU available - using CUDA acceleration")
            except:
                logger.info("GPU not available - using CPU processing")
        
        # Process through each stage
        for stage in stages:
            job.current_stage = stage
            stage_start = time.time()
            
            logger.info(f"Processing stage: {stage}")
            
            if stage == ProcessingStage.ACF:
                job.progress = 20
                results["stages"]["acf"] = await process_acf_stage(file_path, parameters, use_gpu)
            
            elif stage == ProcessingStage.FITACF:
                job.progress = 40
                results["stages"]["fitacf"] = await process_fitacf_stage(file_path, parameters, use_gpu)
            
            elif stage == ProcessingStage.LMFIT:
                job.progress = 60
                results["stages"]["lmfit"] = await process_lmfit_stage(results["stages"], parameters, use_gpu)
            
            elif stage == ProcessingStage.GRID:
                job.progress = 80
                results["stages"]["grid"] = await process_grid_stage(results["stages"], parameters, use_gpu)
            
            elif stage == ProcessingStage.CNVMAP:
                job.progress = 90
                results["stages"]["cnvmap"] = await process_cnvmap_stage(results["stages"], parameters, use_gpu)
            
            stage_time = time.time() - stage_start
            results["timing"][stage.value] = stage_time
            logger.info(f"Stage {stage} completed in {stage_time:.3f}s")
        
        # Calculate total time
        total_time = time.time() - start_time
        results["performance_metrics"]["total_time"] = total_time
        results["performance_metrics"]["mode"] = "GPU" if use_gpu else "CPU"
        
        # Mark job as complete
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.now()
        job.current_stage = ProcessingStage.COMPLETE
        
        # Store results (in production, save to database/file storage)
        from api.routes.results import results as results_storage
        results_storage[job_id] = ProcessingResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            processing_time=total_time,
            stages=results["stages"],
            performance_metrics=results["performance_metrics"]
        )
        
        logger.info(f"Job {job_id} completed successfully in {total_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()

async def process_acf_stage(file_path: Path, params: FitACFParameters, use_gpu: bool) -> Dict:
    """Process ACF stage"""
    logger.info(f"ACF processing - GPU: {use_gpu}")
    
    # Simulate processing (replace with actual superdarn_gpu calls)
    await simulate_processing(0.5)
    
    return {
        "nranges": 75,
        "nlags": 18,
        "processing_time": 0.5,
        "backend": "GPU" if use_gpu else "CPU"
    }

async def process_fitacf_stage(file_path: Path, params: FitACFParameters, use_gpu: bool) -> Dict:
    """Process FITACF stage"""
    logger.info(f"FITACF v3.0 processing - GPU: {use_gpu}, min_power: {params.min_power}")
    
    # Simulate processing
    await simulate_processing(1.0)
    
    return {
        "nranges": 75,
        "good_ranges": 45,
        "velocity_range": [-500, 500],
        "power_range": [0, 40],
        "processing_time": 1.0,
        "parameters": {
            "min_power": params.min_power,
            "phase_tolerance": params.phase_tolerance,
            "elevation_enabled": params.elevation_enabled
        },
        "backend": "GPU" if use_gpu else "CPU"
    }

async def process_lmfit_stage(previous_stages: Dict, params: FitACFParameters, use_gpu: bool) -> Dict:
    """Process LMFIT stage"""
    logger.info(f"LMFIT processing - GPU: {use_gpu}")
    
    await simulate_processing(0.8)
    
    return {
        "iterations": 15,
        "converged": True,
        "residual": 0.025,
        "processing_time": 0.8,
        "backend": "GPU" if use_gpu else "CPU"
    }

async def process_grid_stage(previous_stages: Dict, params: FitACFParameters, use_gpu: bool) -> Dict:
    """Process Grid stage"""
    logger.info(f"Grid processing - GPU: {use_gpu}")
    
    await simulate_processing(0.6)
    
    return {
        "grid_resolution": 1.0,
        "npoints": 2500,
        "interpolation": "cubic",
        "processing_time": 0.6,
        "backend": "GPU" if use_gpu else "CPU"
    }

async def process_cnvmap_stage(previous_stages: Dict, params: FitACFParameters, use_gpu: bool) -> Dict:
    """Process Convection Map stage"""
    logger.info(f"CNVMAP processing - GPU: {use_gpu}")
    
    await simulate_processing(0.7)
    
    return {
        "order": 8,
        "chi_squared": 1.25,
        "potential_max": 80.0,
        "processing_time": 0.7,
        "backend": "GPU" if use_gpu else "CPU"
    }

async def simulate_processing(duration: float):
    """Simulate processing delay"""
    import asyncio
    await asyncio.sleep(duration)
