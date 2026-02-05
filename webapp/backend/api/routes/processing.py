"""
Data processing endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
import uuid
import logging
from datetime import datetime

from models.schemas import (
    ProcessingRequest, JobInfo, JobStatus, ProcessingMode, ProcessingStage
)
from services.processor import process_data_async

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job storage (use Redis/database in production)
jobs: Dict[str, JobInfo] = {}

@router.post("/start", response_model=JobInfo)
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """
    Start data processing job
    
    Creates a new processing job and starts it in the background.
    Returns job information including job_id for status tracking.
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job info
        job = JobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            progress=0,
            created_at=datetime.now(),
            mode=request.mode,
            parameters=request.parameters
        )
        
        # Store job
        jobs[job_id] = job
        
        # Start processing in background
        background_tasks.add_task(
            process_data_async,
            job_id=job_id,
            file_id=request.file_id,
            mode=request.mode,
            parameters=request.parameters,
            stages=request.stages,
            jobs=jobs
        )
        
        logger.info(f"Processing job started: {job_id} for file {request.file_id}")
        
        return job
        
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/status/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str):
    """Get current status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@router.get("/list")
async def list_jobs():
    """List all processing jobs"""
    return {
        "jobs": list(jobs.values()),
        "total": len(jobs)
    }

@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail="Job already finished")
    
    # Update job status
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now()
    
    logger.info(f"Job cancelled: {job_id}")
    
    return {"message": "Job cancelled", "job_id": job_id}

@router.post("/compare")
async def compare_parameters(request: Dict):
    """
    Compare different parameter settings
    
    Runs processing with multiple parameter configurations
    and returns comparison results
    """
    # TODO: Implement parameter comparison
    return {
        "message": "Parameter comparison not yet implemented",
        "comparison_id": str(uuid.uuid4())
    }
