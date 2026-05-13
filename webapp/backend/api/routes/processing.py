"""
Data processing endpoints — backed by SQLite via services/db.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
import uuid
import logging
from datetime import datetime

from models.schemas import (
    ProcessingRequest, JobInfo, JobStatus, ProcessingMode, ProcessingStage
)
from services.processor import process_data_async, probe_backends
import services.db as db

logger = logging.getLogger(__name__)
router = APIRouter()


def _row_to_jobinfo(row: dict) -> JobInfo:
    """Convert a DB row dict to a JobInfo Pydantic model."""
    from models.schemas import FitACFParameters
    params = row.get("parameters") or {}
    if isinstance(params, dict):
        try:
            params = FitACFParameters(**params)
        except Exception:
            params = FitACFParameters()

    def _dt(s):
        if not s or s == "None":
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    stage = row.get("current_stage")
    try:
        stage = ProcessingStage(stage) if stage else None
    except Exception:
        stage = None

    return JobInfo(
        job_id=row["job_id"],
        status=JobStatus(row.get("status", "queued")),
        progress=row.get("progress", 0),
        current_stage=stage,
        created_at=_dt(row.get("created_at")) or datetime.now(),
        started_at=_dt(row.get("started_at")),
        completed_at=_dt(row.get("completed_at")),
        error=row.get("error"),
        mode=ProcessingMode(row.get("mode", "auto")),
        parameters=params,
    )


@router.post("/start", response_model=JobInfo)
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start a processing job and return its info."""
    try:
        job_id  = str(uuid.uuid4())
        now     = datetime.now()
        job_row = {
            "job_id":     job_id,
            "status":     "queued",
            "progress":   0,
            "created_at": str(now),
            "mode":       request.mode.value,
            "parameters": request.parameters.dict(),
            "backend":    request.backend,
        }
        db.upsert_job(job_row)

        background_tasks.add_task(
            process_data_async,
            job_id=job_id,
            file_id=request.file_id,
            mode=request.mode,
            parameters=request.parameters,
            stages=request.stages,
            backend_override=request.backend,
        )

        logger.info(f"Job {job_id} queued for file {request.file_id}")
        return _row_to_jobinfo(job_row)

    except Exception as e:
        logger.error(f"Failed to start processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str):
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return _row_to_jobinfo(row)


@router.get("/list")
async def list_jobs():
    """List all jobs with result summaries from SQLite."""
    rows    = db.list_jobs()
    results = []
    for row in rows:
        entry = dict(row)
        entry["created_at"]   = str(row.get("created_at", ""))
        entry["completed_at"] = str(row.get("completed_at", "")) if row.get("completed_at") else None
        summary = db.get_result_with_summary(row["job_id"])
        entry["summary"] = summary
        results.append(entry)
    return {"jobs": results, "total": len(results)}


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail="Job already finished")
    db.upsert_job({**row, "status": "cancelled", "completed_at": str(datetime.now())})
    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/backends")
async def list_backends():
    """Return availability and status of all algorithm backends."""
    return {"backends": probe_backends()}


