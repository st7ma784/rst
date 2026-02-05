"""
Results retrieval endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Dict
import logging
from pathlib import Path

from models.schemas import ProcessingResult

logger = logging.getLogger(__name__)
router = APIRouter()

# Results storage directory
RESULTS_DIR = Path("/tmp/siw_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory results storage (use database in production)
results: Dict[str, ProcessingResult] = {}

@router.get("/{job_id}", response_model=ProcessingResult)
async def get_results(job_id: str):
    """Get processing results for a completed job"""
    if job_id not in results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return results[job_id]

@router.get("/{job_id}/download/{filename}")
async def download_result_file(job_id: str, filename: str):
    """Download a specific result file"""
    file_path = RESULTS_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@router.get("/{job_id}/visualization")
async def get_visualization_data(job_id: str):
    """Get processed data formatted for visualization"""
    if job_id not in results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    result = results[job_id]
    
    # Extract visualization-ready data
    viz_data = {
        "job_id": job_id,
        "stages": result.stages,
        "performance": result.performance_metrics,
        "plots": {}
    }
    
    # Add plot data if available
    if "fitacf" in result.stages:
        fitacf_data = result.stages["fitacf"]
        viz_data["plots"]["range_time"] = {
            "velocity": fitacf_data.get("velocity", []),
            "power": fitacf_data.get("power", []),
            "width": fitacf_data.get("width", [])
        }
    
    if "grid" in result.stages:
        grid_data = result.stages["grid"]
        viz_data["plots"]["grid"] = {
            "lat": grid_data.get("lat", []),
            "lon": grid_data.get("lon", []),
            "velocity": grid_data.get("velocity", [])
        }
    
    return viz_data

@router.delete("/{job_id}")
async def delete_results(job_id: str):
    """Delete results for a job"""
    if job_id not in results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Remove from storage
    del results[job_id]
    
    # Delete files if they exist
    job_dir = RESULTS_DIR / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)
    
    logger.info(f"Results deleted: {job_id}")
    
    return {"message": "Results deleted successfully", "job_id": job_id}
