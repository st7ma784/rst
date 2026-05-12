"""
Results retrieval endpoints — backed by SQLite via services/db.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pathlib import Path
import logging, csv, io

import services.db as db

logger = logging.getLogger(__name__)
router = APIRouter()

RESULTS_DIR = Path("/tmp/siw_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/{job_id}")
async def get_results(job_id: str):
    """Get processing results for a completed job."""
    result = db.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    return result


@router.get("/{job_id}/visualization")
async def get_visualization_data(job_id: str):
    """Return data shaped for the frontend visualization components."""
    result = db.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")

    viz_data = {
        "job_id":      job_id,
        "stages":      result["stages"],
        "performance": result["performance_metrics"],
        "plots":       {},
    }

    if "fitacf" in result["stages"]:
        fa = result["stages"]["fitacf"]
        viz_data["plots"]["range_profile"] = {k: fa.get(k, []) for k in (
            "velocity", "velocity_error",
            "power", "power_error",
            "spectral_width", "spectral_width_error",
            "spectral_width_sigma", "spectral_width_sigma_error",
            "elevation", "elevation_error",
            "quality_flag", "ground_scatter_flag", "nlag_fit",
        )} | {
            "nranges":     fa.get("nranges", 0),
            "good_ranges": fa.get("good_ranges", 0),
            "backend":     fa.get("backend", ""),
        }

        # Multi-record RTI + per-beam range data
        if "records" in fa:
            viz_data["plots"]["rti"] = {
                "records":  fa["records"],
                "nranges":  fa.get("nranges", 0),
                "nrecords": fa.get("nrecords", 1),
            }
            # Expose per-beam range arrays for the beam selector
            viz_data["plots"]["range_profile"]["records"] = fa["records"]
            viz_data["plots"]["range_profile"]["nrecords"] = fa.get("nrecords", 1)

    if "grid" in result["stages"]:
        gd = result["stages"]["grid"]
        viz_data["plots"]["grid"] = {
            "velocity": gd.get("velocity", []),
            "nlat":     gd.get("nlat", 0),
            "nlon":     gd.get("nlon", 0),
        }

    return viz_data


@router.get("/{job_id}/export/csv")
async def export_csv(job_id: str):
    """Download fitacf results as a CSV file."""
    result = db.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")

    fa = result["stages"].get("fitacf", {})
    nranges = fa.get("nranges", 0)

    fields = [
        ("range_gate",             list(range(nranges))),
        ("velocity",               fa.get("velocity", [])),
        ("velocity_error",         fa.get("velocity_error", [])),
        ("power",                  fa.get("power", [])),
        ("power_error",            fa.get("power_error", [])),
        ("spectral_width",         fa.get("spectral_width", [])),
        ("spectral_width_error",   fa.get("spectral_width_error", [])),
        ("spectral_width_sigma",   fa.get("spectral_width_sigma", [])),
        ("elevation",              fa.get("elevation", [])),
        ("quality_flag",           fa.get("quality_flag", [])),
        ("ground_scatter_flag",    fa.get("ground_scatter_flag", [])),
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([f[0] for f in fields])
    for i in range(nranges):
        writer.writerow([
            col[i] if i < len(col) else ""
            for _, col in fields
        ])

    filename = f"{job_id[:8]}_{fa.get('backend','fitacf')}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete("/{job_id}")
async def delete_results(job_id: str):
    """Delete job, results, and the uploaded source file."""
    # Delete uploaded file
    uploads = Path("/tmp/siw_uploads")
    for p in uploads.glob(f"{job_id[:8]}*"):
        try:
            p.unlink()
        except Exception:
            pass

    # Delete result files on disk
    job_dir = RESULTS_DIR / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)

    db.delete_job(job_id)
    return {"message": "Deleted", "job_id": job_id}


@router.get("/{job_id}/download/{filename}")
async def download_result_file(job_id: str, filename: str):
    file_path = RESULTS_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename,
                        media_type="application/octet-stream")
