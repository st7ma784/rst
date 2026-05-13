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


def _normalize_fitacf(fa: dict) -> dict:
    """
    When records[] is present, reconstruct first-record top-level scalars
    so consumers don't need to know about the records structure.
    """
    if not fa.get("records"):
        return fa
    first = fa["records"][0]
    merged = dict(fa)
    for key in ("velocity", "velocity_error", "power", "power_error",
                "spectral_width", "spectral_width_error",
                "quality_flag", "ground_scatter_flag", "nlag_fit"):
        if key not in merged or not merged[key]:
            merged[key] = first.get(key, [])
    return merged


@router.get("/{job_id}")
async def get_results(job_id: str):
    """Get processing results for a completed job."""
    result = db.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    # Normalise fitacf so consumers get top-level arrays even when records[] is the source
    if "fitacf" in result.get("stages", {}):
        result["stages"]["fitacf"] = _normalize_fitacf(result["stages"]["fitacf"])
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
        records = fa.get("records", [])

        # When multi-record data is present, derive first-record scalars from
        # records[0] rather than storing duplicate top-level arrays.
        first = records[0] if records else fa

        rp = {k: first.get(k, []) for k in (
            "velocity", "velocity_error",
            "power", "power_error",
            "spectral_width", "spectral_width_error",
            "quality_flag", "ground_scatter_flag", "nlag_fit",
        )}
        # Fields only in top-level (not per-beam)
        rp["spectral_width_sigma"]       = fa.get("spectral_width_sigma", [])
        rp["spectral_width_sigma_error"] = fa.get("spectral_width_sigma_error", [])
        rp["elevation"]                  = fa.get("elevation", [])
        rp["elevation_error"]            = fa.get("elevation_error", [])
        rp["nranges"]                    = fa.get("nranges", 0)
        rp["good_ranges"]                = first.get("good_ranges", fa.get("good_ranges", 0))
        rp["backend"]                    = fa.get("backend", "")
        if records:
            rp["records"]  = records
            rp["nrecords"] = fa.get("nrecords", len(records))

        viz_data["plots"]["range_profile"] = rp

        if records:
            viz_data["plots"]["rti"] = {
                "records":  records,
                "nranges":  fa.get("nranges", 0),
                "nrecords": fa.get("nrecords", len(records)),
            }

    if "grid" in result["stages"]:
        gd = result["stages"]["grid"]
        viz_data["plots"]["grid"] = {
            "velocity": gd.get("velocity", []),
            "nlat":     gd.get("nlat", 0),
            "nlon":     gd.get("nlon", 0),
        }

    return viz_data


@router.get("/{job_id}/export/csv")
async def export_csv(job_id: str, stage: str = "fitacf"):
    """Download stage results as a CSV file. stage = fitacf | grid | cnvmap."""
    result = db.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")

    if stage == "grid":
        gd = result["stages"].get("grid", {})
        vel = gd.get("velocity", [])
        nlat, nlon = gd.get("nlat", 0), gd.get("nlon", 0)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["cell_index", "lat_index", "lon_index", "velocity"])
        for i, v in enumerate(vel):
            writer.writerow([i, i // max(nlon, 1), i % max(nlon, 1), v])
        return Response(content=buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={job_id[:8]}_grid.csv"})

    if stage == "cnvmap":
        cm = result["stages"].get("cnvmap", {})
        pot = cm.get("potential", [])
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["lat_index", "lon_index", "potential_V", "velocity_mag"])
        vmag = cm.get("velocity_mag", [])
        for yi, row in enumerate(pot):
            for xi, v in enumerate(row):
                vm = vmag[yi][xi] if yi < len(vmag) and xi < len(vmag[yi]) else None
                writer.writerow([yi, xi, v, vm])
        return Response(content=buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={job_id[:8]}_cnvmap.csv"})

    # Default: fitacf
    fa = result["stages"].get("fitacf", {})
    # Derive first-record data from records if present
    records = fa.get("records", [])
    first = records[0] if records else fa
    nranges = fa.get("nranges", 0)

    fields = [
        ("range_gate",             list(range(nranges))),
        ("velocity",               first.get("velocity", [])),
        ("velocity_error",         first.get("velocity_error", [])),
        ("power",                  first.get("power", [])),
        ("power_error",            first.get("power_error", [])),
        ("spectral_width",         first.get("spectral_width", [])),
        ("spectral_width_error",   first.get("spectral_width_error", [])),
        ("spectral_width_sigma",   fa.get("spectral_width_sigma", [])),
        ("elevation",              fa.get("elevation", [])),
        ("quality_flag",           first.get("quality_flag", [])),
        ("ground_scatter_flag",    first.get("ground_scatter_flag", [])),
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


@router.get("/{job_id}/cnvmap")
async def get_cnvmap_data(job_id: str):
    """Return convection map data (potential grid + stats) for the frontend."""
    result = db.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    cm = result["stages"].get("cnvmap", {})
    if not cm:
        raise HTTPException(status_code=404, detail="cnvmap stage not run")
    # Potential/velocity arrays are stored directly when pythonv2 ran cnvmap
    return cm


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
