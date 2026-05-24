"""
Processing dispatcher.

Since the pythonv2 + CUDArst backends were retired (the C library
optimization track delivers via libgrdopt + libfitacf.3.0, used by
make_fit / make_grid / map_grd), there is now exactly one path:
processor_rst.RSTProcessor. The BACKEND_TYPE env var and
backend_override request param are accepted but ignored — kept so
the existing API surface (probe_backends, request schemas) is
unchanged for callers.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from models.schemas import (
    JobStatus, ProcessingMode, ProcessingStage, FitACFParameters,
    JobInfo, ProcessingResult
)

logger = logging.getLogger(__name__)


def _get_backend(override: str = None):
    """Return the single RST-backed processor. `override` is ignored."""
    from services.processor_rst import RSTProcessor
    return RSTProcessor()


def probe_backends() -> list:
    """
    Return availability info. Always exactly one entry now.
    Shape preserved (id/name/available/gpu/active) so the frontend
    doesn't need to change.
    """
    try:
        from services.processor_rst import RSTProcessor  # noqa
        return [{
            "id": "rst", "name": "RST (C / libgrdopt + libfitacf.3.0)",
            "available": True, "gpu": False, "active": True,
        }]
    except Exception as exc:
        return [{
            "id": "rst", "name": "RST (C / libgrdopt + libfitacf.3.0)",
            "available": False, "gpu": False, "active": True,
            "error": str(exc),
        }]


# ── Public async entry point (called by the processing route) ─────────────────

async def process_data_async(
    job_id: str,
    file_id: str,
    mode: ProcessingMode,
    parameters: FitACFParameters,
    stages: List[ProcessingStage],
    backend_override: str = None,
):
    import services.db as _db

    from core.websocket_manager import manager   # singleton, no circular dependency

    async def _update(patch: dict):
        row = _db.get_job(job_id) or {}
        _db.upsert_job({**row, **patch})
        await manager.broadcast({
            "type":     "job_update",
            "job_id":   job_id,
            "status":   patch.get("status", row.get("status", "running")),
            "progress": patch.get("progress", row.get("progress", 0)),
            "stage":    patch.get("current_stage", ""),
        })

    try:
        await _update({"status": "running", "started_at": str(datetime.now()), "progress": 5})

        upload_dir = Path(os.environ.get("DATA_DIR", "/tmp")) / "siw_uploads"
        matches    = list(upload_dir.glob(f"{file_id}_*"))
        if not matches:
            raise FileNotFoundError(f"Uploaded file {file_id} not found")
        file_path = matches[0]

        # ProcessingMode.CUDA is accepted by the schema for backward
        # compat but the only path is CPU now.
        use_gpu = False

        backend = _get_backend(backend_override)
        logger.info(f"Job {job_id} — backend: {type(backend).__name__}")

        # Pass file path so downstream stages can access raw data if fitacf was skipped
        stage_results: Dict = {"_file_path": str(file_path)}
        timing:         Dict = {}
        progress_map = {
            ProcessingStage.ACF:    (20, "acf"),
            ProcessingStage.FITACF: (40, "fitacf"),
            ProcessingStage.LMFIT:  (60, "lmfit"),
            ProcessingStage.GRID:   (80, "grid"),
            ProcessingStage.CNVMAP: (90, "cnvmap"),
        }
        start = time.time()

        for stage in stages:
            pct, key = progress_map.get(stage, (50, stage.value))
            await _update({"progress": pct, "current_stage": stage.value})
            t0 = time.time()

            if stage == ProcessingStage.ACF:
                result = await backend.process_acf(file_path, parameters, use_gpu)
            elif stage == ProcessingStage.FITACF:
                result = await backend.process_fitacf(file_path, parameters, use_gpu)
            elif stage == ProcessingStage.LMFIT:
                result = await backend.process_lmfit(stage_results, parameters, use_gpu)
            elif stage == ProcessingStage.GRID:
                result = await backend.process_grid(stage_results, parameters, use_gpu)
            elif stage == ProcessingStage.CNVMAP:
                result = await backend.process_cnvmap(stage_results, parameters, use_gpu)
            else:
                result = {}

            stage_results[key] = result
            timing[stage.value] = time.time() - t0
            logger.info(f"Stage {stage} done in {timing[stage.value]:.3f}s")

        total_time = time.time() - start

        # Skip private keys (prefixed _) — these are pass-through context, not stages
        clean_stages = {k: {ik: iv for ik, iv in v.items() if not ik.startswith("_")}
                        for k, v in stage_results.items()
                        if not k.startswith("_") and isinstance(v, dict)}

        _db.upsert_result(
            job_id, total_time, clean_stages,
            {"total_time": total_time, "stage_timing": timing,
             "mode": "CPU", "backend": "rst"},
        )
        await _update({"status": "completed", "progress": 100,
                        "completed_at": str(datetime.now()), "current_stage": "complete"})
        logger.info(f"Job {job_id} completed in {total_time:.3f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await _update({"status": "failed", "error": str(e), "completed_at": str(datetime.now())})
