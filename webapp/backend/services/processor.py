"""
Processing dispatcher.

Reads BACKEND_TYPE env var (default: pythonv2) and routes every job to the
corresponding BackendProcessor implementation.

  BACKEND_TYPE=pythonv2  →  processor_pythonv2.Pythonv2Processor
  BACKEND_TYPE=cuda      →  processor_cuda.CUDArstProcessor
  BACKEND_TYPE=rst       →  processor_rst.RSTProcessor
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

# ── Backend selection ──────────────────────────────────────────────────────────

def _get_backend(override: str = None):
    """
    Return a BackendProcessor instance.

    Priority: per-request `override` → BACKEND_TYPE env var → 'pythonv2'.
    """
    backend_type = (override or os.environ.get("BACKEND_TYPE", "pythonv2")).lower()
    if backend_type == "cuda":
        from services.processor_cuda import CUDArstProcessor
        return CUDArstProcessor()
    elif backend_type == "rst":
        from services.processor_rst import RSTProcessor
        return RSTProcessor()
    else:
        from services.processor_pythonv2 import Pythonv2Processor
        return Pythonv2Processor()


def probe_backends() -> list:
    """
    Return availability info for all three backends.
    Each entry: {id, name, available, gpu, active}.
    """
    active = os.environ.get("BACKEND_TYPE", "pythonv2").lower()
    results = []
    checks = [
        ("pythonv2", "pythonv2 (Python / CuPy)",
         "services.processor_pythonv2", "Pythonv2Processor"),
        ("cuda",     "CUDArst (CUDA / C)",
         "services.processor_cuda",     "CUDArstProcessor"),
        ("rst",      "RST (Reference C)",
         "services.processor_rst",      "RSTProcessor"),
    ]
    for bid, name, module, cls in checks:
        try:
            import importlib
            m = importlib.import_module(module)
            getattr(m, cls)
            gpu = False
            if bid in ("pythonv2", "cuda"):
                try:
                    import cupy  # noqa
                    gpu = True
                except Exception:
                    pass
            results.append({"id": bid, "name": name, "available": True,
                             "gpu": gpu, "active": bid == active})
        except Exception as exc:
            results.append({"id": bid, "name": name, "available": False,
                             "gpu": False, "active": bid == active,
                             "error": str(exc)})
    return results


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

        upload_dir = Path("/tmp/siw_uploads")
        matches    = list(upload_dir.glob(f"{file_id}_*"))
        if not matches:
            raise FileNotFoundError(f"Uploaded file {file_id} not found")
        file_path = matches[0]

        use_gpu = False
        if mode == ProcessingMode.CUDA:
            use_gpu = True
        elif mode == ProcessingMode.AUTO:
            try:
                import cupy  # noqa
                use_gpu = True
            except (ImportError, Exception):
                pass

        backend = _get_backend(backend_override)
        logger.info(f"Job {job_id} — backend: {type(backend).__name__}, gpu: {use_gpu}")

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
             "mode": "GPU" if use_gpu else "CPU",
             "backend": backend_override or os.environ.get("BACKEND_TYPE", "pythonv2")},
        )
        await _update({"status": "completed", "progress": 100,
                        "completed_at": str(datetime.now()), "current_stage": "complete"})
        logger.info(f"Job {job_id} completed in {total_time:.3f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await _update({"status": "failed", "error": str(e), "completed_at": str(datetime.now())})
