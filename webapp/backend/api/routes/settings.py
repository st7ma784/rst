"""
Settings and configuration endpoints — persisted in SQLite.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import platform, os

import services.db as db

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_SETTINGS: Dict[str, Any] = {
    "processing": {
        "default_mode":  "auto",
        "max_batch_size": 256,
        "enable_gpu":    True,
    },
    "visualization": {
        "default_colormap": "viridis",
        "enable_3d":        True,
        "refresh_rate":     30,
    },
    "remote": {
        "timeout":         300,
        "retry_attempts":  3,
    },
}

PRESETS: Dict[str, Any] = {
    "default": {
        "name":        "Default",
        "description": "Standard processing parameters",
        "parameters":  {"min_power": 3.0, "phase_tolerance": 25.0,
                        "elevation_enabled": True,  "elevation_model": "GSM",
                        "batch_size": 64, "xcf_enabled": True},
    },
    "high_quality": {
        "name":        "High Quality",
        "description": "Stricter parameters for better data quality",
        "parameters":  {"min_power": 6.0, "phase_tolerance": 15.0,
                        "elevation_enabled": True,  "elevation_model": "GSM",
                        "batch_size": 64, "xcf_enabled": True},
    },
    "fast": {
        "name":        "Fast Processing",
        "description": "Relaxed parameters for faster processing",
        "parameters":  {"min_power": 1.0, "phase_tolerance": 35.0,
                        "elevation_enabled": False, "elevation_model": "None",
                        "batch_size": 128, "xcf_enabled": False},
    },
}


def _load_settings() -> Dict[str, Any]:
    """Return merged settings: defaults overridden by any persisted values."""
    stored = db.get_settings()
    if not stored:
        return DEFAULT_SETTINGS.copy()
    merged = {k: dict(v) for k, v in DEFAULT_SETTINGS.items()}
    for section, values in stored.items():
        if section in merged and isinstance(values, dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


@router.get("/")
async def get_settings():
    return _load_settings()


@router.put("/")
async def update_settings(settings: Dict[str, Any]):
    current = _load_settings()
    for section, values in settings.items():
        if section in current and isinstance(values, dict):
            current[section].update(values)
        else:
            current[section] = values
    db.save_settings(current)
    return {"message": "Settings saved", "settings": current}


@router.post("/reset")
async def reset_settings():
    db.save_settings(DEFAULT_SETTINGS)
    return {"message": "Settings reset to defaults", "settings": DEFAULT_SETTINGS}


@router.get("/presets")
async def get_presets():
    return {"presets": PRESETS}


@router.get("/system-info")
async def get_system_info():
    gpu_available, gpu_count, gpu_info = False, 0, []
    try:
        import cupy as cp
        gpu_count     = cp.cuda.runtime.getDeviceCount()
        gpu_available = gpu_count > 0
        for i in range(gpu_count):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                gpu_info.append({
                    "id":   i,
                    "name": props["name"].decode("utf-8") if isinstance(props["name"], bytes) else str(props["name"]),
                    "compute_capability": f"{props['major']}.{props['minor']}",
                    "total_memory_gb":    round(props["totalGlobalMem"] / (1024 ** 3), 2),
                })
    except Exception:
        pass

    return {
        "gpu_available":   gpu_available,
        "gpu_count":       gpu_count,
        "gpus":            gpu_info,
        "python_version":  platform.python_version(),
        "os":              platform.system(),
        "cpu_count":       os.cpu_count(),
        "db_path":         str(db.DB_PATH),
    }
