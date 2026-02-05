"""
Settings and configuration endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Default settings
DEFAULT_SETTINGS = {
    "processing": {
        "default_mode": "auto",
        "max_batch_size": 256,
        "enable_gpu": True
    },
    "visualization": {
        "default_colormap": "viridis",
        "enable_3d": True,
        "refresh_rate": 30
    },
    "remote": {
        "timeout": 300,
        "retry_attempts": 3
    }
}

# Current settings (in-memory, use database in production)
current_settings = DEFAULT_SETTINGS.copy()

@router.get("/")
async def get_settings():
    """Get current application settings"""
    return current_settings

@router.put("/")
async def update_settings(settings: Dict[str, Any]):
    """Update application settings"""
    try:
        # Merge new settings with current
        for category, values in settings.items():
            if category in current_settings:
                current_settings[category].update(values)
            else:
                current_settings[category] = values
        
        logger.info(f"Settings updated: {settings.keys()}")
        
        return {
            "message": "Settings updated successfully",
            "settings": current_settings
        }
    except Exception as e:
        logger.error(f"Settings update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Settings update failed: {str(e)}")

@router.post("/reset")
async def reset_settings():
    """Reset settings to defaults"""
    global current_settings
    current_settings = DEFAULT_SETTINGS.copy()
    
    logger.info("Settings reset to defaults")
    
    return {
        "message": "Settings reset to defaults",
        "settings": current_settings
    }

@router.get("/presets")
async def get_presets():
    """Get available parameter presets"""
    presets = {
        "default": {
            "name": "Default",
            "description": "Standard processing parameters",
            "parameters": {
                "min_power": 3.0,
                "phase_tolerance": 25.0,
                "elevation_enabled": True,
                "elevation_model": "GSM"
            }
        },
        "high_quality": {
            "name": "High Quality",
            "description": "Stricter parameters for better data quality",
            "parameters": {
                "min_power": 6.0,
                "phase_tolerance": 15.0,
                "elevation_enabled": True,
                "elevation_model": "GSM"
            }
        },
        "fast": {
            "name": "Fast Processing",
            "description": "Relaxed parameters for faster processing",
            "parameters": {
                "min_power": 1.0,
                "phase_tolerance": 35.0,
                "elevation_enabled": False,
                "elevation_model": "None"
            }
        }
    }
    
    return {"presets": presets}

@router.get("/system-info")
async def get_system_info():
    """Get system information including GPU availability"""
    try:
        import cupy as cp
        gpu_available = True
        gpu_count = cp.cuda.runtime.getDeviceCount()
        gpu_info = []
        
        for i in range(gpu_count):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                gpu_info.append({
                    "id": i,
                    "name": props["name"].decode("utf-8"),
                    "compute_capability": f"{props['major']}.{props['minor']}",
                    "total_memory": props["totalGlobalMem"] / (1024**3),  # GB
                })
    except:
        gpu_available = False
        gpu_count = 0
        gpu_info = []
    
    return {
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpus": gpu_info,
        "cudarst_version": "2.0.0"
    }
