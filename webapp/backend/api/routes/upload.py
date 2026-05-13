"""
File upload endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import uuid
import aiofiles
import os
from pathlib import Path
import logging

from models.schemas import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Upload directory — override with DATA_DIR env var in production
_DATA_DIR  = Path(os.environ.get("DATA_DIR", "/tmp"))
UPLOAD_DIR = _DATA_DIR / "siw_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def detect_file_format(filename: str, content: bytes) -> str:
    """Detect SuperDARN file format"""
    # Simple format detection based on filename and content
    filename_lower = filename.lower()
    
    if "rawacf" in filename_lower or filename_lower.endswith(".rawacf"):
        return "rawacf"
    elif "fitacf" in filename_lower or filename_lower.endswith(".fitacf"):
        return "fitacf"
    elif "grid" in filename_lower:
        return "grid"
    elif "map" in filename_lower:
        return "map"
    elif filename_lower.endswith(".h5") or filename_lower.endswith(".hdf5"):
        return "hdf5"
    else:
        # Try to detect from content
        if b"rawacf" in content[:1024]:
            return "rawacf"
        elif b"fitacf" in content[:1024]:
            return "fitacf"
        return "unknown"

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

def _sweep_old_uploads(max_age_hours: int = 24) -> None:
    """Delete uploaded files older than max_age_hours."""
    import time
    cutoff = time.time() - max_age_hours * 3600
    for p in UPLOAD_DIR.iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                logger.info(f"Swept old upload: {p.name}")
        except Exception:
            pass


@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a SuperDARN data file (max 500 MB)."""
    try:
        # Sweep stale uploads opportunistically
        _sweep_old_uploads()

        file_id = str(uuid.uuid4())

        # Read in chunks to enforce size limit without buffering the whole file
        CHUNK = 1 << 20   # 1 MB
        chunks, total = [], 0
        while True:
            chunk = await file.read(CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of "
                           f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB"
                )
            chunks.append(chunk)
        content   = b"".join(chunks)
        file_size = total
        
        # Detect format
        file_format = detect_file_format(file.filename, content)
        
        # Save file
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"File uploaded: {file.filename} ({file_size} bytes) as {file_id}")
        
        # Extract basic metadata (can be enhanced)
        metadata = {
            "original_filename": file.filename,
            "content_type": file.content_type,
            "file_path": str(file_path)
        }
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            size=file_size,
            upload_time=datetime.now(),
            format=file_format,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/{file_id}")
async def get_file_info(file_id: str):
    """Get information about an uploaded file"""
    # Find file in upload directory
    files = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    
    if not files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = files[0]
    stat = file_path.stat()
    
    return {
        "file_id": file_id,
        "filename": file_path.name.split("_", 1)[1],
        "size": stat.st_size,
        "upload_time": datetime.fromtimestamp(stat.st_ctime),
        "path": str(file_path)
    }

@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file"""
    files = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    
    if not files:
        raise HTTPException(status_code=404, detail="File not found")
    
    for file_path in files:
        os.remove(file_path)
        logger.info(f"File deleted: {file_id}")
    
    return {"message": "File deleted successfully", "file_id": file_id}
