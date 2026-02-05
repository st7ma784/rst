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

# Upload directory
UPLOAD_DIR = Path("/tmp/siw_uploads")
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

@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a SuperDARN data file
    
    Accepts:
    - RAWACF files (.rawacf)
    - FITACF files (.fitacf)
    - Grid files (.grid)
    - HDF5 files (.h5, .hdf5)
    """
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
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
