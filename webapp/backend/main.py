"""
SuperDARN Interactive Workbench - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from api.routes import processing, upload, results, remote, settings
from core.websocket_manager import manager   # singleton used by processor.py too
from services.db import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WebSocket connection manager (singleton from core module)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting SuperDARN Interactive Workbench backend...")
    init_db()
    logger.info("SQLite database initialised")
    
    try:
        # Only the RST C-backed processor remains (pythonv2 + CUDArst
        # retired). RST_BINPATH must point at the dir holding the
        # compiled make_fit / make_grid / map_grd binaries.
        rst_bin = os.environ.get("RST_BINPATH", "/opt/rst/bin")
        logger.info(f"RST C backend active — binaries: {rst_bin}")

        yield
        
    finally:
        # Cleanup
        logger.info("Shutting down SuperDARN Interactive Workbench backend...")

# Create FastAPI application
app = FastAPI(
    title="SuperDARN Interactive Workbench API",
    description="REST API for SuperDARN data processing via the optimized C library stack (libgrdopt + libfitacf.3.0).",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(processing.router, prefix="/api/processing", tags=["processing"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
app.include_router(remote.router, prefix="/api/remote", tags=["remote"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])

# WebSocket endpoint for real-time updates
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time processing progress updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        import cupy as cp
        gpu_available = True
        gpu_count = cp.cuda.runtime.getDeviceCount()
    except:
        gpu_available = False
        gpu_count = 0
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "SuperDARN Interactive Workbench API",
        "version": "1.0.0",
        "description": "CUDA-accelerated SuperDARN data processing",
        "docs": "/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
