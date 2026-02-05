"""
WebSocket connection manager for real-time updates
"""

from typing import List
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket"""
        await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected WebSockets"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
    
    async def send_progress_update(self, job_id: str, progress: int, message: str, stage: str = ""):
        """Send a progress update for a specific job"""
        await self.broadcast({
            "type": "progress",
            "job_id": job_id,
            "progress": progress,
            "message": message,
            "stage": stage
        })
    
    async def send_result(self, job_id: str, result: dict):
        """Send processing results"""
        await self.broadcast({
            "type": "result",
            "job_id": job_id,
            "result": result
        })
    
    async def send_error(self, job_id: str, error: str):
        """Send an error message"""
        await self.broadcast({
            "type": "error",
            "job_id": job_id,
            "error": error
        })
