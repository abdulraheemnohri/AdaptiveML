"""
WebSocket Manager for real-time communication.
"""

from typing import Dict, List
from fastapi import WebSocket
from loguru import logger
import json


class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = "default"):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
        logger.info(f"New WebSocket connection: {client_id}")
    
    def disconnect(self, websocket: WebSocket, client_id: str = "default"):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            if websocket in self.active_connections[client_id]:
                self.active_connections[client_id].remove(websocket)
                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]
        logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict, client_id: str = None):
        """Broadcast a message to all or specific clients."""
        if client_id:
            connections = self.active_connections.get(client_id, [])
        else:
            connections = []
            for conns in self.active_connections.values():
                connections.extend(conns)
        
        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            for client_list in self.active_connections.values():
                if conn in client_list:
                    client_list.remove(conn)
    
    async def disconnect_all(self):
        """Disconnect all active connections."""
        for client_id, connections in list(self.active_connections.items()):
            for connection in connections:
                try:
                    await connection.close()
                except Exception:
                    pass
            del self.active_connections[client_id]
        logger.info("All WebSocket connections closed")
    
    async def broadcast_training_progress(self, job_id: int, progress: float, metrics: dict = None):
        """Broadcast training progress update."""
        message = {
            "type": "training_progress",
            "job_id": job_id,
            "progress": progress,
            "metrics": metrics or {},
        }
        await self.broadcast(message)
    
    async def broadcast_gpu_metrics(self, gpu_usage: float, vram_usage: float, ram_usage: float):
        """Broadcast GPU metrics update."""
        message = {
            "type": "gpu_metrics",
            "gpu_usage": gpu_usage,
            "vram_usage": vram_usage,
            "ram_usage": ram_usage,
        }
        await self.broadcast(message)
    
    async def broadcast_job_status(self, job_id: int, status: str, message: str = None):
        """Broadcast job status update."""
        msg = {
            "type": "job_status",
            "job_id": job_id,
            "status": status,
        }
        if message:
            msg["message"] = message
        await self.broadcast(msg)
    
    async def broadcast_evaluation_result(self, evaluation_id: int, results: dict):
        """Broadcast evaluation results."""
        message = {
            "type": "evaluation_result",
            "evaluation_id": evaluation_id,
            "results": results,
        }
        await self.broadcast(message)
    
    async def broadcast_data_collection(self, source_id: int, items_collected: int):
        """Broadcast data collection progress."""
        message = {
            "type": "data_collection",
            "source_id": source_id,
            "items_collected": items_collected,
        }
        await self.broadcast(message)
    
    async def broadcast_agent_activity(self, agent_id: int, task_id: int, activity: str):
        """Broadcast agent activity."""
        message = {
            "type": "agent_activity",
            "agent_id": agent_id,
            "task_id": task_id,
            "activity": activity,
        }
        await self.broadcast(message)


# Global instance
websocket_manager = ConnectionManager()
