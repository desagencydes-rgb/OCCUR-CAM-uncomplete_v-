"""
OCCUR-CAM Camera Monitor
Monitors camera health and manages automatic recovery.
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from queue import Queue
from pathlib import Path

class CameraMonitor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cameras = {}
        self.camera_health = {}
        self.frame_queues = {}
        self.running = False
        
        # Camera monitoring settings
        self.frame_timeout = self.config.get('frame_timeout', 5.0)
        self.min_frame_quality = self.config.get('min_frame_quality', 0.5)
        self.max_reconnect_attempts = self.config.get('max_reconnect_attempts', 3)
        self.reconnect_delay = self.config.get('reconnect_delay', 2.0)
        
        # Initialize monitoring thread
        self.monitor_thread = None
    
    def start(self):
        """Start camera monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logging.info("Camera monitor started")
    
    def stop(self):
        """Stop camera monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logging.info("Camera monitor stopped")
    
    def register_camera(self, camera_id: str, camera_obj: Any):
        """Register a camera for monitoring."""
        self.cameras[camera_id] = camera_obj
        self.camera_health[camera_id] = {
            'status': 'unknown',
            'last_frame_time': None,
            'frame_count': 0,
            'error_count': 0,
            'reconnect_attempts': 0,
            'last_error': None
        }
        self.frame_queues[camera_id] = Queue(maxsize=30)  # Buffer last 30 frames
        logging.info(f"Registered camera for monitoring: {camera_id}")
    
    def _monitor_loop(self):
        """Main camera monitoring loop."""
        while self.running:
            try:
                for camera_id, camera in self.cameras.items():
                    self._check_camera_health(camera_id, camera)
                time.sleep(0.1)  # Check interval
            except Exception as e:
                logging.error(f"Error in camera monitor loop: {e}")
    
    def _check_camera_health(self, camera_id: str, camera: Any):
        """Check health of a specific camera."""
        try:
            health = self.camera_health[camera_id]
            
            # Check if camera is responsive
            if hasattr(camera, 'is_connected') and not camera.is_connected:
                self._handle_camera_disconnect(camera_id, camera)
                return
            
            # Try to get a frame
            frame = self._get_camera_frame(camera)
            if frame is None:
                self._handle_frame_error(camera_id, "No frame received")
                return
            
            # Validate frame
            if not self._validate_frame(frame):
                self._handle_frame_error(camera_id, "Invalid frame received")
                return
            
            # Update health metrics
            health['status'] = 'healthy'
            health['last_frame_time'] = datetime.now()
            health['frame_count'] += 1
            health['reconnect_attempts'] = 0  # Reset on successful frame
            
            # Store frame in queue
            if not self.frame_queues[camera_id].full():
                self.frame_queues[camera_id].put(frame)
            
        except Exception as e:
            logging.error(f"Error checking camera {camera_id} health: {e}")
            self._handle_frame_error(camera_id, str(e))
    
    def _get_camera_frame(self, camera: Any) -> Optional[np.ndarray]:
        """Get frame from camera with timeout."""
        try:
            if hasattr(camera, 'get_frame'):
                return camera.get_frame()
            elif hasattr(camera, 'read'):
                ret, frame = camera.read()
                return frame if ret else None
            return None
        except Exception as e:
            logging.error(f"Error getting camera frame: {e}")
            return None
    
    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Validate frame quality."""
        try:
            if frame is None or frame.size == 0:
                return False
            
            # Check frame dimensions
            if len(frame.shape) != 3:
                return False
            
            # Check frame isn't all black or white
            mean_value = np.mean(frame)
            if mean_value < 1 or mean_value > 254:
                return False
            
            # Check frame isn't too blurry
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100:  # Arbitrary threshold
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating frame: {e}")
            return False
    
    def _handle_camera_disconnect(self, camera_id: str, camera: Any):
        """Handle camera disconnect event."""
        health = self.camera_health[camera_id]
        health['status'] = 'disconnected'
        
        if health['reconnect_attempts'] < self.max_reconnect_attempts:
            logging.warning(f"Attempting to reconnect camera {camera_id}")
            try:
                if hasattr(camera, 'reconnect'):
                    camera.reconnect()
                elif hasattr(camera, 'connect'):
                    camera.connect()
                health['reconnect_attempts'] += 1
                time.sleep(self.reconnect_delay)
            except Exception as e:
                logging.error(f"Failed to reconnect camera {camera_id}: {e}")
        else:
            logging.error(f"Max reconnection attempts reached for camera {camera_id}")
    
    def _handle_frame_error(self, camera_id: str, error_msg: str):
        """Handle frame processing error."""
        health = self.camera_health[camera_id]
        health['error_count'] += 1
        health['last_error'] = error_msg
        health['status'] = 'error'
        
        logging.warning(f"Camera {camera_id} error: {error_msg}")
    
    def get_camera_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific camera."""
        return self.camera_health.get(camera_id)
    
    def get_all_cameras_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all cameras."""
        return self.camera_health.copy()

# Global monitor instance
_camera_monitor_instance = None

def get_camera_monitor(config: Dict[str, Any] = None) -> CameraMonitor:
    """Get global camera monitor instance."""
    global _camera_monitor_instance
    
    if _camera_monitor_instance is None:
        _camera_monitor_instance = CameraMonitor(config)
    
    return _camera_monitor_instance

def cleanup_camera_monitor():
    """Cleanup global camera monitor instance."""
    global _camera_monitor_instance
    
    if _camera_monitor_instance:
        _camera_monitor_instance.stop()
        _camera_monitor_instance = None