"""
OCCUR-CAM Camera Manager
Manages multiple camera sources including USB, IP, IVCam, and RTSP cameras.
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import requests
from queue import Queue, Empty
import json

from config.settings import config
from config.database import get_auth_db, get_main_db
from database.schemas.auth_schemas import CameraConfig as AuthCameraConfig
from database.schemas.main_schemas import Camera, CameraConfig as MainCameraConfig
from core.utils import validate_image, save_image, create_timestamped_filename

class CameraSource:
    """Individual camera source handler."""
    
    def __init__(self, camera_id: str, source: str, camera_type: str, 
                 config: Dict[str, Any] = None):
        """Initialize camera source."""
        self.camera_id = camera_id
        self.source = source
        self.camera_type = camera_type
        self.config = config or {}
        
        # Camera state
        self.cap = None
        self.is_connected = False
        self.is_streaming = False
        self.last_frame = None
        self.last_frame_time = None
        self.frame_count = 0
        self.error_count = 0
        self.last_error = None
        self.last_error_time = None
        
        # Performance metrics
        self.fps = 0.0
        self.avg_processing_time = 0.0
        self.health_score = 0.0
        
        # Threading
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.frame_queue = Queue(maxsize=10)
        
        # Callbacks
        self.frame_callbacks = []
        self.error_callbacks = []
    
    def connect(self) -> bool:
        """Connect to camera source."""
        try:
            logging.info(f"Connecting to camera {self.camera_id} ({self.camera_type})")
            
            if self.camera_type == "usb":
                return self._connect_usb()
            elif self.camera_type == "ip":
                return self._connect_ip()
            elif self.camera_type == "ivcam":
                return self._connect_ivcam()
            elif self.camera_type == "rtsp":
                return self._connect_rtsp()
            else:
                logging.error(f"Unsupported camera type: {self.camera_type}")
                return False
                
        except Exception as e:
            logging.error(f"Error connecting to camera {self.camera_id}: {e}")
            self._handle_error(str(e))
            return False
    
    def _connect_usb(self) -> bool:
        """Connect to USB camera."""
        try:
            # Parse source (should be integer for USB)
            source = int(self.source)
            
            # Open camera
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open USB camera {source}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('width', config.camera.WIDTH))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('height', config.camera.HEIGHT))
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('fps', config.camera.FPS))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ValueError("Failed to capture test frame")
            
            self.is_connected = True
            self.last_frame = frame
            self.last_frame_time = datetime.now()
            
            logging.info(f"USB camera {self.camera_id} connected successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting USB camera: {e}")
            return False
    
    def _connect_ip(self) -> bool:
        """Connect to IP camera."""
        try:
            # Open IP camera stream
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open IP camera {self.source}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('width', config.camera.WIDTH))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('height', config.camera.HEIGHT))
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('fps', config.camera.FPS))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ValueError("Failed to capture test frame")
            
            self.is_connected = True
            self.last_frame = frame
            self.last_frame_time = datetime.now()
            
            logging.info(f"IP camera {self.camera_id} connected successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting IP camera: {e}")
            return False
    
    def _connect_ivcam(self) -> bool:
        """Connect to IVCam mobile app."""
        try:
            # IVCam typically uses HTTP stream
            ivcam_url = f"http://{config.camera.IVCAM_IP}:{config.camera.IVCAM_PORT}/video"
            
            # Test connection first
            response = requests.get(ivcam_url, timeout=5)
            if response.status_code != 200:
                raise ValueError(f"IVCam not accessible: {response.status_code}")
            
            # Open video stream
            self.cap = cv2.VideoCapture(ivcam_url)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open IVCam stream {ivcam_url}")
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('width', config.camera.WIDTH))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('height', config.camera.HEIGHT))
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('fps', config.camera.FPS))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ValueError("Failed to capture test frame from IVCam")
            
            self.is_connected = True
            self.last_frame = frame
            self.last_frame_time = datetime.now()
            
            logging.info(f"IVCam {self.camera_id} connected successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting IVCam: {e}")
            return False
    
    def _connect_rtsp(self) -> bool:
        """Connect to RTSP camera stream."""
        try:
            # Open RTSP stream
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open RTSP stream {self.source}")
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('width', config.camera.WIDTH))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('height', config.camera.HEIGHT))
            self.cap.set(cv2.CAP_PROP_FPS, self.config.get('fps', config.camera.FPS))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ValueError("Failed to capture test frame from RTSP")
            
            self.is_connected = True
            self.last_frame = frame
            self.last_frame_time = datetime.now()
            
            logging.info(f"RTSP camera {self.camera_id} connected successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting RTSP camera: {e}")
            return False
    
    def start_streaming(self):
        """Start camera streaming in separate thread."""
        if not self.is_connected:
            logging.error(f"Camera {self.camera_id} not connected")
            return False
        
        if self.is_streaming:
            logging.warning(f"Camera {self.camera_id} already streaming")
            return True
        
        try:
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            self.is_streaming = True
            
            logging.info(f"Started streaming for camera {self.camera_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error starting stream for camera {self.camera_id}: {e}")
            return False
    
    def stop_streaming(self):
        """Stop camera streaming."""
        try:
            self.is_streaming = False
            self.stop_event.set()
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5)
            
            logging.info(f"Stopped streaming for camera {self.camera_id}")
            
        except Exception as e:
            logging.error(f"Error stopping stream for camera {self.camera_id}: {e}")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        last_fps_time = time.time()
        frame_times = []
        
        while not self.stop_event.is_set() and self.is_connected:
            try:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self._handle_error("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Validate frame
                if not validate_image(frame):
                    self._handle_error("Invalid frame captured")
                    time.sleep(0.1)
                    continue
                
                # Update frame data
                self.last_frame = frame
                self.last_frame_time = datetime.now()
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                frame_times.append(current_time)
                
                # Keep only last 30 frame times for FPS calculation
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                if len(frame_times) > 1:
                    self.fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    pass  # Queue full, skip frame
                
                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame, self.camera_id)
                    except Exception as e:
                        logging.warning(f"Error in frame callback: {e}")
                
                # Calculate processing time
                processing_time = time.time() - start_time
                self.avg_processing_time = (self.avg_processing_time * 0.9 + processing_time * 0.1)
                
                # Update health score
                self._update_health_score()
                
                # Control frame rate
                target_fps = self.config.get('fps', config.camera.FPS)
                if target_fps > 0:
                    sleep_time = 1.0 / target_fps - processing_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
            except Exception as e:
                self._handle_error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def _update_health_score(self):
        """Update camera health score."""
        try:
            # Base health score
            health = 1.0
            
            # Reduce for errors
            if self.error_count > 0:
                health -= min(self.error_count * 0.1, 0.5)
            
            # Reduce for low FPS
            target_fps = self.config.get('fps', config.camera.FPS)
            if target_fps > 0 and self.fps < target_fps * 0.5:
                health -= 0.3
            
            # Reduce for old frames
            if self.last_frame_time:
                age = (datetime.now() - self.last_frame_time).total_seconds()
                if age > 5.0:  # No frame for 5 seconds
                    health -= 0.5
            
            # Reduce for high processing time
            if self.avg_processing_time > 0.1:  # More than 100ms per frame
                health -= 0.2
            
            self.health_score = max(0.0, min(1.0, health))
            
        except Exception as e:
            logging.warning(f"Error updating health score: {e}")
            self.health_score = 0.0
    
    def _handle_error(self, error_msg: str):
        """Handle camera error."""
        self.error_count += 1
        self.last_error = error_msg
        self.last_error_time = datetime.now()
        
        logging.warning(f"Camera {self.camera_id} error: {error_msg}")
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_msg, self.camera_id)
            except Exception as e:
                logging.warning(f"Error in error callback: {e}")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get latest frame from camera."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame without waiting."""
        return self.last_frame
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, str], None]):
        """Add frame callback function."""
        self.frame_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, str], None]):
        """Add error callback function."""
        self.error_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get camera status information."""
        return {
            "camera_id": self.camera_id,
            "source": self.source,
            "camera_type": self.camera_type,
            "is_connected": self.is_connected,
            "is_streaming": self.is_streaming,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "health_score": self.health_score,
            "avg_processing_time": self.avg_processing_time,
            "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None
        }
    
    def disconnect(self):
        """Disconnect from camera."""
        try:
            self.stop_streaming()
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.is_connected = False
            self.last_frame = None
            self.last_frame_time = None
            
            logging.info(f"Disconnected camera {self.camera_id}")
            
        except Exception as e:
            logging.error(f"Error disconnecting camera {self.camera_id}: {e}")

class CameraManager:
    """Main camera management system."""
    
    def __init__(self):
        """Initialize camera manager."""
        self.cameras: Dict[str, CameraSource] = {}
        self.is_running = False
        self.health_check_thread = None
        self.stop_event = threading.Event()
        
        # Load camera configurations
        self._load_camera_configs()
    
    def _load_camera_configs(self):
        """Load camera configurations from database and config files."""
        try:
            # Load from YAML config
            config_file = Path("config/camera_config.yaml")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    cameras_config = yaml_config.get('cameras', {})
                    
                    for camera_id, cam_config in cameras_config.items():
                        if cam_config.get('enabled', True):
                            self._add_camera_from_config(camera_id, cam_config)
            
            # Load from database
            self._load_cameras_from_database()
            
            logging.info(f"Loaded {len(self.cameras)} camera configurations")
            
        except Exception as e:
            logging.error(f"Error loading camera configurations: {e}")
    
    def _add_camera_from_config(self, camera_id: str, config: Dict[str, Any]):
        """Add camera from configuration."""
        try:
            camera_source = CameraSource(
                camera_id=camera_id,
                source=config['source'],
                camera_type=config['type'],
                config={
                    'width': config.get('width', config.camera.WIDTH),
                    'height': config.get('height', config.camera.HEIGHT),
                    'fps': config.get('fps', config.camera.FPS),
                    'timeout': config.get('timeout', config.camera.TIMEOUT)
                }
            )
            
            self.cameras[camera_id] = camera_source
            logging.info(f"Added camera {camera_id} from config")
            
        except Exception as e:
            logging.error(f"Error adding camera {camera_id}: {e}")
    
    def _load_cameras_from_database(self):
        """Load cameras from database."""
        try:
            with get_auth_db() as db:
                db_cameras = db.query(AuthCameraConfig).filter(
                    AuthCameraConfig.is_enabled == True
                ).all()
                
                for db_camera in db_cameras:
                    if db_camera.camera_id not in self.cameras:
                        camera_source = CameraSource(
                            camera_id=db_camera.camera_id,
                            source=db_camera.source,
                            camera_type=db_camera.camera_type,
                            config={
                                'width': db_camera.width,
                                'height': db_camera.height,
                                'fps': db_camera.fps,
                                'timeout': db_camera.timeout
                            }
                        )
                        
                        self.cameras[db_camera.camera_id] = camera_source
                        logging.info(f"Added camera {db_camera.camera_id} from database")
            
        except Exception as e:
            logging.error(f"Error loading cameras from database: {e}")
    
    def add_camera(self, camera_id: str, source: str, camera_type: str, 
                   config: Dict[str, Any] = None) -> bool:
        """Add a new camera."""
        try:
            if camera_id in self.cameras:
                logging.warning(f"Camera {camera_id} already exists")
                return False
            
            camera_source = CameraSource(camera_id, source, camera_type, config)
            self.cameras[camera_id] = camera_source
            
            logging.info(f"Added camera {camera_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding camera {camera_id}: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera."""
        try:
            if camera_id not in self.cameras:
                logging.warning(f"Camera {camera_id} not found")
                return False
            
            camera = self.cameras[camera_id]
            camera.disconnect()
            del self.cameras[camera_id]
            
            logging.info(f"Removed camera {camera_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error removing camera {camera_id}: {e}")
            return False
    
    def connect_camera(self, camera_id: str) -> bool:
        """Connect to a specific camera."""
        try:
            if camera_id not in self.cameras:
                logging.error(f"Camera {camera_id} not found")
                return False
            
            camera = self.cameras[camera_id]
            return camera.connect()
            
        except Exception as e:
            logging.error(f"Error connecting camera {camera_id}: {e}")
            return False
    
    def disconnect_camera(self, camera_id: str) -> bool:
        """Disconnect from a specific camera."""
        try:
            if camera_id not in self.cameras:
                logging.error(f"Camera {camera_id} not found")
                return False
            
            camera = self.cameras[camera_id]
            camera.disconnect()
            
            return True
            
        except Exception as e:
            logging.error(f"Error disconnecting camera {camera_id}: {e}")
            return False
    
    def start_camera(self, camera_id: str) -> bool:
        """Start streaming from a specific camera."""
        try:
            if camera_id not in self.cameras:
                logging.error(f"Camera {camera_id} not found")
                return False
            
            camera = self.cameras[camera_id]
            
            if not camera.is_connected:
                if not camera.connect():
                    return False
            
            return camera.start_streaming()
            
        except Exception as e:
            logging.error(f"Error starting camera {camera_id}: {e}")
            return False
    
    def stop_camera(self, camera_id: str) -> bool:
        """Stop streaming from a specific camera."""
        try:
            if camera_id not in self.cameras:
                logging.error(f"Camera {camera_id} not found")
                return False
            
            camera = self.cameras[camera_id]
            camera.stop_streaming()
            
            return True
            
        except Exception as e:
            logging.error(f"Error stopping camera {camera_id}: {e}")
            return False
    
    def get_camera_frame(self, camera_id: str, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get frame from specific camera."""
        try:
            if camera_id not in self.cameras:
                return None
            
            camera = self.cameras[camera_id]
            return camera.get_frame(timeout)
            
        except Exception as e:
            logging.error(f"Error getting frame from camera {camera_id}: {e}")
            return None
    
    def get_camera_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific camera."""
        try:
            if camera_id not in self.cameras:
                return None
            
            camera = self.cameras[camera_id]
            return camera.get_status()
            
        except Exception as e:
            logging.error(f"Error getting status for camera {camera_id}: {e}")
            return None
    
    def get_all_cameras_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all cameras."""
        status = {}
        for camera_id, camera in self.cameras.items():
            status[camera_id] = camera.get_status()
        return status
    
    def start_all_cameras(self) -> bool:
        """Start all enabled cameras."""
        try:
            success_count = 0
            for camera_id in self.cameras:
                if self.start_camera(camera_id):
                    success_count += 1
            
            logging.info(f"Started {success_count}/{len(self.cameras)} cameras")
            return success_count > 0
            
        except Exception as e:
            logging.error(f"Error starting all cameras: {e}")
            return False
    
    def stop_all_cameras(self) -> bool:
        """Stop all cameras."""
        try:
            for camera_id in self.cameras:
                self.stop_camera(camera_id)
            
            logging.info("Stopped all cameras")
            return True
            
        except Exception as e:
            logging.error(f"Error stopping all cameras: {e}")
            return False
    
    def start_health_monitoring(self):
        """Start camera health monitoring."""
        try:
            if self.health_check_thread and self.health_check_thread.is_alive():
                logging.warning("Health monitoring already running")
                return
            
            self.stop_event.clear()
            self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self.health_check_thread.start()
            
            logging.info("Started camera health monitoring")
            
        except Exception as e:
            logging.error(f"Error starting health monitoring: {e}")
    
    def stop_health_monitoring(self):
        """Stop camera health monitoring."""
        try:
            self.stop_event.set()
            
            if self.health_check_thread and self.health_check_thread.is_alive():
                self.health_check_thread.join(timeout=5)
            
            logging.info("Stopped camera health monitoring")
            
        except Exception as e:
            logging.error(f"Error stopping health monitoring: {e}")
    
    def _health_check_loop(self):
        """Health monitoring loop."""
        while not self.stop_event.is_set():
            try:
                for camera_id, camera in self.cameras.items():
                    self._check_camera_health(camera_id, camera)
                
                # Sleep for health check interval
                time.sleep(config.camera.HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logging.error(f"Error in health check loop: {e}")
                time.sleep(5)
    
    def _check_camera_health(self, camera_id: str, camera: CameraSource):
        """Check health of individual camera."""
        try:
            # Check if camera is responsive
            if camera.is_connected and camera.last_frame_time:
                age = (datetime.now() - camera.last_frame_time).total_seconds()
                if age > 10.0:  # No frame for 10 seconds
                    logging.warning(f"Camera {camera_id} appears unresponsive")
                    camera._handle_error("Camera unresponsive - no frames received")
            
            # Update health score
            camera._update_health_score()
            
            # Log health status
            if camera.health_score < 0.5:
                logging.warning(f"Camera {camera_id} health score low: {camera.health_score:.2f}")
            
        except Exception as e:
            logging.error(f"Error checking health for camera {camera_id}: {e}")
    
    def cleanup(self):
        """Cleanup all cameras and resources."""
        try:
            self.stop_health_monitoring()
            self.stop_all_cameras()
            
            for camera in self.cameras.values():
                camera.disconnect()
            
            self.cameras.clear()
            
            logging.info("Camera manager cleaned up")
            
        except Exception as e:
            logging.error(f"Error during camera manager cleanup: {e}")

# Global camera manager instance
_camera_manager = None

def get_camera_manager() -> CameraManager:
    """Get global camera manager instance."""
    global _camera_manager
    
    if _camera_manager is None:
        _camera_manager = CameraManager()
    
    return _camera_manager

def cleanup_camera_manager():
    """Cleanup global camera manager instance."""
    global _camera_manager
    
    if _camera_manager:
        _camera_manager.cleanup()
        _camera_manager = None
