"""
OCCUR-CAM IVCam Integration
Integration with IVCam mobile app and other 3rd party camera applications.
"""

import cv2
import numpy as np
import logging
import requests
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from urllib.parse import urljoin
import json
import base64

from config.settings import config
from core.utils import validate_image, save_image, create_timestamped_filename

class IVCamConnection:
    """IVCam mobile app connection handler."""
    
    def __init__(self, ip: str, port: int = 8080, timeout: int = 5):
        """Initialize IVCam connection."""
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{ip}:{port}"
        self.video_url = f"{self.base_url}/video"
        self.is_connected = False
        self.last_ping = None
        self.connection_quality = 0.0
        
        # Connection settings
        self.ping_interval = 10  # seconds
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # Threading
        self.ping_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.connection_callbacks = []
        self.error_callbacks = []
    
    def connect(self) -> bool:
        """Connect to IVCam device."""
        try:
            logging.info(f"Connecting to IVCam at {self.ip}:{self.port}")
            
            # Test connection
            if not self._test_connection():
                return False
            
            # Start ping monitoring
            self._start_ping_monitoring()
            
            self.is_connected = True
            self.last_ping = datetime.now()
            
            # Notify callbacks
            for callback in self.connection_callbacks:
                try:
                    callback(True, self.ip)
                except Exception as e:
                    logging.warning(f"Error in connection callback: {e}")
            
            logging.info(f"Successfully connected to IVCam at {self.ip}:{self.port}")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting to IVCam: {e}")
            self._handle_error(str(e))
            return False
    
    def disconnect(self):
        """Disconnect from IVCam device."""
        try:
            self.is_connected = False
            self.stop_event.set()
            
            if self.ping_thread and self.ping_thread.is_alive():
                self.ping_thread.join(timeout=5)
            
            # Notify callbacks
            for callback in self.connection_callbacks:
                try:
                    callback(False, self.ip)
                except Exception as e:
                    logging.warning(f"Error in disconnection callback: {e}")
            
            logging.info(f"Disconnected from IVCam at {self.ip}:{self.port}")
            
        except Exception as e:
            logging.error(f"Error disconnecting from IVCam: {e}")
    
    def _test_connection(self) -> bool:
        """Test connection to IVCam device."""
        try:
            # Test basic connectivity
            response = requests.get(self.base_url, timeout=self.timeout)
            if response.status_code != 200:
                logging.error(f"IVCam returned status {response.status_code}")
                return False
            
            # Test video stream
            response = requests.get(self.video_url, timeout=self.timeout)
            if response.status_code != 200:
                logging.error(f"IVCam video stream not accessible: {response.status_code}")
                return False
            
            # Test video stream with OpenCV
            cap = cv2.VideoCapture(self.video_url)
            if not cap.isOpened():
                logging.error("Could not open IVCam video stream")
                return False
            
            # Try to capture a frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logging.error("Could not capture frame from IVCam")
                return False
            
            if not validate_image(frame):
                logging.error("Invalid frame received from IVCam")
                return False
            
            # Calculate connection quality
            self.connection_quality = self._calculate_connection_quality(response)
            
            return True
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error connecting to IVCam: {e}")
            return False
        except Exception as e:
            logging.error(f"Error testing IVCam connection: {e}")
            return False
    
    def _calculate_connection_quality(self, response: requests.Response) -> float:
        """Calculate connection quality based on response."""
        try:
            quality = 1.0
            
            # Factor in response time
            response_time = response.elapsed.total_seconds()
            if response_time > 2.0:
                quality -= 0.3
            elif response_time > 1.0:
                quality -= 0.1
            
            # Factor in response size (if available)
            content_length = response.headers.get('content-length')
            if content_length:
                size = int(content_length)
                if size < 1000:  # Very small response
                    quality -= 0.2
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logging.warning(f"Error calculating connection quality: {e}")
            return 0.5
    
    def _start_ping_monitoring(self):
        """Start ping monitoring thread."""
        try:
            self.stop_event.clear()
            self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
            self.ping_thread.start()
            
        except Exception as e:
            logging.error(f"Error starting ping monitoring: {e}")
    
    def _ping_loop(self):
        """Ping monitoring loop."""
        while not self.stop_event.is_set() and self.is_connected:
            try:
                # Ping the device
                response = requests.get(self.base_url, timeout=self.timeout)
                
                if response.status_code == 200:
                    self.last_ping = datetime.now()
                    self.connection_quality = self._calculate_connection_quality(response)
                else:
                    self._handle_error(f"Ping failed with status {response.status_code}")
                
                # Sleep for ping interval
                time.sleep(self.ping_interval)
                
            except Exception as e:
                self._handle_error(f"Ping error: {e}")
                time.sleep(self.retry_delay)
    
    def _handle_error(self, error_msg: str):
        """Handle connection error."""
        logging.warning(f"IVCam connection error: {error_msg}")
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_msg, self.ip)
            except Exception as e:
                logging.warning(f"Error in error callback: {e}")
    
    def get_video_capture(self) -> Optional[cv2.VideoCapture]:
        """Get OpenCV VideoCapture for IVCam stream."""
        try:
            if not self.is_connected:
                logging.error("IVCam not connected")
                return None
            
            cap = cv2.VideoCapture(self.video_url)
            
            if not cap.isOpened():
                logging.error("Could not open IVCam video stream")
                return None
            
            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.camera.FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            return cap
            
        except Exception as e:
            logging.error(f"Error getting IVCam video capture: {e}")
            return None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame from IVCam."""
        try:
            cap = self.get_video_capture()
            if not cap:
                return None
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return None
            
            if not validate_image(frame):
                return None
            
            return frame
            
        except Exception as e:
            logging.error(f"Error capturing frame from IVCam: {e}")
            return None
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get IVCam device information."""
        try:
            if not self.is_connected:
                return {}
            
            # Try to get device info from IVCam API
            info_url = urljoin(self.base_url, "/info")
            response = requests.get(info_url, timeout=self.timeout)
            
            if response.status_code == 200:
                try:
                    device_info = response.json()
                except:
                    device_info = {}
            else:
                device_info = {}
            
            # Add connection info
            device_info.update({
                "ip": self.ip,
                "port": self.port,
                "is_connected": self.is_connected,
                "connection_quality": self.connection_quality,
                "last_ping": self.last_ping.isoformat() if self.last_ping else None,
                "base_url": self.base_url,
                "video_url": self.video_url
            })
            
            return device_info
            
        except Exception as e:
            logging.error(f"Error getting IVCam device info: {e}")
            return {
                "ip": self.ip,
                "port": self.port,
                "is_connected": self.is_connected,
                "error": str(e)
            }
    
    def add_connection_callback(self, callback: Callable[[bool, str], None]):
        """Add connection status callback."""
        self.connection_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, str], None]):
        """Add error callback."""
        self.error_callbacks.append(callback)

class ThirdPartyCameraManager:
    """Manager for 3rd party camera applications."""
    
    def __init__(self):
        """Initialize 3rd party camera manager."""
        self.ivcam_connections: Dict[str, IVCamConnection] = {}
        self.other_cameras: Dict[str, Any] = {}
        
        # Load configurations
        self._load_camera_configs()
    
    def _load_camera_configs(self):
        """Load 3rd party camera configurations."""
        try:
            # Load IVCam configurations
            if config.camera.IVCAM_ENABLED:
                ivcam_connection = IVCamConnection(
                    ip=config.camera.IVCAM_IP,
                    port=config.camera.IVCAM_PORT,
                    timeout=config.camera.TIMEOUT
                )
                
                self.ivcam_connections["ivcam_default"] = ivcam_connection
                
                # Add callbacks
                ivcam_connection.add_connection_callback(self._on_ivcam_connection_change)
                ivcam_connection.add_error_callback(self._on_ivcam_error)
            
            logging.info(f"Loaded {len(self.ivcam_connections)} IVCam connections")
            
        except Exception as e:
            logging.error(f"Error loading 3rd party camera configs: {e}")
    
    def _on_ivcam_connection_change(self, connected: bool, ip: str):
        """Handle IVCam connection change."""
        status = "connected" if connected else "disconnected"
        logging.info(f"IVCam {ip} {status}")
    
    def _on_ivcam_error(self, error_msg: str, ip: str):
        """Handle IVCam error."""
        logging.warning(f"IVCam {ip} error: {error_msg}")
    
    def add_ivcam_connection(self, name: str, ip: str, port: int = 8080) -> bool:
        """Add new IVCam connection."""
        try:
            if name in self.ivcam_connections:
                logging.warning(f"IVCam connection {name} already exists")
                return False
            
            connection = IVCamConnection(ip, port)
            connection.add_connection_callback(self._on_ivcam_connection_change)
            connection.add_error_callback(self._on_ivcam_error)
            
            self.ivcam_connections[name] = connection
            
            logging.info(f"Added IVCam connection {name} at {ip}:{port}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding IVCam connection: {e}")
            return False
    
    def remove_ivcam_connection(self, name: str) -> bool:
        """Remove IVCam connection."""
        try:
            if name not in self.ivcam_connections:
                logging.warning(f"IVCam connection {name} not found")
                return False
            
            connection = self.ivcam_connections[name]
            connection.disconnect()
            del self.ivcam_connections[name]
            
            logging.info(f"Removed IVCam connection {name}")
            return True
            
        except Exception as e:
            logging.error(f"Error removing IVCam connection: {e}")
            return False
    
    def connect_ivcam(self, name: str) -> bool:
        """Connect to IVCam device."""
        try:
            if name not in self.ivcam_connections:
                logging.error(f"IVCam connection {name} not found")
                return False
            
            connection = self.ivcam_connections[name]
            return connection.connect()
            
        except Exception as e:
            logging.error(f"Error connecting to IVCam {name}: {e}")
            return False
    
    def disconnect_ivcam(self, name: str) -> bool:
        """Disconnect from IVCam device."""
        try:
            if name not in self.ivcam_connections:
                logging.error(f"IVCam connection {name} not found")
                return False
            
            connection = self.ivcam_connections[name]
            connection.disconnect()
            
            return True
            
        except Exception as e:
            logging.error(f"Error disconnecting from IVCam {name}: {e}")
            return False
    
    def get_ivcam_capture(self, name: str) -> Optional[cv2.VideoCapture]:
        """Get VideoCapture for IVCam device."""
        try:
            if name not in self.ivcam_connections:
                return None
            
            connection = self.ivcam_connections[name]
            return connection.get_video_capture()
            
        except Exception as e:
            logging.error(f"Error getting IVCam capture {name}: {e}")
            return None
    
    def capture_ivcam_frame(self, name: str) -> Optional[np.ndarray]:
        """Capture frame from IVCam device."""
        try:
            if name not in self.ivcam_connections:
                return None
            
            connection = self.ivcam_connections[name]
            return connection.capture_frame()
            
        except Exception as e:
            logging.error(f"Error capturing frame from IVCam {name}: {e}")
            return None
    
    def get_ivcam_info(self, name: str) -> Dict[str, Any]:
        """Get IVCam device information."""
        try:
            if name not in self.ivcam_connections:
                return {}
            
            connection = self.ivcam_connections[name]
            return connection.get_device_info()
            
        except Exception as e:
            logging.error(f"Error getting IVCam info {name}: {e}")
            return {}
    
    def get_all_ivcam_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all IVCam devices."""
        info = {}
        for name, connection in self.ivcam_connections.items():
            info[name] = connection.get_device_info()
        return info
    
    def connect_all_ivcam(self) -> int:
        """Connect to all IVCam devices."""
        connected_count = 0
        for name in self.ivcam_connections:
            if self.connect_ivcam(name):
                connected_count += 1
        return connected_count
    
    def disconnect_all_ivcam(self) -> int:
        """Disconnect from all IVCam devices."""
        disconnected_count = 0
        for name in self.ivcam_connections:
            if self.disconnect_ivcam(name):
                disconnected_count += 1
        return disconnected_count
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get status of all 3rd party camera connections."""
        status = {
            "ivcam_connections": len(self.ivcam_connections),
            "connected_ivcam": 0,
            "other_cameras": len(self.other_cameras)
        }
        
        for name, connection in self.ivcam_connections.items():
            if connection.is_connected:
                status["connected_ivcam"] += 1
        
        return status
    
    def cleanup(self):
        """Cleanup all 3rd party camera connections."""
        try:
            # Disconnect all IVCam connections
            for name, connection in self.ivcam_connections.items():
                connection.disconnect()
            
            self.ivcam_connections.clear()
            self.other_cameras.clear()
            
            logging.info("3rd party camera manager cleaned up")
            
        except Exception as e:
            logging.error(f"Error during 3rd party camera cleanup: {e}")

# Global 3rd party camera manager instance
_third_party_manager = None

def get_third_party_camera_manager() -> ThirdPartyCameraManager:
    """Get global 3rd party camera manager instance."""
    global _third_party_manager
    
    if _third_party_manager is None:
        _third_party_manager = ThirdPartyCameraManager()
    
    return _third_party_manager

def cleanup_third_party_camera_manager():
    """Cleanup global 3rd party camera manager instance."""
    global _third_party_manager
    
    if _third_party_manager:
        _third_party_manager.cleanup()
        _third_party_manager = None
