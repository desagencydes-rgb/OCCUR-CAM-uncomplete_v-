#!/usr/bin/env python3
"""
OCCUR-CAM AI Authentication System v2.0.0
Production-ready main entry point for the OCCUR-CAM face recognition authentication system.
Features:
- Automatic continuous authentication with live camera feed
- Automatic user registration for unknown faces
- Separate databases for users and authentication logs
- Real-time monitoring and logging
- Graceful shutdown handling
"""

import sys
import os
import logging
import signal
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
from datetime import datetime, timedelta
import json
import uuid
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core modules
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.camera_manager import CameraManager
from core.auth_engine import AuthenticationEngine
from core.face_engine import FaceEngine, get_face_engine
from config.settings import config
from config.database import get_auth_db, get_main_db, create_tables, check_database_health
from core.utils import setup_logging
from models.face_models import FrameAnalysis, FaceRecognition, FaceRecognitionConfig

# Database schemas
from database.schemas.auth_schemas import Employee, AuthLog, AuthSession as DBAuthSession, SystemLog
from database.schemas.main_schemas import SystemMetrics, AlertLog

class SystemStatus(Enum):
    """System status types."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class UserRegistration:
    """User registration data."""
    user_id: str
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    face_embedding: Optional[str] = None
    face_photo_path: Optional[str] = None
    quality_score: Optional[float] = None
    registered_at: datetime = None

class OCCURCamSystem:
    """Main OCCUR-CAM system class for production use."""
    
    def __init__(self, camera_source: str = "0", debug: bool = False, dashboard_callback=None):
        """Initialize the OCCUR-CAM system."""
        self.camera_source = camera_source
        self.debug = debug
        self.dashboard_callback = dashboard_callback
        self.status = SystemStatus.STARTING
        
        # Core components
        self.face_engine = None
        self.camera_manager = None
        self.auth_engine = None
        
        # System state
        self.is_running = False
        self.stop_event = threading.Event()
        self.monitoring_thread = None
        
        # Statistics
        self.stats = {
            "total_frames_processed": 0,
            "total_faces_detected": 0,
            "total_authentications": 0,
            "total_registrations": 0,
            "start_time": None,
            "last_activity": None
        }
        
        # User management
        self.known_users = {}  # user_id -> UserRegistration
        self.pending_registrations = {}  # user_id -> UserRegistration
        
        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        setup_logging(log_level)
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logging.info("OCCUR-CAM System initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            logging.info("Initializing OCCUR-CAM system components...")
            
            # Check database health
            db_health = check_database_health()
            if not db_health["overall"]:
                logging.error("Database health check failed")
                return False
            
            # Create tables if they don't exist
            if not create_tables():
                logging.error("Failed to create database tables")
                return False
            
            # Initialize face engine
            logging.info("Initializing face processing engine...")
            face_config = FaceRecognitionConfig(
                detection_threshold=0.4,
                recognition_threshold=0.6,
                max_faces_per_frame=5,
                face_size=(112, 112),
                embedding_size=512,
                model_name=config.face_recognition.MODEL_NAME,
                batch_size=1,
                enable_landmarks=True,
                enable_quality_assessment=True
            )
            
            self.face_engine = get_face_engine(face_config)
            if not self.face_engine.is_initialized:
                logging.error("Failed to initialize face engine")
                return False
            
            # Initialize camera manager
            logging.info("Initializing camera manager...")
            self.camera_manager = CameraManager()
            
            # Connect to camera - use CAM001 if camera_source is 0, otherwise use the source
            camera_id = "CAM001" if self.camera_source == "0" else self.camera_source
            if not self.camera_manager.connect_camera(camera_id):
                logging.error(f"Failed to connect to camera {camera_id}")
                return False
            
            # Initialize authentication engine
            logging.info("Initializing authentication engine...")
            self.auth_engine = AuthenticationEngine()
            
            # Load existing users from database
            self._load_existing_users()
            
            self.status = SystemStatus.RUNNING
            self.stats["start_time"] = datetime.now()
            
            logging.info("OCCUR-CAM system initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize system: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    def _load_existing_users(self):
        """Load existing users from database."""
        try:
            with get_auth_db() as db:
                employees = db.query(Employee).filter(Employee.is_active == True).all()
                
                for emp in employees:
                    user_reg = UserRegistration(
                        user_id=emp.employee_id,
                        first_name=emp.first_name,
                        last_name=emp.last_name,
                        email=emp.email,
                        phone=emp.phone,
                        department=emp.department,
                        position=emp.position,
                        face_embedding=emp.face_embedding,
                        face_photo_path=emp.face_photo_path,
                        quality_score=emp.face_quality_score,
                        registered_at=emp.created_at
                    )
                    self.known_users[emp.employee_id] = user_reg
                
                logging.info(f"Loaded {len(self.known_users)} existing users from database")
                
        except Exception as e:
            logging.error(f"Error loading existing users: {e}")
    
    def start_monitoring(self):
        """Start the continuous monitoring process."""
        if self.status != SystemStatus.RUNNING:
            logging.error("System not in running state")
            return False
        
        try:
            logging.info("Starting continuous authentication monitoring...")
            self.is_running = True
            self.stop_event.clear()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logging.info("Monitoring started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop for continuous authentication."""
        try:
            # Try different camera backends
            cap = None
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(int(self.camera_source) if self.camera_source.isdigit() else 0, backend)
                if cap.isOpened():
                    break
                cap.release()
            
            if not cap or not cap.isOpened():
                logging.error("Failed to open camera with any backend")
                return
            
            # Set camera properties with better error handling
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
            
            frame_count = 0
            last_auth_time = 0
            auth_cooldown = 2.0  # Minimum seconds between authentication attempts
            
            logging.info("Entering continuous monitoring loop...")
            
            while self.is_running and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Validate frame
                if frame is None or frame.size == 0:
                    logging.warning("Invalid frame received")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                self.stats["total_frames_processed"] = frame_count
                self.stats["last_activity"] = datetime.now()
                
                # Process frame for authentication
                current_time = time.time()
                if current_time - last_auth_time >= auth_cooldown:
                    self._process_frame_for_auth(frame, frame_count)
                    last_auth_time = current_time
                
                # Display frame with overlay (optional)
                if self.debug:
                    self._display_frame_with_overlay(frame, frame_count)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.067)  # ~15 FPS for stability
                
        except Exception as e:
            logging.error(f"Error in monitoring loop: {e}")
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            logging.info("Monitoring loop ended")
    
    def _process_frame_for_auth(self, frame: np.ndarray, frame_count: int):
        """Process a single frame for authentication."""
        try:
            # Process frame with face engine
            analysis = self.face_engine.process_frame(frame, "MAIN_CAM")
            
            # Update statistics
            self.stats["total_faces_detected"] += len(analysis.face_detections)
            
            # Check for faces
            if not analysis.face_detections:
                return
            
            # Get recognized faces with lower threshold for better detection
            recognized_faces = analysis.get_recognized_faces(0.3)
            unknown_faces = analysis.get_unknown_faces(0.3)
            
            # Handle recognized faces
            for recognition in recognized_faces:
                if recognition.employee_id and recognition.confidence > 0.4:  # Lower threshold
                    self._handle_known_user(recognition, frame, frame_count)
            
            # Handle unknown faces - report but don't register
            if not recognized_faces:
                for detection in unknown_faces:
                    self._handle_unknown_face(detection, frame, frame_count)
                
        except Exception as e:
            logging.error(f"Error processing frame for auth: {e}")
    
    def _handle_known_user(self, recognition: FaceRecognition, frame: np.ndarray, frame_count: int):
        """Handle authentication of known user."""
        try:
            user_id = recognition.employee_id
            confidence = recognition.confidence
            
            # Log authentication
            self._log_authentication(user_id, "success", confidence, frame_count)
            
            # Update user activity
            if user_id in self.known_users:
                self.known_users[user_id].last_seen = datetime.now()
            
            self.stats["total_authentications"] += 1
            
            # Notify dashboard
            if self.dashboard_callback:
                user_info = self.known_users.get(user_id)
                user_dict = None
                
                if user_info:
                    user_dict = {
                        'first_name': getattr(user_info, 'first_name', 'Unknown'),
                        'last_name': getattr(user_info, 'last_name', 'User')
                    }
                    logging.info(f"Found user in known_users: {user_dict}")
                else:
                    # Try to get user info from database
                    try:
                        from database.schemas.auth_schemas import Employee
                        from config.database import get_auth_db
                        with get_auth_db() as db:
                            employee = db.query(Employee).filter(Employee.employee_id == user_id).first()
                            if employee:
                                user_dict = {
                                    'first_name': employee.first_name or 'Unknown',
                                    'last_name': employee.last_name or 'User'
                                }
                                logging.info(f"Found user in database: {user_dict}")
                            else:
                                logging.warning(f"User {user_id} not found in database")
                                user_dict = None
                    except Exception as e:
                        logging.warning(f"Could not fetch user info from database: {e}")
                        user_dict = None
                
                self.dashboard_callback.notify_recognized_face(user_id, confidence, user_dict)
            
            logging.info(f"Authenticated user {user_id} with confidence {confidence:.2f}")
            
        except Exception as e:
            logging.error(f"Error handling known user: {e}")
    
    def _handle_unknown_face(self, detection, frame: np.ndarray, frame_count: int):
        """Handle unknown face detection - report only, no registration."""
        try:
            # Generate unique detection ID for logging
            detection_id = f"DET_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_count}"
            
            # Notify dashboard about unknown face
            if self.dashboard_callback:
                self.dashboard_callback.notify_unknown_face(detection_id)
            
            # Log unknown face detection
            self._log_authentication("UNKNOWN", "unknown_face", detection.confidence, frame_count)
            
            logging.info(f"Unknown face detected: {detection_id}")
                
        except Exception as e:
            logging.error(f"Error handling unknown face: {e}")
    
    def _save_user_to_database(self, user_reg: UserRegistration):
        """Save user registration to database."""
        try:
            with get_auth_db() as db:
                # Get face embedding from face engine
                face_embedding = None
                if hasattr(self.face_engine.recognizer, 'get_face_embedding'):
                    face_embedding = self.face_engine.recognizer.get_face_embedding(user_reg.user_id)
                
                employee = Employee(
                    employee_id=user_reg.user_id,
                    first_name=user_reg.first_name,
                    last_name=user_reg.last_name,
                    email=user_reg.email,
                    phone=user_reg.phone,
                    department=user_reg.department,
                    position=user_reg.position,
                    is_active=True,
                    hire_date=user_reg.registered_at,
                    face_embedding=json.dumps(face_embedding) if face_embedding else None,
                    face_photo_path=user_reg.face_photo_path,
                    face_quality_score=user_reg.quality_score,
                    created_at=user_reg.registered_at
                )
                
                db.add(employee)
                db.commit()
                
                logging.info(f"User {user_reg.user_id} saved to database")
                
        except Exception as e:
            logging.error(f"Error saving user to database: {e}")
    
    def _log_authentication(self, user_id: str, auth_type: str, confidence: float, frame_count: int):
        """Log authentication attempt to database."""
        try:
            with get_auth_db() as db:
                auth_log = AuthLog(
                    employee_id=user_id,
                    auth_type=auth_type,
                    confidence_score=confidence,
                    processing_time=0.0,  # Could be calculated
                    camera_id="MAIN_CAM",
                    camera_location="Main Entrance",
                    snapshot_path=None,  # Could save snapshot
                    face_count=1,
                    timestamp=datetime.now(),
                    ip_address="127.0.0.1",
                    metadata_json=json.dumps({
                        "frame_count": frame_count,
                        "system_version": "2.0.0"
                    })
                )
                
                db.add(auth_log)
                db.commit()
                
        except Exception as e:
            logging.error(f"Error logging authentication: {e}")
    
    def _display_frame_with_overlay(self, frame: np.ndarray, frame_count: int):
        """Display frame with system information overlay."""
        try:
            # Add system info overlay
            cv2.putText(frame, f"OCCUR-CAM v2.0.0 - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Users: {len(self.known_users)} | Auth: {self.stats['total_authentications']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Registrations: {self.stats['total_registrations']}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face detection area
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detection Area", 
                       (w//4, h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("OCCUR-CAM Live Feed", frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                
        except Exception as e:
            logging.warning(f"Error displaying frame: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            "status": self.status.value,
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "statistics": self.stats.copy(),
            "known_users_count": len(self.known_users),
            "pending_registrations": len(self.pending_registrations),
            "camera_connected": self.camera_manager is not None,
            "face_engine_ready": self.face_engine is not None and self.face_engine.is_initialized,
            "auth_engine_ready": self.auth_engine is not None
        }
    
    def stop(self):
        """Stop the system gracefully."""
        try:
            logging.info("Stopping OCCUR-CAM system...")
            self.status = SystemStatus.STOPPING
            
            # Signal stop
            self.is_running = False
            self.stop_event.set()
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            # Cleanup components
            if self.face_engine:
                self.face_engine.cleanup()
            
            if self.camera_manager:
                self.camera_manager.cleanup()
            
            if self.auth_engine:
                self.auth_engine.cleanup()
            
            self.status = SystemStatus.STOPPED
            logging.info("OCCUR-CAM system stopped successfully")
            
        except Exception as e:
            logging.error(f"Error stopping system: {e}")
            self.status = SystemStatus.ERROR

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="OCCUR-CAM AI Authentication System v2.0.0 - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with default camera (0)
  python main.py --camera 1         # Start with specific camera
  python main.py --debug            # Start in debug mode with visual display
  python main.py --setup            # Setup system and exit
        """
    )
    
    parser.add_argument(
        '--camera', 
        type=str, 
        default='0', 
        help='Camera source (USB index, IP address, or camera ID)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode with visual display'
    )
    
    parser.add_argument(
        '--setup', 
        action='store_true', 
        help='Setup system (create databases, load initial data) and exit'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='OCCUR-CAM v2.0.0'
    )
    
    args = parser.parse_args()
    
    try:
        # Handle setup mode
        if args.setup:
            logging.info("Running system setup...")
            from database.migrations.create_tables import create_all_tables
            from database.migrations.seed_data import seed_initial_data
            
            if create_all_tables() and seed_initial_data():
                logging.info("System setup completed successfully!")
                return 0
            else:
                logging.error("System setup failed!")
                return 1
        
        # Create and initialize system
        system = OCCURCamSystem(camera_source=args.camera, debug=args.debug)
        
        if not system.initialize():
            logging.error("Failed to initialize system")
            return 1
        
        # Start monitoring
        if not system.start_monitoring():
            logging.error("Failed to start monitoring")
            return 1
        
        # Display status
        print("\n" + "=" * 60)
        print("🎬 OCCUR-CAM v2.0.0 - Production System")
        print("=" * 60)
        print("System Status: RUNNING")
        print("Features:")
        print("  ✓ Automatic face detection and recognition")
        print("  ✓ Automatic user registration for unknown faces")
        print("  ✓ Real-time authentication logging")
        print("  ✓ Separate user and authentication databases")
        print("  ✓ Continuous monitoring until stopped")
        print("\nPress Ctrl+C to stop the system")
        print("=" * 60)
        
        # Keep running until stopped
        try:
            while system.is_running:
                time.sleep(1)
                
                # Display periodic status
                if system.debug and system.stats["total_frames_processed"] % 300 == 0:  # Every 10 seconds at 30fps
                    status = system.get_system_status()
                    print(f"\rFrames: {status['statistics']['total_frames_processed']} | "
                          f"Faces: {status['statistics']['total_faces_detected']} | "
                          f"Auth: {status['statistics']['total_authentications']} | "
                          f"Users: {status['known_users_count']}", end="")
        
        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal...")
        
        # Stop system
        system.stop()
        
        # Final status
        final_status = system.get_system_status()
        print(f"\n\nFinal Statistics:")
        print(f"  Total frames processed: {final_status['statistics']['total_frames_processed']}")
        print(f"  Total faces detected: {final_status['statistics']['total_faces_detected']}")
        print(f"  Total authentications: {final_status['statistics']['total_authentications']}")
        print(f"  Total user registrations: {final_status['statistics']['total_registrations']}")
        print(f"  Known users: {final_status['known_users_count']}")
        
        print("\nOCCUR-CAM system shutdown complete")
        return 0
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
