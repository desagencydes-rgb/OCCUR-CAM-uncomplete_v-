"""
OCCUR-CAM Authentication Engine
Robust authentication system with advanced face recognition for all lighting conditions.
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid

from config.settings import config
from config.database import get_auth_db, get_main_db
from database.schemas.auth_schemas import Employee, AuthLog, AuthSession, SystemLog
from database.schemas.main_schemas import AlertLog, SystemMetrics
from core.face_engine import get_face_engine, FaceEngine
from core.camera_manager import get_camera_manager
from core.utils import enhance_image, normalize_image, validate_image, save_image
from models.face_models import FrameAnalysis, FaceRecognition, FaceRecognitionConfig

class AuthStatus(Enum):
    """Authentication status types."""
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"

class AuthResult(Enum):
    """Authentication result types."""
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    UNKNOWN_FACE = "unknown_face"
    LOW_CONFIDENCE = "low_confidence"
    MULTIPLE_FACES = "multiple_faces"
    NO_FACE = "no_face"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class AuthenticationAttempt:
    """Authentication attempt data."""
    attempt_id: str
    employee_id: Optional[str]
    confidence: float
    status: AuthStatus
    result: AuthResult
    processing_time: float
    camera_id: str
    timestamp: datetime
    frame_analysis: Optional[FrameAnalysis]
    metadata: Dict[str, Any]

@dataclass
class AuthSession:
    """Active authentication session."""
    session_id: str
    employee_id: str
    start_time: datetime
    last_activity: datetime
    camera_id: str
    ip_address: Optional[str]
    is_active: bool
    confidence_history: List[float]

class AuthenticationEngine:
    """Main authentication engine with robust face recognition."""
    
    def __init__(self):
        """Initialize authentication engine."""
        self.face_engine = None
        self.camera_manager = None
        self.is_initialized = False
        
        # Authentication settings
        self.recognition_threshold = config.face_recognition.RECOGNITION_THRESHOLD
        self.detection_threshold = config.face_recognition.DETECTION_THRESHOLD
        self.max_attempts = config.authentication.MAX_ATTEMPTS
        self.timeout = config.authentication.AUTH_TIMEOUT
        self.cooldown_period = config.authentication.COOLDOWN_PERIOD
        
        # Session management
        self.active_sessions: Dict[str, AuthSession] = {}
        self.attempt_history: List[AuthenticationAttempt] = []
        self.employee_cooldowns: Dict[str, datetime] = {}
        
        # Performance tracking
        self.total_attempts = 0
        self.successful_authentications = 0
        self.failed_authentications = 0
        
        # Callbacks
        self.auth_callbacks: List[Callable[[AuthenticationAttempt], None]] = []
        self.session_callbacks: List[Callable[[AuthSession], None]] = []
        
        # Initialize components
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize authentication engine components."""
        try:
            logging.info("Initializing authentication engine...")
            
            # Initialize face engine with enhanced configuration
            face_config = FaceRecognitionConfig(
                detection_threshold=0.4,  # Lower threshold for better detection
                recognition_threshold=0.6,  # Higher threshold for security
                max_faces_per_frame=5,
                face_size=(112, 112),
                embedding_size=512,
                model_name=config.face_recognition.MODEL_NAME,
                batch_size=1,  # Process one at a time for better quality
                enable_landmarks=True,
                enable_quality_assessment=True
            )
            
            self.face_engine = get_face_engine(face_config)
            self.camera_manager = get_camera_manager()
            
            self.is_initialized = True
            logging.info("Authentication engine initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize authentication engine: {e}")
            raise
    
    def authenticate_face(self, frame: np.ndarray, camera_id: str, 
                        ip_address: str = None) -> AuthenticationAttempt:
        """
        Authenticate face from camera frame.
        
        Args:
            frame: Camera frame as numpy array
            camera_id: Camera identifier
            ip_address: Client IP address
            
        Returns:
            AuthenticationAttempt result
        """
        if not self.is_initialized:
            raise RuntimeError("Authentication engine not initialized")
        
        try:
            start_time = time.time()
            attempt_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Preprocess frame for better recognition
            enhanced_frame = self._enhance_frame_for_auth(frame)
            
            # Process frame with face engine
            frame_analysis = self.face_engine.process_frame(enhanced_frame, camera_id)
            
            # Analyze authentication results
            auth_result = self._analyze_authentication(frame_analysis)
            
            # Create authentication attempt
            attempt = AuthenticationAttempt(
                attempt_id=attempt_id,
                employee_id=auth_result.employee_id,
                confidence=auth_result.confidence,
                status=auth_result.status,
                result=auth_result.result,
                processing_time=time.time() - start_time,
                camera_id=camera_id,
                timestamp=timestamp,
                frame_analysis=frame_analysis,
                metadata_json=json.dumps(auth_result.metadata) if auth_result.metadata else None
            )
            
            # Update statistics
            self._update_statistics(attempt)
            
            # Log authentication attempt
            self._log_authentication_attempt(attempt, ip_address)
            
            # Handle successful authentication
            if attempt.status == AuthStatus.SUCCESS:
                self._handle_successful_auth(attempt, ip_address)
            
            # Add to history
            self.attempt_history.append(attempt)
            self._cleanup_old_attempts()
            
            # Call callbacks
            for callback in self.auth_callbacks:
                try:
                    callback(attempt)
                except Exception as e:
                    logging.warning(f"Error in auth callback: {e}")
            
            return attempt
            
        except Exception as e:
            logging.error(f"Error in face authentication: {e}")
            return AuthenticationAttempt(
                attempt_id=str(uuid.uuid4()),
                employee_id=None,
                confidence=0.0,
                status=AuthStatus.ERROR,
                result=AuthResult.ERROR,
                processing_time=0.0,
                camera_id=camera_id,
                timestamp=datetime.now(),
                frame_analysis=None,
                metadata={"error": str(e)}
            )
    
    def _enhance_frame_for_auth(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame for better authentication."""
        try:
            # Validate frame
            if not validate_image(frame):
                return frame
            
            # Apply multiple enhancement techniques
            enhanced = frame.copy()
            
            # 1. Histogram equalization for better contrast
            if len(enhanced.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 2. Denoising
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 3. Sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 4. Brightness and contrast adjustment
            enhanced = self._adjust_brightness_contrast(enhanced)
            
            # 5. Gamma correction for low-light conditions
            enhanced = self._apply_gamma_correction(enhanced)
            
            return enhanced
            
        except Exception as e:
            logging.warning(f"Error enhancing frame: {e}")
            return frame
    
    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast of image."""
        try:
            # Calculate image statistics
            mean_brightness = np.mean(image)
            
            # Adjust brightness
            if mean_brightness < 80:  # Dark image
                alpha = 1.2  # Contrast
                beta = 30    # Brightness
            elif mean_brightness > 180:  # Bright image
                alpha = 0.8  # Contrast
                beta = -20   # Brightness
            else:  # Normal image
                alpha = 1.0  # Contrast
                beta = 0     # Brightness
            
            # Apply adjustment
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return adjusted
            
        except Exception as e:
            logging.warning(f"Error adjusting brightness/contrast: {e}")
            return image
    
    def _apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction for better lighting."""
        try:
            # Calculate gamma based on image brightness
            mean_brightness = np.mean(image)
            
            if mean_brightness < 100:  # Very dark
                gamma = 0.7
            elif mean_brightness < 150:  # Dark
                gamma = 0.8
            elif mean_brightness > 200:  # Very bright
                gamma = 1.3
            else:  # Normal
                gamma = 1.0
            
            # Apply gamma correction
            gamma_corrected = np.power(image / 255.0, gamma) * 255.0
            return gamma_corrected.astype(np.uint8)
            
        except Exception as e:
            logging.warning(f"Error applying gamma correction: {e}")
            return image
    
    def _analyze_authentication(self, frame_analysis: FrameAnalysis) -> 'AuthResult':
        """Analyze frame analysis for authentication decision."""
        try:
            # Check if any faces were detected
            if not frame_analysis.face_detections:
                return AuthResult(
                    employee_id=None,
                    confidence=0.0,
                    status=AuthStatus.FAILURE,
                    result=AuthResult.NO_FACE,
                    metadata={"reason": "no_faces_detected"}
                )
            
            # Check for multiple faces
            if len(frame_analysis.face_detections) > 1:
                return AuthResult(
                    employee_id=None,
                    confidence=0.0,
                    status=AuthStatus.FAILURE,
                    result=AuthResult.MULTIPLE_FACES,
                    metadata={"reason": "multiple_faces", "face_count": len(frame_analysis.face_detections)}
                )
            
            # Get face recognition results
            recognitions = frame_analysis.get_recognized_faces(self.recognition_threshold)
            
            if not recognitions:
                # Check if there are unknown faces
                unknown_faces = frame_analysis.get_unknown_faces(self.recognition_threshold)
                if unknown_faces:
                    return AuthResult(
                        employee_id=None,
                        confidence=0.0,
                        status=AuthStatus.FAILURE,
                        result=AuthResult.UNKNOWN_FACE,
                        metadata={"reason": "unknown_face", "face_count": len(unknown_faces)}
                    )
                else:
                    return AuthResult(
                        employee_id=None,
                        confidence=0.0,
                        status=AuthStatus.FAILURE,
                        result=AuthResult.LOW_CONFIDENCE,
                        metadata={"reason": "low_confidence"}
                    )
            
            # Get the best recognition
            best_recognition = max(recognitions, key=lambda x: x.confidence)
            
            # Check confidence threshold
            if best_recognition.confidence < self.recognition_threshold:
                return AuthResult(
                    employee_id=best_recognition.employee_id,
                    confidence=best_recognition.confidence,
                    status=AuthStatus.FAILURE,
                    result=AuthResult.LOW_CONFIDENCE,
                    metadata={"reason": "low_confidence", "threshold": self.recognition_threshold}
                )
            
            # Check employee cooldown
            if best_recognition.employee_id in self.employee_cooldowns:
                cooldown_end = self.employee_cooldowns[best_recognition.employee_id]
                if datetime.now() < cooldown_end:
                    remaining = (cooldown_end - datetime.now()).total_seconds()
                    return AuthResult(
                        employee_id=best_recognition.employee_id,
                        confidence=best_recognition.confidence,
                        status=AuthStatus.FAILURE,
                        result=AuthResult.TIMEOUT,
                        metadata={"reason": "cooldown", "remaining_seconds": remaining}
                    )
            
            # Successful authentication
            return AuthResult(
                employee_id=best_recognition.employee_id,
                confidence=best_recognition.confidence,
                status=AuthStatus.SUCCESS,
                result=AuthResult.AUTHENTICATED,
                metadata={"reason": "successful_authentication"}
            )
            
        except Exception as e:
            logging.error(f"Error analyzing authentication: {e}")
            return AuthResult(
                employee_id=None,
                confidence=0.0,
                status=AuthStatus.ERROR,
                result=AuthResult.ERROR,
                metadata={"error": str(e)}
            )
    
    def _update_statistics(self, attempt: AuthenticationAttempt):
        """Update authentication statistics."""
        self.total_attempts += 1
        
        if attempt.status == AuthStatus.SUCCESS:
            self.successful_authentications += 1
        else:
            self.failed_authentications += 1
    
    def _log_authentication_attempt(self, attempt: AuthenticationAttempt, ip_address: str = None):
        """Log authentication attempt to database."""
        try:
            with get_auth_db() as db:
                auth_log = AuthLog(
                    employee_id=attempt.employee_id,
                    auth_type=attempt.status.value,
                    confidence_score=attempt.confidence,
                    processing_time=attempt.processing_time,
                    camera_id=attempt.camera_id,
                    camera_location="Unknown",  # Could be enhanced with camera location
                    snapshot_path=None,  # Could save snapshot
                    face_count=len(attempt.frame_analysis.face_detections) if attempt.frame_analysis else 0,
                    timestamp=attempt.timestamp,
                    ip_address=ip_address,
                    metadata_json=json.dumps(attempt.metadata) if attempt.metadata else None
                )
                
                db.add(auth_log)
                db.commit()
                
        except Exception as e:
            logging.error(f"Error logging authentication attempt: {e}")
    
    def _handle_successful_auth(self, attempt: AuthenticationAttempt, ip_address: str = None):
        """Handle successful authentication."""
        try:
            # Create or update session
            session_id = str(uuid.uuid4())
            session = AuthSession(
                session_id=session_id,
                employee_id=attempt.employee_id,
                start_time=attempt.timestamp,
                last_activity=attempt.timestamp,
                camera_id=attempt.camera_id,
                ip_address=ip_address,
                is_active=True,
                confidence_history=[attempt.confidence]
            )
            
            # Store session
            self.active_sessions[attempt.employee_id] = session
            
            # Set cooldown period
            cooldown_end = attempt.timestamp + timedelta(seconds=self.cooldown_period)
            self.employee_cooldowns[attempt.employee_id] = cooldown_end
            
            # Log session to database
            self._log_auth_session(session)
            
            # Call session callbacks
            for callback in self.session_callbacks:
                try:
                    callback(session)
                except Exception as e:
                    logging.warning(f"Error in session callback: {e}")
            
            logging.info(f"Successful authentication for employee {attempt.employee_id}")
            
        except Exception as e:
            logging.error(f"Error handling successful authentication: {e}")
    
    def _log_auth_session(self, session: AuthSession):
        """Log authentication session to database."""
        try:
            with get_auth_db() as db:
                db_session = AuthSession(
                    session_id=session.session_id,
                    employee_id=session.employee_id,
                    created_at=session.start_time,
                    expires_at=session.start_time + timedelta(seconds=config.authentication.SESSION_DURATION),
                    last_activity=session.last_activity,
                    ip_address=session.ip_address,
                    user_agent=None,
                    camera_id=session.camera_id,
                    is_active=session.is_active
                )
                
                db.add(db_session)
                db.commit()
                
        except Exception as e:
            logging.error(f"Error logging auth session: {e}")
    
    def _cleanup_old_attempts(self):
        """Cleanup old authentication attempts."""
        try:
            # Keep only last 1000 attempts
            if len(self.attempt_history) > 1000:
                self.attempt_history = self.attempt_history[-1000:]
            
            # Cleanup old sessions
            current_time = datetime.now()
            expired_sessions = [
                emp_id for emp_id, session in self.active_sessions.items()
                if (current_time - session.last_activity).total_seconds() > config.authentication.SESSION_DURATION
            ]
            
            for emp_id in expired_sessions:
                del self.active_sessions[emp_id]
            
            # Cleanup old cooldowns
            expired_cooldowns = [
                emp_id for emp_id, cooldown_end in self.employee_cooldowns.items()
                if current_time > cooldown_end
            ]
            
            for emp_id in expired_cooldowns:
                del self.employee_cooldowns[emp_id]
                
        except Exception as e:
            logging.warning(f"Error cleaning up old attempts: {e}")
    
    def get_authentication_stats(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        success_rate = 0.0
        if self.total_attempts > 0:
            success_rate = self.successful_authentications / self.total_attempts
        
        return {
            "total_attempts": self.total_attempts,
            "successful_authentications": self.successful_authentications,
            "failed_authentications": self.failed_authentications,
            "success_rate": success_rate,
            "active_sessions": len(self.active_sessions),
            "employees_in_cooldown": len(self.employee_cooldowns),
            "recognition_threshold": self.recognition_threshold,
            "detection_threshold": self.detection_threshold,
            "is_initialized": self.is_initialized
        }
    
    def get_recent_attempts(self, limit: int = 100) -> List[AuthenticationAttempt]:
        """Get recent authentication attempts."""
        return self.attempt_history[-limit:] if self.attempt_history else []
    
    def get_active_sessions(self) -> Dict[str, AuthSession]:
        """Get active authentication sessions."""
        return self.active_sessions.copy()
    
    def add_auth_callback(self, callback: Callable[[AuthenticationAttempt], None]):
        """Add authentication callback."""
        self.auth_callbacks.append(callback)
    
    def add_session_callback(self, callback: Callable[[AuthSession], None]):
        """Add session callback."""
        self.session_callbacks.append(callback)
    
    def update_thresholds(self, detection_threshold: float = None, 
                         recognition_threshold: float = None):
        """Update authentication thresholds."""
        try:
            if detection_threshold is not None:
                self.detection_threshold = detection_threshold
                if self.face_engine:
                    self.face_engine.config.detection_threshold = detection_threshold
            
            if recognition_threshold is not None:
                self.recognition_threshold = recognition_threshold
                if self.face_engine:
                    self.face_engine.config.recognition_threshold = recognition_threshold
            
            logging.info(f"Updated thresholds - detection: {self.detection_threshold}, recognition: {self.recognition_threshold}")
            
        except Exception as e:
            logging.error(f"Error updating thresholds: {e}")
    
    def cleanup(self):
        """Cleanup authentication engine."""
        try:
            self.active_sessions.clear()
            self.attempt_history.clear()
            self.employee_cooldowns.clear()
            self.auth_callbacks.clear()
            self.session_callbacks.clear()
            
            logging.info("Authentication engine cleaned up")
            
        except Exception as e:
            logging.error(f"Error during auth engine cleanup: {e}")

# Global authentication engine instance
_auth_engine = None

def get_auth_engine() -> AuthenticationEngine:
    """Get global authentication engine instance."""
    global _auth_engine
    
    if _auth_engine is None:
        _auth_engine = AuthenticationEngine()
    
    return _auth_engine

def cleanup_auth_engine():
    """Cleanup global authentication engine instance."""
    global _auth_engine
    
    if _auth_engine:
        _auth_engine.cleanup()
        _auth_engine = None
