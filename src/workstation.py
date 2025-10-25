#!/usr/bin/env python3
"""
OCCUR-CAM Workstation Interface v1.0.0
Production-ready interface for real-world face recognition authentication work.
"""

import sys
import os
import logging
import signal
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
from datetime import datetime
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level to reach project root
sys.path.insert(0, str(project_root))

# Import core modules
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.camera_manager import CameraManager
from core.auth_engine import AuthenticationEngine
from config.settings import config
from core.utils import setup_logging
from config.database import auth_engine, main_engine
from database.schemas.auth_schemas import Employee, AuthLog
from sqlalchemy.orm import sessionmaker

class OCCURCamWorkstation:
    """Production-ready OCCUR-CAM workstation interface with automatic authentication."""
    
    def __init__(self, camera_source="0", debug_mode=False):
        self.camera_source = camera_source
        self.debug_mode = debug_mode
        self.is_running = False
        self.face_engine = None
        self.camera_manager = None
        self.auth_engine = None
        self.current_employee = None
        self.auth_sessions = []
        self.alerts = []
        
        # Automatic authentication settings
        self.auto_auth_enabled = True
        self.auto_auth_thread = None
        self.auto_auth_running = False
        self.last_face_detection = None
        self.face_detection_interval = 2.0  # Check every 2 seconds
        self.recognition_threshold = 0.6
        
        # Database sessions
        self.auth_session = sessionmaker(bind=auth_engine)()
        self.main_session = sessionmaker(bind=main_engine)()
        
        # In-memory cache for quick access
        self.employee_cache = {}
        
    def initialize(self):
        """Initialize the workstation."""
        try:
            logging.info("Initializing OCCUR-CAM Workstation...")
            
            # Initialize face processing engine
            logging.info("Loading AI face recognition models...")
            from core.face_engine import FaceEngine
            self.face_engine = FaceEngine()
            
            # Load employee embeddings from database
            self._load_employee_embeddings()
            
            logging.info("Face recognition AI loaded")
            
            # Initialize camera manager
            logging.info("Initializing camera system...")
            self.camera_manager = CameraManager()
            # Try to connect camera (simplified approach)
            try:
                import cv2
                cap = cv2.VideoCapture(int(self.camera_source))
                if cap.isOpened():
                    cap.release()
                    logging.info("Camera connected and ready")
                else:
                    logging.warning("Camera connection failed")
            except Exception as e:
                logging.warning(f"Camera connection failed: {e}")
            
            # Initialize authentication engine
            logging.info("Initializing authentication system...")
            self.auth_engine = AuthenticationEngine()
            logging.info("Authentication system ready")
            
            self.is_running = True
            
            # Start automatic authentication
            if self.auto_auth_enabled:
                self.start_automatic_authentication()
            
            logging.info("OCCUR-CAM Workstation initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize workstation: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Stop automatic authentication
            self.stop_automatic_authentication()
            
            if self.camera_manager:
                self.camera_manager.cleanup()
            if self.face_engine:
                self.face_engine.cleanup()
            if self.auth_engine:
                self.auth_engine.cleanup()
            logging.info("Workstation cleanup completed")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def _load_employee_embeddings(self):
        """Load employee embeddings from database into face recognizer."""
        try:
            employees = self.auth_session.query(Employee).filter(Employee.is_active == True).all()
            
            for employee in employees:
                if employee.face_embedding:
                    try:
                        # Parse the JSON embedding
                        embedding_data = json.loads(employee.face_embedding)
                        
                        # Handle both old format (direct array) and new format (with 'embedding' key)
                        if isinstance(embedding_data, list):
                            # Old format - direct array
                            embedding_array = np.array(embedding_data, dtype=np.float32)
                        elif isinstance(embedding_data, dict) and 'embedding' in embedding_data:
                            # New format - with 'embedding' key
                            embedding_array = np.array(embedding_data['embedding'], dtype=np.float32)
                        else:
                            logging.warning(f"Unknown embedding format for employee {employee.employee_id}")
                            continue
                        
                        # Add to face recognizer
                        self.face_engine.recognizer.employee_embeddings[employee.employee_id] = embedding_array
                        
                        # Update cache
                        self.employee_cache[employee.employee_id] = {
                            "name": f"{employee.first_name} {employee.last_name}".strip(),
                            "department": employee.department,
                            "face_embedding": embedding,
                            "registered_at": employee.created_at.isoformat() if employee.created_at else "",
                            "confidence": 0.0
                        }
                        
                    except Exception as e:
                        logging.warning(f"Failed to load embedding for employee {employee.employee_id}: {e}")
            
            logging.info(f"Loaded {len(employees)} employee embeddings")
            
        except Exception as e:
            logging.error(f"Error loading employee embeddings: {e}")
    
    def start_automatic_authentication(self):
        """Start automatic face authentication in background thread."""
        try:
            if self.auto_auth_running:
                return
            
            self.auto_auth_running = True
            self.auto_auth_thread = threading.Thread(target=self._automatic_auth_loop, daemon=True)
            self.auto_auth_thread.start()
            logging.info("Automatic authentication started")
            
        except Exception as e:
            logging.error(f"Failed to start automatic authentication: {e}")
    
    def stop_automatic_authentication(self):
        """Stop automatic face authentication."""
        try:
            self.auto_auth_running = False
            if self.auto_auth_thread and self.auto_auth_thread.is_alive():
                self.auto_auth_thread.join(timeout=1.0)
            logging.info("Automatic authentication stopped")
        except Exception as e:
            logging.error(f"Error stopping automatic authentication: {e}")
    
    def _automatic_auth_loop(self):
        """Background loop for automatic face authentication with camera feed."""
        import cv2
        
        try:
            # Initialize camera for continuous monitoring
            cap = cv2.VideoCapture(int(self.camera_source))
            if not cap.isOpened():
                logging.error("Cannot access camera for automatic authentication")
                return
            
            logging.info("Automatic authentication monitoring started")
            print("\n🎥 Camera feed active - Press 'q' to stop auto authentication")
            
            while self.auto_auth_running and self.is_running:
                try:
                    # Capture frame
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Create display frame
                    display_frame = frame.copy()
                    
                    # Process frame for face detection and recognition
                    analysis = self.face_engine.process_frame(frame, "auto_auth_cam")
                    
                    # Draw face detection rectangles and info
                    if analysis.face_detections:
                        for i, detection in enumerate(analysis.face_detections):
                            # Draw face rectangle
                            x, y, w, h = detection.bbox
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Draw confidence
                            cv2.putText(display_frame, f"Face {i+1}: {detection.confidence:.2f}", 
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Get best detection
                        best_detection = max(analysis.face_detections, key=lambda d: d.confidence)
                        logging.debug(f"Face detected with confidence: {best_detection.confidence:.2f}")
                        
                        # Check if this is a new face (avoid duplicate detections)
                        if (self.last_face_detection is None or 
                            self._is_different_face(best_detection, self.last_face_detection)):
                            
                            self.last_face_detection = best_detection
                            logging.debug("New face detected, processing recognition...")
                            
                            # Try to recognize the face
                            if analysis.face_recognitions:
                                best_recognition = max(analysis.face_recognitions, 
                                                    key=lambda r: r.confidence if r.confidence else 0)
                                
                                if (best_recognition.employee_id and 
                                    best_recognition.confidence >= self.recognition_threshold):
                                    
                                    # Employee recognized!
                                    logging.info(f"Employee recognized: {best_recognition.employee_id} (confidence: {best_recognition.confidence:.2f})")
                                    self._handle_employee_recognition(best_recognition, best_detection)
                                    
                                    # Draw recognition info on frame
                                    x, y, w, h = best_detection.bbox
                                    cv2.putText(display_frame, f"RECOGNIZED: {best_recognition.employee_id}", 
                                               (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    cv2.putText(display_frame, f"Confidence: {best_recognition.confidence:.2f}", 
                                               (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    # Unknown face
                                    logging.debug(f"Unknown face detected (confidence: {best_recognition.confidence:.2f})")
                                    self._handle_unknown_face(best_detection)
                                    
                                    # Draw unknown face info
                                    x, y, w, h = best_detection.bbox
                                    cv2.putText(display_frame, "UNKNOWN FACE", 
                                               (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                # No recognition result, treat as unknown
                                logging.debug("No recognition result, treating as unknown")
                                self._handle_unknown_face(best_detection)
                                
                                # Draw unknown face info
                                x, y, w, h = best_detection.bbox
                                cv2.putText(display_frame, "UNKNOWN FACE", 
                                           (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        logging.debug("No faces detected in frame")
                        # Draw "No face detected" message
                        h, w = display_frame.shape[:2]
                        cv2.putText(display_frame, "No face detected", 
                                   (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Draw status info
                    cv2.putText(display_frame, "AUTO AUTH ACTIVE - Press 'q' to stop", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Registered Users: {len(self.employee_cache)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Active Sessions: {len(self.auth_sessions)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show frame
                    cv2.imshow("OCCUR-CAM Auto Authentication", display_frame)
                    
                    # Check for 'q' key to stop
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n🛑 Stopping auto authentication...")
                        self.stop_automatic_authentication()
                        break
                    
                    # Sleep to avoid overwhelming the system
                    time.sleep(self.face_detection_interval)
                    
                except Exception as e:
                    logging.warning(f"Error in automatic auth loop: {e}")
                    time.sleep(1.0)
            
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Automatic authentication monitoring stopped")
            
        except Exception as e:
            logging.error(f"Fatal error in automatic auth loop: {e}")
    
    def _is_different_face(self, detection1, detection2):
        """Check if two face detections are different faces."""
        if detection2 is None:
            return True
        
        # Calculate distance between face centers
        # detection.bbox is (x, y, width, height)
        center1_x = detection1.bbox[0] + detection1.bbox[2] // 2
        center1_y = detection1.bbox[1] + detection1.bbox[3] // 2
        
        center2_x = detection2.bbox[0] + detection2.bbox[2] // 2
        center2_y = detection2.bbox[1] + detection2.bbox[3] // 2
        
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        # If faces are more than 100 pixels apart, consider them different
        return distance > 100
    
    def _handle_employee_recognition(self, recognition, detection):
        """Handle successful employee recognition."""
        try:
            employee_id = recognition.employee_id
            confidence = recognition.confidence
            
            # Get employee from database
            employee = self.auth_session.query(Employee).filter(Employee.employee_id == employee_id).first()
            
            if employee:
                # Create authentication log in database
                auth_log = AuthLog(
                    employee_id=employee.id,  # Use the database ID, not the employee_id string
                    auth_type="success",
                    confidence_score=confidence,
                    processing_time=0.0,  # Could calculate actual processing time
                    camera_id="auto_auth_cam",
                    timestamp=datetime.now(),
                    ip_address="127.0.0.1",  # Local for now
                    user_agent="OCCUR-CAM Workstation",
                metadata_json=json.dumps({
                    "face_detection_confidence": float(detection.confidence),
                    "recognition_confidence": float(confidence),
                    "bbox": [int(float(x)) for x in detection.bbox.tolist()] if hasattr(detection.bbox, 'tolist') else [int(float(x)) for x in detection.bbox]
                })
                )
                
                self.auth_session.add(auth_log)
                self.auth_session.commit()
                
                # Create in-memory session for display
                session = {
                    "employee_id": employee_id,
                    "name": f"{employee.first_name} {employee.last_name}".strip(),
                    "department": employee.department,
                    "timestamp": datetime.now().isoformat(),
                    "face_detected": True,
                    "confidence": confidence,
                    "recognition_result": "success",
                    "success": True
                }
                
                self.auth_sessions.append(session)
                self.current_employee = session
                
                # Log the recognition
                employee_name = f"{employee.first_name} {employee.last_name}".strip()
                logging.info(f"Employee {employee_name} recognized (confidence: {confidence:.2f})")
                
                # Show notification
                print(f"\n🔐 EMPLOYEE RECOGNIZED: {employee_name} (Confidence: {confidence:.1%})")
                
            else:
                # Employee not in database
                logging.warning(f"Recognized employee {employee_id} not found in database")
                
        except Exception as e:
            logging.error(f"Error handling employee recognition: {e}")
            self.auth_session.rollback()
    
    def _handle_unknown_face(self, detection):
        """Handle unknown face detection."""
        try:
            # Create authentication log in database for unknown face
            auth_log = AuthLog(
                employee_id=None,
                auth_type="unknown",
                confidence_score=detection.confidence,
                processing_time=0.0,
                camera_id="auto_auth_cam",
                timestamp=datetime.now(),
                ip_address="127.0.0.1",
                user_agent="OCCUR-CAM Workstation",
                metadata_json=json.dumps({
                    "face_detection_confidence": float(detection.confidence),
                    "bbox": [int(float(x)) for x in detection.bbox.tolist()] if hasattr(detection.bbox, 'tolist') else [int(float(x)) for x in detection.bbox]
                })
            )
            
            self.auth_session.add(auth_log)
            self.auth_session.commit()
            
            # Create in-memory session for display
            session = {
                "employee_id": None,
                "name": "Unknown Person",
                "timestamp": datetime.now().isoformat(),
                "face_detected": True,
                "confidence": detection.confidence,
                "recognition_result": "unknown",
                "success": False
            }
            
            self.auth_sessions.append(session)
            
            # Log the detection
            logging.info(f"Unknown face detected (confidence: {detection.confidence:.2f})")
            
        except Exception as e:
            logging.error(f"Error handling unknown face: {e}")
            self.auth_session.rollback()
    
    def get_system_status(self):
        """Get current system status."""
        try:
            # Get database counts
            try:
                employee_count = self.auth_session.query(Employee).count()
                auth_log_count = self.auth_session.query(AuthLog).count()
            except:
                employee_count = len(self.employee_cache)
                auth_log_count = len(self.auth_sessions)
            
            status = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "face_engine": "Ready" if self.face_engine else "Not Available",
                "camera": "Connected" if self.camera_manager else "Not Available",
                "auth_engine": "Ready" if self.auth_engine else "Not Available",
                "auto_auth": "Active" if self.auto_auth_running else "Inactive",
                "current_employee": self.current_employee or {},
                "active_sessions": len(self.auth_sessions),
                "registered_employees": employee_count,
                "total_auth_logs": auth_log_count,
                "alerts": len(self.alerts)
            }
            return status
        except Exception as e:
            logging.error(f"Error getting system status: {e}")
            return {
                "timestamp": "Unknown",
                "face_engine": "Error",
                "camera": "Error",
                "auth_engine": "Error",
                "current_employee": {},
                "active_sessions": 0,
                "alerts": 0
            }
    
    def authenticate_employee(self, employee_id: str, name: str = None):
        """Authenticate an employee."""
        try:
            if not self.face_engine or not self.camera_manager:
                return {"success": False, "error": "System not ready"}
            
            logging.info(f"Starting authentication for employee: {employee_id}")
            
            # Capture frame from camera
            import cv2
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {"success": False, "error": "Camera not available"}
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return {"success": False, "error": "Failed to capture frame"}
            
            # Process frame for face detection and recognition
            analysis = self.face_engine.process_frame(frame, "workstation_cam")
            
            if not analysis.face_detections:
                return {"success": False, "error": "No face detected"}
            
            # Get the best face detection
            best_detection = max(analysis.face_detections, key=lambda d: d.confidence)
            
            # Get the corresponding recognition result
            recognition_result = None
            if analysis.face_recognitions:
                # Find recognition result that matches the best detection
                for rec in analysis.face_recognitions:
                    if (hasattr(rec, 'face_detection') and 
                        rec.face_detection is not None and 
                        rec.face_detection.bbox == best_detection.bbox):
                        recognition_result = rec
                        break
            
            # Create authentication session
            session = {
                "employee_id": employee_id,
                "name": name or f"Employee {employee_id}",
                "timestamp": datetime.now().isoformat(),
                "face_detected": True,
                "confidence": best_detection.confidence,
                "recognition_result": recognition_result.result if recognition_result else "Unknown",
                "success": recognition_result.result == "success" if recognition_result else False
            }
            
            self.auth_sessions.append(session)
            self.current_employee = session if session["success"] else None
            
            # Log the authentication
            if session["success"]:
                logging.info(f"Authentication successful for {session['name']}")
            else:
                logging.warning(f"Authentication failed for {session['name']}")
            
            return {
                "success": session["success"],
                "employee_id": employee_id,
                "name": session["name"],
                "confidence": best_detection.confidence,
                "timestamp": session["timestamp"]
            }
            
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return {"success": False, "error": str(e)}
    
    def add_employee(self, employee_id: str, name: str, department: str = None):
        """Add a new employee to the system."""
        try:
            if not self.face_engine or not self.camera_manager:
                return {"success": False, "error": "System not ready"}
            
            logging.info(f"Adding new employee: {name} (ID: {employee_id})")
            
            # Capture face for registration
            import cv2
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {"success": False, "error": "Camera not available"}
            
            print(f"📸 Please look at the camera for {name}...")
            print("Press 'c' to capture, 'q' to cancel")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Show preview
                cv2.putText(frame, f"Registering: {name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'c' to capture, 'q' to cancel", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw face detection area
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
                
                cv2.imshow("Employee Registration", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return {"success": False, "error": "Registration cancelled"}
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Process frame for face detection
            analysis = self.face_engine.process_frame(frame, "workstation_cam")
            
            if not analysis.face_detections:
                return {"success": False, "error": "No face detected in captured image"}
            
            # Get the best face detection
            best_detection = max(analysis.face_detections, key=lambda d: d.confidence)
            
            # Extract face region from the detection
            x, y, w, h = best_detection.bbox
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size == 0:
                return {"success": False, "error": "Failed to extract face region"}
            
            # Generate face embedding from the face region
            try:
                embedding = self.face_engine.recognizer.generate_embedding(face_region)
                if embedding is None:
                    return {"success": False, "error": "Failed to generate face embedding"}
            except Exception as e:
                return {"success": False, "error": f"Error generating embedding: {str(e)}"}
            
            # Save employee data (simplified - in real system, save to database)
            try:
                # Convert embedding to list safely
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                elif isinstance(embedding, np.ndarray):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding) if embedding is not None else []
                
                employee_data = {
                    "employee_id": employee_id,
                    "name": name,
                    "department": department or "General",
                    "face_embedding": embedding_list,
                    "registered_at": datetime.now().isoformat(),
                    "confidence": best_detection.confidence
                }
            except Exception as e:
                return {"success": False, "error": f"Error processing embedding data: {str(e)}"}
            
            # Save to database
            try:
                # Check if employee already exists
                existing_employee = self.auth_session.query(Employee).filter(Employee.employee_id == employee_id).first()
                
                if existing_employee:
                    # Update existing employee
                    existing_employee.first_name = name.split()[0] if name else "Unknown"
                    existing_employee.last_name = name.split()[1] if len(name.split()) > 1 else ""
                    existing_employee.department = department or "General"
                    existing_employee.face_embedding = json.dumps({
                        "embedding": embedding_list,
                        "created_at": datetime.now().isoformat()
                    })
                    existing_employee.updated_at = datetime.now()
                    self.auth_session.commit()
                    logging.info(f"Updated existing employee {name} (ID: {employee_id})")
                else:
                    # Create new employee record
                    employee = Employee(
                        employee_id=employee_id,
                        first_name=name.split()[0] if name else "Unknown",
                        last_name=name.split()[1] if len(name.split()) > 1 else "",
                        department=department or "General",
                        face_embedding=json.dumps({
                            "embedding": embedding_list,
                            "created_at": datetime.now().isoformat()
                        }),  # Store as JSON string with proper format
                        is_active=True,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    self.auth_session.add(employee)
                    self.auth_session.commit()
                    logging.info(f"Created new employee {name} (ID: {employee_id})")
                
                # Update cache
                self.employee_cache[employee_id] = {
                    "name": name,
                    "department": department or "General",
                    "face_embedding": embedding_list,
                    "registered_at": datetime.now().isoformat(),
                    "confidence": best_detection.confidence
                }
                
                # Add embedding to face recognizer
                embedding_array = np.array(embedding_list, dtype=np.float32)
                self.face_engine.recognizer.employee_embeddings[employee_id] = embedding_array
                self.face_engine.recognizer.employee_metadata[employee_id] = {
                    'name': name,
                    'department': department or "General",
                    'quality_score': best_detection.confidence,
                    'created_at': datetime.now().isoformat()
                }
                
                logging.info(f"Employee {name} registered successfully in database")
                
            except Exception as e:
                logging.error(f"Error saving employee to database: {e}")
                self.auth_session.rollback()
                return {"success": False, "error": f"Database error: {str(e)}"}
            
            return {
                "success": True,
                "employee_id": employee_id,
                "name": name,
                "confidence": best_detection.confidence
            }
            
        except Exception as e:
            logging.error(f"Error adding employee: {e}")
            return {"success": False, "error": str(e)}
    
    def get_recent_sessions(self, limit=10):
        """Get recent authentication sessions."""
        return self.auth_sessions[-limit:] if self.auth_sessions else []
    
    def clear_sessions(self):
        """Clear all authentication sessions."""
        self.auth_sessions.clear()
        self.current_employee = None
        logging.info("Authentication sessions cleared")
    
    def export_sessions(self, filename=None, format="xlsx"):
        """Export authentication sessions to file."""
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"auth_sessions_{timestamp}.{format}"
            
            # Get data from database
            auth_logs = self.auth_session.query(AuthLog).order_by(AuthLog.timestamp.desc()).all()
            
            if format.lower() == "xlsx":
                return self._export_to_xlsx(auth_logs, filename)
            elif format.lower() == "csv":
                return self._export_to_csv(auth_logs, filename)
            else:
                return {"success": False, "error": "Unsupported format. Use 'xlsx' or 'csv'"}
            
        except Exception as e:
            logging.error(f"Error exporting sessions: {e}")
            return {"success": False, "error": str(e)}
    
    def _export_to_xlsx(self, auth_logs, filename):
        """Export to XLSX format."""
        try:
            import pandas as pd
            
            data = []
            for log in auth_logs:
                data.append({
                    "ID": log.id,
                    "Employee ID": log.employee_id or "Unknown",
                    "Camera ID": log.camera_id,
                    "Result": log.result,
                    "Confidence": float(log.confidence) if log.confidence else 0.0,
                    "Processing Time": float(log.processing_time) if log.processing_time else 0.0,
                    "Timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "IP Address": log.ip_address,
                    "User Agent": log.user_agent,
                    "Metadata": log.metadata_json
                })
            
            df = pd.DataFrame(data)
            df.to_excel(filename, index=False, engine='openpyxl')
            
            logging.info(f"Sessions exported to {filename}")
            return {"success": True, "filename": filename}
            
        except ImportError:
            return {"success": False, "error": "pandas and openpyxl required for XLSX export. Install with: pip install pandas openpyxl"}
        except Exception as e:
            return {"success": False, "error": f"XLSX export error: {str(e)}"}
    
    def _export_to_csv(self, auth_logs, filename):
        """Export to CSV format."""
        try:
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    "ID", "Employee ID", "Camera ID", "Result", "Confidence", 
                    "Processing Time", "Timestamp", "IP Address", "User Agent", "Metadata"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for log in auth_logs:
                    writer.writerow({
                        "ID": log.id,
                        "Employee ID": log.employee_id or "Unknown",
                        "Camera ID": log.camera_id,
                        "Result": log.result,
                        "Confidence": float(log.confidence) if log.confidence else 0.0,
                        "Processing Time": float(log.processing_time) if log.processing_time else 0.0,
                        "Timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "IP Address": log.ip_address,
                        "User Agent": log.user_agent,
                        "Metadata": log.metadata_json
                    })
            
            logging.info(f"Sessions exported to {filename}")
            return {"success": True, "filename": filename}
            
        except Exception as e:
            return {"success": False, "error": f"CSV export error: {str(e)}"}

def display_main_dashboard(workstation):
    """Display the main workstation dashboard."""
    try:
        status = workstation.get_system_status()
        
        print("\n" + "=" * 80)
        print("🎬 OCCUR-CAM WORKSTATION v1.0.0")
        print("=" * 80)
        print(f"📅 {status.get('timestamp', 'Unknown')}")
        print(f"🤖 Face AI: {status.get('face_engine', 'Unknown')}")
        print(f"📹 Camera: {status.get('camera', 'Unknown')}")
        print(f"🔐 Auth System: {status.get('auth_engine', 'Unknown')}")
        print(f"🔄 Auto Auth: {status.get('auto_auth', 'Unknown')}")
        print(f"👤 Current Employee: {status.get('current_employee', {}).get('name', 'None')}")
        print(f"📊 Active Sessions: {status.get('active_sessions', 0)}")
        print(f"👥 Registered Employees: {status.get('registered_employees', 0)}")
        print(f"📋 Total Auth Logs: {status.get('total_auth_logs', 0)}")
        print(f"⚠️ Alerts: {status.get('alerts', 0)}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error displaying dashboard: {e}")

def display_work_menu():
    """Display the work menu."""
    print("\n" + "=" * 80)
    print("🎬 OCCUR-CAM WORK MENU")
    print("=" * 80)
    print("1. Register New Employee")
    print("2. View Recent Sessions")
    print("3. Export Sessions")
    print("4. Clear Sessions")
    print("5. System Status")
    print("6. Test Camera")
    print("7. Toggle Auto Auth")
    print("8. Exit")
    print("=" * 80)

def authenticate_employee_workflow(workstation):
    """Workflow for authenticating an employee."""
    try:
        print("\n" + "=" * 60)
        print("🔐 EMPLOYEE AUTHENTICATION")
        print("=" * 60)
        
        employee_id = input("Enter Employee ID: ").strip()
        if not employee_id:
            print("❌ Employee ID required")
            return
        
        name = input("Enter Employee Name (optional): ").strip()
        
        print(f"\n📸 Authenticating {name or employee_id}...")
        print("Please look at the camera...")
        
        result = workstation.authenticate_employee(employee_id, name)
        
        if result["success"]:
            print(f"✅ Authentication successful!")
            print(f"   Employee: {result['name']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Time: {result['timestamp']}")
        else:
            print(f"❌ Authentication failed: {result.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Authentication cancelled")
    except Exception as e:
        print(f"❌ Authentication error: {e}")

def register_employee_workflow(workstation):
    """Workflow for registering a new employee."""
    try:
        print("\n" + "=" * 60)
        print("👤 EMPLOYEE REGISTRATION")
        print("=" * 60)
        
        employee_id = input("Enter Employee ID: ").strip()
        if not employee_id:
            print("❌ Employee ID required")
            return
        
        name = input("Enter Employee Name: ").strip()
        if not name:
            print("❌ Employee name required")
            return
        
        department = input("Enter Department (optional): ").strip()
        
        result = workstation.add_employee(employee_id, name, department)
        
        if result["success"]:
            print(f"✅ Employee registered successfully!")
            print(f"   ID: {result['employee_id']}")
            print(f"   Name: {result['name']}")
            print(f"   Confidence: {result['confidence']:.2f}")
        else:
            print(f"❌ Registration failed: {result.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Registration cancelled")
    except Exception as e:
        print(f"❌ Registration error: {e}")

def view_sessions_workflow(workstation):
    """Workflow for viewing recent sessions."""
    try:
        print("\n" + "=" * 80)
        print("📊 RECENT AUTHENTICATION SESSIONS")
        print("=" * 80)
        
        sessions = workstation.get_recent_sessions(20)
        if not sessions:
            print("No sessions found")
            return
        
        for i, session in enumerate(sessions, 1):
            status = "✅" if session["success"] else "❌"
            print(f"{i:2d}. {status} {session['name']} ({session['employee_id']})")
            print(f"     Time: {session['timestamp']}")
            print(f"     Confidence: {session['confidence']:.2f}")
            print()
            
    except Exception as e:
        print(f"❌ Error viewing sessions: {e}")

def test_camera_workflow(workstation):
    """Workflow for testing camera."""
    try:
        print("\n" + "=" * 60)
        print("📹 CAMERA TEST")
        print("=" * 60)
        print("Testing camera feed...")
        print("Press 'q' to quit")
        
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available")
            return
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Add test info
            cv2.putText(frame, f"Camera Test - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face detection area
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detection Area", 
                       (w//4, h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Camera Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"✅ Camera test completed: {frame_count} frames in {elapsed:.1f}s")
        print(f"   Average FPS: {frame_count/elapsed:.1f}")
        
    except Exception as e:
        print(f"❌ Camera test error: {e}")

def export_sessions_workflow(workstation):
    """Workflow for exporting authentication sessions."""
    try:
        print("\n" + "=" * 60)
        print("📤 EXPORT AUTHENTICATION SESSIONS")
        print("=" * 60)
        
        # Choose format
        print("Select export format:")
        print("1. XLSX (Excel)")
        print("2. CSV (Comma Separated Values)")
        
        format_choice = input("Enter choice (1-2): ").strip()
        
        if format_choice == '1':
            format_type = "xlsx"
        elif format_choice == '2':
            format_type = "csv"
        else:
            print("❌ Invalid choice")
            return
        
        # Optional filename
        filename = input("Enter filename (or press Enter for auto-generated): ").strip()
        if not filename:
            filename = None
        
        print(f"\n📤 Exporting sessions to {format_type.upper()} format...")
        
        result = workstation.export_sessions(filename, format_type)
        
        if result["success"]:
            print(f"✅ Sessions exported successfully!")
            print(f"   File: {result['filename']}")
            print(f"   Format: {format_type.upper()}")
        else:
            print(f"❌ Export failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Export error: {e}")

def toggle_auto_auth_workflow(workstation):
    """Workflow for toggling automatic authentication."""
    try:
        print("\n" + "=" * 60)
        print("🔄 AUTO AUTHENTICATION TOGGLE")
        print("=" * 60)
        
        current_status = "Active" if workstation.auto_auth_running else "Inactive"
        print(f"Current Status: {current_status}")
        
        if workstation.auto_auth_running:
            print("Stopping automatic authentication...")
            workstation.stop_automatic_authentication()
            print("✅ Automatic authentication stopped")
        else:
            print("Starting automatic authentication...")
            workstation.start_automatic_authentication()
            print("✅ Automatic authentication started")
        
        print(f"New Status: {'Active' if workstation.auto_auth_running else 'Inactive'}")
        
    except Exception as e:
        print(f"❌ Error toggling auto authentication: {e}")

def main_workflow(workstation):
    """Main workflow loop."""
    while True:
        try:
            display_main_dashboard(workstation)
            display_work_menu()
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                register_employee_workflow(workstation)
            elif choice == '2':
                view_sessions_workflow(workstation)
            elif choice == '3':
                export_sessions_workflow(workstation)
            elif choice == '4':
                workstation.clear_sessions()
                print("✅ Sessions cleared")
            elif choice == '5':
                display_main_dashboard(workstation)
            elif choice == '6':
                test_camera_workflow(workstation)
            elif choice == '7':
                toggle_auto_auth_workflow(workstation)
            elif choice == '8':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-8.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("Press Enter to continue...")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="OCCUR-CAM Workstation Interface v1.0.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python occur_cam_workstation.py              # Start workstation
  python occur_cam_workstation.py --camera 0  # Use specific camera
  python occur_cam_workstation.py --debug     # Debug mode
        """
    )
    
    parser.add_argument(
        '--camera', 
        type=str, 
        default='0', 
        help='Camera source (USB index)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='OCCUR-CAM Workstation v1.0.0'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(log_level)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Shutting down...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logging.info("=" * 80)
        logging.info("OCCUR-CAM Workstation Interface v1.0.0")
        logging.info("=" * 80)
        logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Camera source: {args.camera}")
        logging.info(f"Debug mode: {args.debug}")
        
        # Initialize workstation
        workstation = OCCURCamWorkstation(args.camera, args.debug)
        
        if not workstation.initialize():
            logging.error("Failed to initialize workstation")
            return 1
        
        # Start main workflow
        main_workflow(workstation)
        
        # Cleanup
        workstation.cleanup()
        logging.info("Workstation shutdown complete")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        return 0
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

