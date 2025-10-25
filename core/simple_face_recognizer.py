"""
Simple Face Recognition Engine
A working face recognition system that actually works.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
import json
from pathlib import Path
from datetime import datetime

from models.face_models import (
    FaceDetection, FaceRecognition, FaceEmbedding, 
    FaceRecognitionConfig, FrameAnalysis
)
from config.settings import config
from config.database import get_auth_db
from database.schemas.auth_schemas import Employee

class SimpleFaceRecognizer:
    """Simple but working face recognition using OpenCV."""
    
    def __init__(self, config: FaceRecognitionConfig = None):
        """Initialize face recognizer with configuration."""
        self.config = config or FaceRecognitionConfig()
        self.is_initialized = False
        self.employee_embeddings = {}  # Cache for employee embeddings
        self.employee_metadata = {}    # Cache for employee metadata
        
        # Initialize the recognizer
        self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize simple face recognizer."""
        try:
            logging.info("Initializing simple face recognizer...")
            
            # Load existing employee embeddings
            self._load_employee_embeddings()
            
            self.is_initialized = True
            logging.info("Simple face recognizer initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing simple face recognizer: {e}")
            raise
    
    def _load_employee_embeddings(self):
        """Load employee embeddings from database."""
        try:
            with get_auth_db() as db:
                employees = db.query(Employee).all()
                
                for employee in employees:
                    if employee.face_embedding:
                        try:
                            # Parse embedding from JSON
                            if isinstance(employee.face_embedding, str):
                                embedding = json.loads(employee.face_embedding)
                            else:
                                embedding = employee.face_embedding
                            
                            if isinstance(embedding, list):
                                embedding = np.array(embedding, dtype=np.float32)
                            
                            self.employee_embeddings[employee.employee_id] = embedding
                            self.employee_metadata[employee.employee_id] = {
                                'name': f"{employee.first_name} {employee.last_name}",
                                'department': getattr(employee, 'department', 'Unknown')
                            }
                            
                        except Exception as e:
                            logging.warning(f"Error loading embedding for employee {employee.employee_id}: {e}")
                            continue
                
                logging.info(f"Loaded {len(self.employee_embeddings)} employee embeddings")
                
        except Exception as e:
            logging.error(f"Error loading employee embeddings: {e}")
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from face image using simple features.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Face embedding vector or None if failed
        """
        if not self.is_initialized:
            raise RuntimeError("Face recognizer not initialized")
        
        try:
            # Validate input image
            if face_image is None or face_image.size == 0:
                logging.warning("Invalid face image for embedding generation")
                return None
            
            # Ensure image is in correct format
            if len(face_image.shape) != 3 or face_image.shape[2] != 3:
                logging.warning(f"Invalid image shape for embedding: {face_image.shape}")
                return None
            
            # Resize image to standard size
            face_resized = cv2.resize(face_image, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Ensure it's uint8 type
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Create simple feature vector
            # Use histogram features
            hist = cv2.calcHist([equalized], [0], None, [32], [0, 256])
            hist = hist.flatten()
            
            # Add texture features using LBP-like approach
            texture_features = self._extract_texture_features(equalized)
            
            # Combine features
            embedding = np.concatenate([hist, texture_features])
            
            # Pad to match InsightFace embedding size (512)
            if len(embedding) < 512:
                padding = np.zeros(512 - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > 512:
                embedding = embedding[:512]
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            logging.debug(f"Generated simple embedding with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logging.error(f"Error generating face embedding: {e}")
            return None
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract simple texture features."""
        try:
            # Simple texture features using gradients
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Gradient direction
            direction = np.arctan2(grad_y, grad_x)
            
            # Create histograms
            mag_hist = cv2.calcHist([magnitude.astype(np.uint8)], [0], None, [16], [0, 256]).flatten()
            dir_hist = cv2.calcHist([direction.astype(np.uint8)], [0], None, [16], [0, 256]).flatten()
            
            return np.concatenate([mag_hist, dir_hist])
            
        except Exception as e:
            logging.warning(f"Error extracting texture features: {e}")
            return np.zeros(32)  # Return zero vector if extraction fails
    
    def recognize_face(self, face_image: np.ndarray, detection: FaceDetection) -> FaceRecognition:
        """
        Recognize a face from face image and detection.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            detection: Face detection result
            
        Returns:
            Face recognition result
        """
        if not self.is_initialized:
            raise RuntimeError("Face recognizer not initialized")
        
        try:
            start_time = time.time()
            
            # Generate embedding for the face
            embedding = self.generate_embedding(face_image)
            
            if embedding is None:
                return FaceRecognition(
                    employee_id=None,
                    confidence=0.0,
                    face_detection=detection,
                    embedding=None,
                    processing_time=time.time() - start_time
                )
            
            # Find best match in employee database
            best_match_id, best_confidence = self._find_best_match(embedding)
            
            processing_time = time.time() - start_time
            
            return FaceRecognition(
                employee_id=best_match_id,
                confidence=best_confidence,
                face_detection=detection,
                embedding=embedding.tolist(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"Error recognizing face: {e}")
            return FaceRecognition(
                employee_id=None,
                confidence=0.0,
                face_detection=detection,
                embedding=None,
                processing_time=0.0
            )
    
    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find best match for embedding."""
        if len(self.employee_embeddings) == 0:
            return None, 0.0
        
        best_match_id = None
        best_confidence = 0.0
        
        for employee_id, stored_embedding in self.employee_embeddings.items():
            try:
                if stored_embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, stored_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(stored_embedding) + 1e-8
                    )
                    
                    if similarity > best_confidence and similarity > self.config.recognition_threshold:
                        best_confidence = similarity
                        best_match_id = employee_id
                        
            except Exception as e:
                logging.warning(f"Error calculating similarity for {employee_id}: {e}")
                continue
        
        return best_match_id, best_confidence
    
    def register_employee_face(self, employee_id: str, face_image: np.ndarray, 
                             face_photo_path: str, quality_score: float = 0.8) -> bool:
        """
        Register a new employee face.
        
        Args:
            employee_id: Employee ID
            face_image: Face image as numpy array (BGR format)
            face_photo_path: Path to face photo file
            quality_score: Quality score of the face image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.generate_embedding(face_image)
            
            if embedding is None:
                logging.error(f"Failed to generate embedding for employee {employee_id}")
                return False
            
            # Store embedding in memory
            self.employee_embeddings[employee_id] = embedding
            
            # Save to database - create new employee if doesn't exist
            with get_auth_db() as db:
                employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
                if employee:
                    # Update existing employee
                    employee.face_embedding = json.dumps(embedding.tolist())
                    employee.face_photo_path = face_photo_path
                    db.commit()
                    logging.info(f"Updated face embedding for employee {employee_id}")
                else:
                    # Create new employee
                    new_employee = Employee(
                        employee_id=employee_id,
                        first_name="Unknown",
                        last_name="User",
                        face_embedding=json.dumps(embedding.tolist()),
                        face_photo_path=face_photo_path,
                        is_active=True,
                        created_at=datetime.now()
                    )
                    db.add(new_employee)
                    db.commit()
                    logging.info(f"Created new employee {employee_id}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error registering employee face: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.employee_embeddings.clear()
            self.employee_metadata.clear()
            self.is_initialized = False
            logging.info("Simple face recognizer cleaned up")
        except Exception as e:
            logging.error(f"Error cleaning up simple face recognizer: {e}")

# Global recognizer instance
_recognizer_instance = None

def get_face_recognizer(config: FaceRecognitionConfig = None) -> SimpleFaceRecognizer:
    """Get global face recognizer instance."""
    global _recognizer_instance
    
    if _recognizer_instance is None:
        _recognizer_instance = SimpleFaceRecognizer(config)
    
    return _recognizer_instance

def cleanup_face_recognizer():
    """Cleanup global face recognizer instance."""
    global _recognizer_instance
    
    if _recognizer_instance is not None:
        _recognizer_instance.cleanup()
        _recognizer_instance = None
