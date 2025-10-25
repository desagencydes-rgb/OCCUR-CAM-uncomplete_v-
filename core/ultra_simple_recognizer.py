"""
Ultra Simple Face Recognition System
Maximum simplicity to ensure it works.
"""

import cv2
import numpy as np
import logging
import json
import time
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from models.face_models import (
    FaceDetection, FaceRecognition, FaceEmbedding, 
    FaceRecognitionConfig, FrameAnalysis
)
from config.database import get_auth_db
from database.schemas.auth_schemas import Employee

class UltraSimpleRecognizer:
    """Ultra simple face recognition system."""
    
    def __init__(self, config: FaceRecognitionConfig = None):
        """Initialize ultra simple recognizer."""
        self.config = config or FaceRecognitionConfig()
        self.is_initialized = False
        self.employee_embeddings = {}
        self.employee_metadata = {}
        
        # Recognition settings
        self.recognition_threshold = 0.3  # Very low threshold
        self.face_size = (64, 64)  # Smaller face size
        
        # Initialize the recognizer
        self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize ultra simple recognizer."""
        try:
            logging.info("Initializing ultra simple recognizer...")
            
            # Load existing employee embeddings
            self._load_employee_embeddings()
            
            self.is_initialized = True
            logging.info("Ultra simple recognizer initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing ultra simple recognizer: {e}")
            raise
    
    def _load_employee_embeddings(self):
        """Load employee embeddings from database."""
        try:
            with get_auth_db() as db:
                employees = db.query(Employee).filter(Employee.is_active == True).all()
                
                for employee in employees:
                    if employee.face_embedding:
                        try:
                            # Parse JSON embedding
                            embedding_data = json.loads(employee.face_embedding)
                            embedding = np.array(embedding_data, dtype=np.float32)
                            
                            self.employee_embeddings[employee.employee_id] = embedding
                            self.employee_metadata[employee.employee_id] = {
                                'first_name': employee.first_name,
                                'last_name': employee.last_name,
                                'email': employee.email
                            }
                            
                        except Exception as e:
                            logging.warning(f"Error loading embedding for {employee.employee_id}: {e}")
                            continue
                
                logging.info(f"Loaded {len(self.employee_embeddings)} employee embeddings")
                
        except Exception as e:
            logging.error(f"Error loading employee embeddings: {e}")
            raise
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding with ultra simple approach.
        
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
            
            # Resize to small face size
            face_resized = cv2.resize(face_image, self.face_size)
            
            # Convert to grayscale manually (avoid OpenCV issues)
            gray = np.mean(face_resized, axis=2).astype(np.uint8)
            
            # Create ultra simple embedding
            embedding = []
            
            # 1. Basic pixel statistics
            embedding.append(np.mean(gray))
            embedding.append(np.std(gray))
            embedding.append(np.var(gray))
            embedding.append(np.median(gray))
            embedding.append(np.min(gray))
            embedding.append(np.max(gray))
            
            # 2. Simple histogram
            hist, _ = np.histogram(gray.ravel(), bins=32, range=(0, 256))
            embedding.extend(hist)
            
            # 3. Simple spatial features (2x2 grid)
            h, w = gray.shape
            for i in range(2):
                for j in range(2):
                    y1, y2 = i * h // 2, (i + 1) * h // 2
                    x1, x2 = j * w // 2, (j + 1) * w // 2
                    region = gray[y1:y2, x1:x2]
                    if region.size > 0:
                        embedding.append(np.mean(region))
                        embedding.append(np.std(region))
                    else:
                        embedding.extend([0, 0])
            
            # 4. Simple edge features
            try:
                # Simple gradient
                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                embedding.append(np.mean(grad_x))
                embedding.append(np.mean(grad_y))
            except:
                embedding.extend([0, 0])
            
            # 5. Image hash for uniqueness
            image_hash = hashlib.md5(face_image.tobytes()).hexdigest()
            hash_values = [int(image_hash[i:i+2], 16) / 255.0 for i in range(0, len(image_hash), 2)]
            embedding.extend(hash_values)
            
            # Convert to numpy array
            embedding = np.array(embedding, dtype=np.float32)
            
            # Add unique noise based on image content
            np.random.seed(int(image_hash[:8], 16))
            noise = np.random.normal(0, 0.5, len(embedding))
            embedding = embedding + noise
            
            # Normalize the embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Pad or truncate to standard size (256)
            if len(embedding) < 256:
                padding = np.random.normal(0, 0.1, 256 - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > 256:
                embedding = embedding[:256]
            
            # Final normalization
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            logging.debug(f"Generated ultra simple embedding with shape: {embedding.shape}")
            return embedding.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error generating face embedding: {e}")
            return None
    
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
        """Find best match for embedding using cosine similarity."""
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
                    
                    # Use very low threshold
                    if similarity > best_confidence and similarity > self.recognition_threshold:
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
            
            # Save to database
            with get_auth_db() as db:
                employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
                if employee:
                    employee.face_embedding = json.dumps(embedding.tolist())
                    employee.face_photo_path = face_photo_path
                    db.commit()
                else:
                    logging.error(f"Employee {employee_id} not found in database")
                    return False
            
            logging.info(f"Successfully registered face for employee {employee_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error registering employee face: {e}")
            return False
    
    def remove_employee_face(self, employee_id: str) -> bool:
        """Remove employee face from recognition database."""
        try:
            # Remove from memory
            if employee_id in self.employee_embeddings:
                del self.employee_embeddings[employee_id]
            
            if employee_id in self.employee_metadata:
                del self.employee_metadata[employee_id]
            
            # Remove from database
            with get_auth_db() as db:
                employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
                if employee:
                    employee.face_embedding = None
                    employee.face_photo_path = None
                    db.commit()
            
            logging.info(f"Successfully removed face for employee {employee_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error removing employee face: {e}")
            return False
    
    def get_employee_info(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Get employee information."""
        return self.employee_metadata.get(employee_id)
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """Get recognition statistics."""
        return {
            "total_employees": len(self.employee_embeddings),
            "recognition_threshold": self.recognition_threshold,
            "is_initialized": self.is_initialized
        }
    
    def refresh_employee_cache(self):
        """Refresh employee database from storage."""
        try:
            self._load_employee_embeddings()
            logging.info("Employee cache refreshed")
        except Exception as e:
            logging.error(f"Error refreshing employee cache: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        self.employee_embeddings.clear()
        self.employee_metadata.clear()
        self.is_initialized = False
        logging.info("Ultra simple recognizer cleaned up")

# Global instance
_ultra_simple_recognizer = None

def get_ultra_simple_recognizer(config: FaceRecognitionConfig = None) -> UltraSimpleRecognizer:
    """Get global ultra simple recognizer instance."""
    global _ultra_simple_recognizer
    if _ultra_simple_recognizer is None:
        _ultra_simple_recognizer = UltraSimpleRecognizer(config)
    return _ultra_simple_recognizer

def cleanup_ultra_simple_recognizer():
    """Cleanup global ultra simple recognizer instance."""
    global _ultra_simple_recognizer
    if _ultra_simple_recognizer:
        _ultra_simple_recognizer.cleanup()
        _ultra_simple_recognizer = None

